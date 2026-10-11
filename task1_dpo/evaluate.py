from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.rl_utils import evaluate_generations, length_stats


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def preference_eval(model, tokenizer, rows, beta, max_length, batch_size=4, stratum_key=None):
    """Held-out DPO margin m_theta, preference accuracy (m_theta > 0) and DPO loss.

    Sequence log-probs are summed over response tokens only (response_sequence_logprobs).
    """
    device = next(model.parameters()).device
    records, skipped = [], 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        enc_c, enc_r, kept = [], [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                enc_c.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
                enc_r.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
                kept.append(row)
            except ValueError:
                skipped += 1
        if not kept:
            continue
        bc = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_c).items()}
        br = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_r).items()}
        pc, _, mc = response_sequence_logprobs(model, bc)
        pr, _, mr = response_sequence_logprobs(model, br)
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, bc)
            rr, _, _ = response_sequence_logprobs(model, br)
        margin = (pc - rc) - (pr - rr)
        loss = -F.logsigmoid(beta * margin)
        for i, row in enumerate(kept):
            records.append({
                "source_index": row.get("source_index"),
                "prompt_id": row.get("prompt_id"),
                "stratum": row.get(stratum_key) if stratum_key else None,
                "margin": float(margin[i].item()),
                "dpo_loss": float(loss[i].item()),
                "correct": bool(margin[i].item() > 0),
                "policy_chosen_logp": float(pc[i].item()),
                "policy_rejected_logp": float(pr[i].item()),
                "ref_chosen_logp": float(rc[i].item()),
                "ref_rejected_logp": float(rr[i].item()),
                "chosen_reward": float(beta * (pc[i] - rc[i]).item()),
                "rejected_reward": float(beta * (pr[i] - rr[i]).item()),
                "chosen_tokens": int(mc[i].sum().item()),
                "rejected_tokens": int(mr[i].sum().item()),
            })
        print(f"  pref-eval {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    def agg(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "preference_accuracy": float(np.mean([r["correct"] for r in rs])),
            "dpo_loss": float(np.mean([r["dpo_loss"] for r in rs])),
            "margin_mean": float(np.mean([r["margin"] for r in rs])),
            "margin_std": float(np.std([r["margin"] for r in rs])),
            "chosen_reward_mean": float(np.mean([r["chosen_reward"] for r in rs])),
            "rejected_reward_mean": float(np.mean([r["rejected_reward"] for r in rs])),
            # Diagnostic: fraction of pairs in which the chosen response is the longer one.
            "chosen_longer_fraction": float(np.mean([r["chosen_tokens"] > r["rejected_tokens"] for r in rs])),
        }

    metrics = {"overall": agg(records), "skipped_too_long": skipped, "beta_used_for_loss": beta}
    if stratum_key:
        by = defaultdict(list)
        for r in records:
            by[r["stratum"]].append(r)
        metrics["by_stratum"] = {k: agg(v) for k, v in sorted(by.items())}
    return metrics, records


def generation_prompt_rows(cfg):
    """Fixed generation set: the first N held-out DPO prompts (same for every condition)."""
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    n = int(cfg.get("eval_generation_prompts", 64))
    out = []
    for r in rows[:n]:
        out.append({"source_index": r.get("source_index"), "messages": prompt_messages_from_preference(r)})
    return out


def run_evaluation(cfg, adapter, name, pref_path=None, stratum_key=None, beta=None, tag=None, do_generation=True):
    """Evaluate one DPO condition and write results/task1_dpo/eval_<name>[_<tag>].json."""
    beta = float(cfg["beta"] if beta is None else beta)
    results_dir = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    out_name = name if tag is None else f"{name}_{tag}"
    bs = int(cfg.get("eval_batch_size", 4))

    pref_rows = read_jsonl(pref_path or cfg["paths"]["dpo_standard_eval"])
    pref_metrics, pref_records = preference_eval(
        policy, tokenizer, pref_rows, beta, int(cfg["max_sequence_length"]), bs, stratum_key)
    write_jsonl(results_dir / f"eval_{out_name}_pairs.jsonl", pref_records)
    payload = {"name": out_name, "adapter": adapter, "beta": beta,
               "preference_eval_file": pref_path or cfg["paths"]["dpo_standard_eval"],
               "preference": pref_metrics}

    if do_generation:
        rm, rm_tok = load_reward_model(cfg)
        gen_kw = dict(
            max_prompt_length=int(cfg.get("max_prompt_length", 512)),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            batch_size=bs, seed=int(cfg["seed"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        gm, grec = evaluate_generations(policy, tokenizer, rm, rm_tok, generation_prompt_rows(cfg), **gen_kw)
        write_jsonl(results_dir / f"eval_{out_name}_generations.jsonl", grec)
        payload["generation"] = gm

        # Word-limit compliance on the common prompt set: greedy decoding so the
        # instruction-following check is deterministic.
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wkw = dict(gen_kw, do_sample=False, temperature=1.0, top_p=1.0)
        wm, wrec = evaluate_generations(policy, tokenizer, rm, rm_tok, wl_rows, **wkw)
        from common.metrics import word_count, parse_word_limit
        for r in wrec:
            r["word_limit"] = parse_word_limit(r["prompt"])
            r["word_count"] = word_count(r["response"])
        write_jsonl(results_dir / f"eval_{out_name}_wordlimit.jsonl", wrec)
        payload["word_limit"] = {
            "compliance": wm.get("word_limit_compliance"),
            "n": wm.get("word_limit_n"),
            "response_length": wm["response_length"],
            "word_count": length_stats([r["word_count"] for r in wrec]),
            "mean_words_over_limit": float(np.mean([max(0, r["word_count"] - (r["word_limit"] or 0)) for r in wrec])),
        }
        clear_gpu(rm)

    save_json(results_dir / f"eval_{out_name}.json", payload)
    print(f"Saved {results_dir / f'eval_{out_name}.json'}")
    clear_gpu(policy)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT starting policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: config beta)")
    ap.add_argument("--no-generation", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    adapter = None if args.adapter.lower() == "none" else args.adapter
    run_evaluation(cfg, adapter, args.name, beta=args.beta, do_generation=not args.no_generation)


if __name__ == "__main__":
    main()
