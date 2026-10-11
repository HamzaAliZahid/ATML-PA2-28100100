from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from common.rl_utils import length_stats
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_math(cfg, rows, tokenizer, name, batch_size):
    """Deterministic (greedy) responses with the common math generation cap."""
    model = load_frozen_policy(cfg, name)
    out = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        gen = batch_generate(model, tokenizer, [prompt_messages(r) for r in chunk],
                             max_prompt_length=512, max_new_tokens=int(cfg["math_max_new_tokens"]),
                             temperature=1.0, top_p=1.0, do_sample=False)
        for r, resp, n, tr in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(resp)
            out.append({
                "source_index": r.get("source_index"), "policy": name,
                "question": r.get("question"), "gold_final": str(r["gold_final"]),
                "response": resp, "response_tokens": int(n), "truncated": bool(tr),
                "pred_final": pred, "format_ok": pred is not None,
                "exact_correct": bool(exact_reward(resp, str(r["gold_final"]))),
                "svamp_type": r.get("svamp_type"),
            })
        print(f"  {name}: {min(s + batch_size, len(rows))}/{len(rows)}", flush=True)
    clear_gpu(model)
    return out


def pairwise_vs_sft(judge, gens, name):
    """AI pairwise result of `name` vs SFT on every problem (win=1, tie=0.5, loss=0), plus
    verifier-vs-judge agreement on the same pairs."""
    sft = {g["source_index"]: g for g in gens["sft"]}
    recs = []
    for g in gens[name]:
        s = sft[g["source_index"]]
        pref = judge.compare(g["question"], g["response"], s["response"])  # A = policy, B = SFT
        score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
        v_a, v_b = float(g["exact_correct"]), float(s["exact_correct"])
        v_pref = "A" if v_a > v_b else ("B" if v_b > v_a else "TIE")
        recs.append({"source_index": g["source_index"], "judge_pref": pref, "score": score,
                     "verifier_pref": v_pref, "agree": pref == v_pref})
    decisive = [r for r in recs if r["verifier_pref"] != "TIE"]
    v_ties = [r for r in recs if r["verifier_pref"] == "TIE"]
    return {
        "win_rate_vs_sft": float(np.mean([r["score"] for r in recs])),
        "wins": sum(r["judge_pref"] == "A" for r in recs), "losses": sum(r["judge_pref"] == "B" for r in recs),
        "ties": sum(r["judge_pref"] == "TIE" for r in recs), "n": len(recs),
        "verifier_judge_agreement_all": float(np.mean([r["agree"] for r in recs])),
        "verifier_decisive_pairs": len(decisive),
        "judge_agrees_on_verifier_decisive": float(np.mean([r["agree"] for r in decisive])) if decisive else float("nan"),
        "judge_tie_rate_when_verifier_ties": float(np.mean([r["judge_pref"] == "TIE" for r in v_ties])) if v_ties else float("nan"),
    }, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    ap.add_argument("--max-examples", type=int, help="fixed prefix of the eval file (default: all rows)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gens = {}
    for name in policy_specs(cfg):
        path = outdir / f"generated_{args.dataset}_{name}.jsonl"
        if args.stage in ("generate", "all") and not (args.skip_existing and path.exists()):
            write_jsonl(path, generate_math(cfg, rows, tok, name, args.batch_size))
        if path.exists():
            gens[name] = read_jsonl(path)
    if args.stage == "generate":
        return

    summary = {"dataset": args.dataset, "n": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]), "policies": {}}
    for name, g in gens.items():
        summary["policies"][name] = {
            "exact_accuracy": float(np.mean([r["exact_correct"] for r in g])),
            "format_compliance": float(np.mean([r["format_ok"] for r in g])),
            "accuracy_given_format": float(np.mean([r["exact_correct"] for r in g if r["format_ok"]])) if any(r["format_ok"] for r in g) else float("nan"),
            "truncated_fraction": float(np.mean([r["truncated"] for r in g])),
            "response_length": length_stats([r["response_tokens"] for r in g]),
        }
        if args.dataset == "transfer":
            by = {}
            for r in g:
                by.setdefault(r.get("svamp_type"), []).append(r["exact_correct"])
            summary["policies"][name]["accuracy_by_svamp_type"] = {k: float(np.mean(v)) for k, v in by.items()}

    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
    for name in ["rlvr", "rlaif"]:
        res, recs = pairwise_vs_sft(judge, gens, name)
        summary["policies"][name]["pairwise_vs_sft"] = res
        write_jsonl(outdir / f"pairwise_{args.dataset}_{name}_vs_sft.jsonl", recs)
        print(f"{name}: acc={summary['policies'][name]['exact_accuracy']:.3f} "
              f"win_vs_sft={res['win_rate_vs_sft']:.3f} ties={res['ties']}")
    print(f"sft: acc={summary['policies']['sft']['exact_accuracy']:.3f}")
    save_json(outdir / f"math_eval_{args.dataset}.json", summary)
    print("Saved", outdir / f"math_eval_{args.dataset}.json")


if __name__ == "__main__":
    main()
