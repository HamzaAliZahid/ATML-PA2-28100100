"""Shared helpers for training loops and the common held-out evaluation protocol.

Everything here is task-agnostic. Task-specific logic stays inside each task folder.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from common.data import prompt_messages, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl, word_limit_compliance
from common.models import reference_mode


# ----------------------------------------------------------------------------------------
# Training-time helpers
# ----------------------------------------------------------------------------------------

def disable_dropout(model: nn.Module) -> None:
    """Set every dropout probability to 0.

    Needed for on-policy RL: with dropout active, log pi_theta(a|s) computed in the update
    pass would differ from the rollout-time log-prob even before any parameter update, so the
    PPO/GRPO importance ratio would not start at 1. Train mode is kept (gradient checkpointing
    in HF is only active in train mode).
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        # PEFT LoRA keeps its dropout in a ModuleDict of nn.Dropout -> covered above.


def cast_trainable_to_fp32(model: nn.Module) -> None:
    """Keep trainable parameters (LoRA / value head) in fp32 so AdamW is numerically safe."""
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def masked_full_entropy(logits: torch.Tensor, mask: torch.Tensor, chunk: int = 128) -> float:
    """Mean token-level entropy H(pi(.|s_t)) over valid response tokens (no grad, chunked)."""
    with torch.no_grad():
        total, count = 0.0, 0.0
        T = logits.shape[1]
        for s in range(0, T, chunk):
            lg = logits[:, s:s + chunk].float()
            logp = torch.log_softmax(lg, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            m = mask[:, s:s + chunk].to(ent.dtype)
            total += float((ent * m).sum().item())
            count += float(m.sum().item())
            del lg, logp, ent
        return total / max(count, 1.0)


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    return 0.0


def length_stats(lengths) -> dict:
    a = np.asarray(lengths, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "iqr": float("nan"), "n": 0}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "iqr": float(q3 - q1),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def summarize(values) -> dict:
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


# ----------------------------------------------------------------------------------------
# Common held-out generation evaluation
# ----------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_generations(
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows: list[dict],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int = 4,
    seed: int = 6304,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    reward_max_length: int = 1280,
    id_key_candidates=("source_index", "prompt_id", "id", "idx"),
    compute_full_entropy: bool = True,
):
    """Generate one response per prompt and compute reward / KL / entropy / length.

    KL uses the released sampled-response estimator (common.metrics.sampled_kl): the
    response-token log-prob difference between the evaluated policy and the frozen reference
    (the same base model with LoRA adapters disabled). We report two aggregations, always the
    same for every condition:
      * kl_token_mean   : sampled_kl over all valid tokens in the eval set (course helper)
      * kl_seq_mean     : mean over responses of the summed per-token log-ratio
    """
    set_seed(seed)  # identical sampling stream for every evaluated condition
    records = []
    device = next(policy.parameters()).device
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        gen = batch_generate(
            policy, tokenizer, prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        seq, attn, pw, rid, rmask = (gen["sequences"], gen["attention_mask"], gen["prompt_width"],
                                      gen["response_ids"], gen["response_mask"])
        pol_logp, logits = response_token_logprobs(policy, seq, attn, pw, rid)
        if compute_full_entropy:
            ent_full = [masked_full_entropy(logits[i:i + 1], rmask[i:i + 1]) for i in range(len(chunk))]
        else:
            ent_full = [float("nan")] * len(chunk)
        del logits
        with reference_mode(policy):
            ref_logp, ref_logits = response_token_logprobs(policy, seq, attn, pw, rid)
        del ref_logits
        rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts, gen["responses"],
                                     max_length=reward_max_length).cpu().tolist()
        for i, row in enumerate(chunk):
            m = rmask[i]
            n = float(m.sum().item())
            lr = (pol_logp[i] - ref_logp[i]) * m
            rid_val = next((row[k] for k in id_key_candidates if k in row), start + i)
            records.append({
                "prompt_id": rid_val,
                "prompt": prompts[i][-1]["content"] if prompts[i] else "",
                "response": gen["responses"][i],
                "response_tokens": int(n),
                "truncated": bool(gen["truncated"][i]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "reward": float(rewards[i]),
                "kl_seq": float(lr.sum().item()),
                "kl_token_sum": float(lr.sum().item()),
                "n_tokens": n,
                "sample_entropy": float(sample_entropy(pol_logp[i:i + 1], m[None]).item()),
                "full_entropy": ent_full[i],
                "word_limit_compliant": word_limit_compliance(prompts[i][-1]["content"], gen["responses"][i])
                if prompts[i] else None,
            })
        del pol_logp, ref_logp, seq, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  eval {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    total_tokens = sum(r["n_tokens"] for r in records)
    metrics = {
        "n_prompts": len(records),
        "reward": summarize([r["reward"] for r in records]),
        "kl_token_mean": float(sum(r["kl_token_sum"] for r in records) / max(total_tokens, 1.0)),
        "kl_seq_mean": summarize([r["kl_seq"] for r in records])["mean"],
        "kl_seq_std": summarize([r["kl_seq"] for r in records])["std"],
        "sample_entropy": summarize([r["sample_entropy"] for r in records])["mean"],
        "full_entropy": summarize([r["full_entropy"] for r in records])["mean"],
        "response_length": length_stats([r["response_tokens"] for r in records]),
        "truncated_fraction": float(np.mean([r["truncated"] for r in records])) if records else float("nan"),
        "decoding": {"temperature": temperature, "top_p": top_p, "do_sample": do_sample,
                     "max_new_tokens": max_new_tokens, "seed": seed},
    }
    wl = [r["word_limit_compliant"] for r in records if r["word_limit_compliant"] is not None]
    if wl:
        metrics["word_limit_compliance"] = float(np.mean(wl))
        metrics["word_limit_n"] = len(wl)
    return metrics, records


def save_eval(out_dir, name: str, metrics: dict, records: list[dict], extra: dict | None = None):
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    payload["name"] = name
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(f"{out_dir}/eval_{name}.json", payload)
    write_jsonl(f"{out_dir}/eval_{name}_generations.jsonl", records)
    print(f"Saved {out_dir}/eval_{name}.json")
