from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.rl_eval import evaluate_rl_policy
from task3_grpo.continue_train import run_grpo


def length_conditioned(comp_rows, edges):
    """Gradient/advantage statistics per response-length bin (bins shared by both conditions)."""
    out = {}
    names = ["short", "medium", "long"]
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        rs = [r for r in comp_rows if not r["masked"] and lo <= r["length"] < hi]
        if not rs:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": len(rs), "length_range": [float(lo), float(hi)],
            "grad_norm_mean": float(np.mean([r["grad_norm"] for r in rs])),
            "per_token_weight_mean": float(np.mean([r["per_token_weight"] for r in rs])),
            "abs_advantage_mean": float(np.mean([abs(r["advantage"]) for r in rs])),
            "grad_norm_per_abs_adv": float(np.mean([r["grad_norm"] / max(abs(r["advantage"]), 1e-6) for r in rs])),
            "positive_adv_fraction": float(np.mean([r["advantage"] > 0 for r in rs])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    results_dir = repo_path(cfg["results_dir"])

    conds = {}
    for lt in ["grpo", "dr_grpo"]:
        tag = f"norm_{lt}"
        adapter = f"outputs/task3_grpo/{tag}"
        if not (args.skip_existing and (repo_path(adapter) / "adapter_config.json").exists()):
            # identical midpoint, prompt schedule, K, reward, beta, eps, generation settings, update budget
            run_grpo(args.config, output=adapter, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=tag)
        conds[lt] = {
            "tag": tag,
            "heldout": evaluate_rl_policy(cfg, adapter, tag, skip_existing=args.skip_existing),
            "train_summary": load_json(results_dir / f"train_{tag}_summary.json"),
            "log": read_jsonl(results_dir / f"train_{tag}_log.jsonl"),
            "completions": read_jsonl(results_dir / f"train_{tag}_completions.jsonl"),
        }

    # Shared length bins: terciles of training-completion length pooled over both conditions.
    pooled = [r["length"] for c in conds.values() for r in c["completions"] if not r["masked"]]
    q = np.quantile(pooled, [1 / 3, 2 / 3])
    edges = [0, float(q[0]), float(q[1]), float(max(pooled)) + 1]

    summary = {"length_bins": edges, "bin_rule": "terciles of pooled training-completion lengths", "conditions": {}}
    for lt, c in conds.items():
        comps = [r for r in c["completions"] if not r["masked"]]
        summary["conditions"][lt] = {
            "heldout": c["heldout"],
            "train_summary": c["train_summary"],
            "train_response_length_trajectory": [r["response_length"] for r in c["log"]],
            "train_reward_trajectory": [r["reward_mean"] for r in c["log"]],
            "train_kl_trajectory": [r["kl_ref_token"] for r in c["log"]],
            "corr_length_gradnorm": safe_corr([r["length"] for r in comps], [r["grad_norm"] for r in comps]),
            "corr_length_advantage": safe_corr([r["length"] for r in comps], [r["advantage"] for r in comps]),
            "length_conditioned": length_conditioned(comps, edges),
            "masked_truncated_completions": int(sum(r["masked"] for r in c["completions"])),
        }
    path = results_dir / "normalization_study.json"
    save_json(path, summary)
    print("Saved", path)


if __name__ == "__main__":
    main()
