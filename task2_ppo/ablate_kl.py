from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.rl_eval import evaluate_rl_policy
from task2_ppo.analyze_clipping import stability_stats
from task2_ppo.continue_train import run_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    results_dir = repo_path(cfg["results_dir"])
    summary = {}
    for b in [float(x) for x in cfg["kl_values"]]:
        tag = f"kl_beta{b:g}".replace(".", "p")
        adapter = f"outputs/task2_ppo/{tag}"
        if not (args.skip_existing and (repo_path(adapter) / "adapter_config.json").exists()):
            # Same midpoint policy+critic, prompt schedule, eps, seed and update budget; only beta_KL differs.
            run_ppo(args.config, output=adapter, updates=int(cfg["fork_updates"]), kl_beta=b, run_name=tag)
        ev = evaluate_rl_policy(cfg, adapter, tag, skip_existing=args.skip_existing)
        logs = read_jsonl(results_dir / f"train_{tag}_log.jsonl")
        summary[tag] = {
            "kl_beta": b,
            "heldout": ev,
            "train_summary": load_json(results_dir / f"train_{tag}_summary.json"),
            "stability": stability_stats(results_dir / f"train_{tag}_log.jsonl"),
            # Per-update trajectories so you can see which quantity moves first.
            "trajectory": {k: [r[k] for r in logs] for k in
                           ["reward_raw_mean", "kl_ref_token", "entropy", "response_length", "value_loss",
                            "clip_fraction", "policy_grad_norm"]},
        }
    save_json(results_dir / "kl_study_summary.json", summary)
    print("Saved", results_dir / "kl_study_summary.json")


if __name__ == "__main__":
    main()
