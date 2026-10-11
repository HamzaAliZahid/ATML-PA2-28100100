from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def beta_tag(b: float) -> str:
    return f"beta{b:g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="*", help="override the beta list (default from config)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse adapters/evals already on disk")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n)
    results_dir = repo_path(cfg["results_dir"])

    summary = {"budget": {"examples": n, "epochs": int(cfg["epochs"]),
                          "note": "short fork budget; the standard run uses the full training set"},
               "conditions": {}}
    for b in betas:
        tag = beta_tag(b)
        out = f"outputs/task1_dpo/{tag}"
        if not (args.skip_existing and (repo_path(out) / "adapter_config.json").exists()):
            # Same initialization, data subset (first n rows), data order, optimizer, seed and LoRA config;
            # only beta changes.
            run_training(args.config, run_name=tag, output_path=out, beta=b, max_examples=n)
        eval_file = results_dir / f"eval_{tag}.json"
        if args.skip_existing and eval_file.exists():
            res = load_json(eval_file)
        else:
            # Held-out loss is reported at the condition's own beta; accuracy/margin are beta-free.
            res = run_evaluation(cfg, out, tag, beta=b)
        summary["conditions"][tag] = {"beta": b, "adapter": out, **res}
    save_json(results_dir / "beta_sweep_summary.json", summary)
    print("Saved", results_dir / "beta_sweep_summary.json")


if __name__ == "__main__":
    main()
