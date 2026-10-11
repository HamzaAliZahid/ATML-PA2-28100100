from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training

STRATUM_KEY = "length_stratum"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="reuse the length-balanced adapter on disk")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    results_dir = repo_path(cfg["results_dir"])

    standard = cfg["standard_output"]
    if not (repo_path(standard) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Train standard DPO first: {standard}")

    balanced_out = cfg["length_output"]
    if not (args.skip_train and (repo_path(balanced_out) / "adapter_config.json").exists()):
        # Same init, beta, optimizer, seed, LoRA and epoch count as standard; only the training file differs.
        run_training(args.config, run_name="length_balanced",
                     dataset_path=cfg["paths"]["dpo_length_train"], output_path=balanced_out)

    # Dataset-level length structure (a property of the data, not of a policy).
    def data_profile(rows):
        import numpy as np
        out = {}
        for s in sorted({r.get(STRATUM_KEY) for r in rows}):
            rs = [r for r in rows if r.get(STRATUM_KEY) == s]
            out[s] = {"n": len(rs),
                      "chosen_tokens_mean": float(np.mean([r["chosen_tokens"] for r in rs])),
                      "rejected_tokens_mean": float(np.mean([r["rejected_tokens"] for r in rs])),
                      "length_difference_mean": float(np.mean([r["length_difference"] for r in rs]))}
        return out

    std_train = read_jsonl(cfg["paths"]["dpo_standard_train"])
    summary = {
        "data_profile": {
            "length_balanced_train": data_profile(balanced),
            "length_stratified_eval": data_profile(stratified),
        },
        "conditions": {},
    }
    # Standard-train chosen-vs-rejected length correlation (tokenized here, same tokenizer).
    from common.models import load_tokenizer
    from common.data import preference_responses
    import numpy as np
    tok = load_tokenizer(cfg["base_model"])
    diffs = []
    for r in std_train:
        yc, yr = preference_responses(r)
        diffs.append(len(tok(yc, add_special_tokens=False)["input_ids"]) - len(tok(yr, add_special_tokens=False)["input_ids"]))
    diffs = np.asarray(diffs)
    summary["data_profile"]["standard_train"] = {
        "n": int(diffs.size),
        "chosen_longer_fraction": float((diffs > 0).mean()),
        "length_difference_mean": float(diffs.mean()),
        "length_difference_median": float(np.median(diffs)),
    }

    from common.logging_utils import load_json
    std_eval = results_dir / "eval_standard.json"
    for name, adapter in [("standard", standard), ("length_balanced", balanced_out)]:
        # The standard model's generation/word-limit results are identical to its main evaluation,
        # so reuse them when available instead of regenerating.
        reuse = name == "standard" and std_eval.exists()
        res = run_evaluation(cfg, adapter, name, pref_path=cfg["paths"]["dpo_length_eval"],
                             stratum_key=STRATUM_KEY, tag="lengthstrat", do_generation=not reuse)
        if reuse:
            prev = load_json(std_eval)
            res["generation"] = prev.get("generation")
            res["word_limit"] = prev.get("word_limit")
        summary["conditions"][name] = res
    save_json(results_dir / "length_study_summary.json", summary)
    print("Saved", results_dir / "length_study_summary.json")


if __name__ == "__main__":
    main()
