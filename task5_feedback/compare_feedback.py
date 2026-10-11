from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = load_json(outdir / "math_eval_gsm.json")
    tr = load_json(outdir / "math_eval_transfer.json")

    rows = []
    for p in ["sft", "rlvr", "rlaif"]:
        g, t = gsm["policies"][p], tr["policies"][p]
        row = {
            "policy": p,
            "gsm_exact_acc": g["exact_accuracy"], "gsm_format": g["format_compliance"],
            "gsm_len_mean": g["response_length"]["mean"], "gsm_len_std": g["response_length"]["std"],
            "svamp_exact_acc": t["exact_accuracy"], "svamp_format": t["format_compliance"],
            "svamp_len_mean": t["response_length"]["mean"], "svamp_len_std": t["response_length"]["std"],
            "acc_drop_gsm_to_svamp": g["exact_accuracy"] - t["exact_accuracy"],
        }
        for name, d in [("gsm", g), ("svamp", t)]:
            pw = d.get("pairwise_vs_sft")
            if pw:
                row[f"{name}_win_vs_sft"] = pw["win_rate_vs_sft"]
                row[f"{name}_judge_ties"] = pw["ties"]
                row[f"{name}_verifier_judge_agree"] = pw["verifier_judge_agreement_all"]
                row[f"{name}_judge_agree_on_decisive"] = pw["judge_agrees_on_verifier_decisive"]
        if "gsm_win_vs_sft" in row and "svamp_win_vs_sft" in row:
            row["win_drop_gsm_to_svamp"] = row["gsm_win_vs_sft"] - row["svamp_win_vs_sft"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("policy")
    table.to_csv(outdir / "feedback_comparison.csv")
    print(table.round(3).T.to_string())

    out = {"in_domain_vs_transfer": table.reset_index().to_dict(orient="records")}
    diag = outdir / "diagnostic_study.json"
    if diag.exists():
        d = load_json(diag)
        out["diagnostics"] = {k: d[k] for k in ["S_reason", "S_outcome", "S_outcome_incl_distractor",
                                                "filler_susceptibility_ai_judge", "per_variant_mean_signal"]}
        dt = []
        for kind, v in d["by_pair"].items():
            for mech in ["verifier", "ai_judge"]:
                dt.append({"pair": kind, "mechanism": mech, **v[mech]})
        pd.DataFrame(dt).to_csv(outdir / "diagnostic_table.csv", index=False)
        print(pd.DataFrame(dt).round(3).to_string(index=False))
    # Inference-cost proxy: judge calls are cached, so count cache entries
    cache = outdir / "pairwise_judge_cache.json"
    if cache.exists():
        out["ai_judge_calls_cached"] = len(load_json(cache))
    save_json(outdir / "feedback_comparison.json", out)
    print("Saved", outdir / "feedback_comparison.json")


if __name__ == "__main__":
    main()
