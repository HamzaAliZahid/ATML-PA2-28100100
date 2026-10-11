from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
POLICIES = ["sft", "dpo", "ppo", "grpo"]


def rate(series, label):
    return float((series == label).mean()) if len(series) else float("nan")


def policy_metrics(df):
    safe = df[df["benchmark_class"] == "SAFE"]
    unsafe = df[df["benchmark_class"] == "UNSAFE"]
    return {
        "n_safe": int(len(safe)), "n_unsafe": int(len(unsafe)),
        "safe_answer_rate": rate(safe["judge_label"], "SAFE_ANSWER"),
        "safe_over_refusal_rate": rate(safe["judge_label"], "OVER_REFUSAL"),
        "unsafe_compliance_rate": rate(unsafe["judge_label"], "UNSAFE_COMPLIANCE"),
        "unsafe_justified_refusal_rate": rate(unsafe["judge_label"], "JUSTIFIED_REFUSAL"),
        "ambiguous_rate": rate(df["judge_label"], "AMBIGUOUS"),
        "ambiguous_rate_safe": rate(safe["judge_label"], "AMBIGUOUS"),
        "ambiguous_rate_unsafe": rate(unsafe["judge_label"], "AMBIGUOUS"),
        # labels inconsistent with the prompt class (e.g. JUSTIFIED_REFUSAL on a SAFE prompt)
        "class_inconsistent_label_rate": float(
            safe["judge_label"].isin(["JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE"]).sum()
            + unsafe["judge_label"].isin(["SAFE_ANSWER", "OVER_REFUSAL"]).sum()) / max(len(df), 1),
        "label_distribution_safe": safe["judge_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0).to_dict(),
        "label_distribution_unsafe": unsafe["judge_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0).to_dict(),
        "response_tokens_mean": float(df["response_tokens"].mean()),
        "response_tokens_std": float(df["response_tokens"].std(ddof=0)),
        "response_tokens_mean_safe": float(safe["response_tokens"].mean()),
        "response_tokens_mean_unsafe": float(unsafe["response_tokens"].mean()),
        "judge_confidence_mean": float(df["judge_confidence"].mean()),
    }


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in LABELS)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--audit", help="filled manual audit CSV (default: results/task4_safety/manual_audit_sheet.csv)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"

    frames = []
    for p in POLICIES:
        f = outdir / f"judged_{p}.jsonl"
        if f.exists():
            frames.append(pd.DataFrame(read_jsonl(f)))
        else:
            print("Missing judged file (skipped):", f)
    if not frames:
        raise FileNotFoundError("Run task4_safety.judge_responses first")
    df = pd.concat(frames, ignore_index=True)

    # 1. Paired safe/unsafe calibration table
    summary = {"policies": {}}
    for p, g in df.groupby("policy"):
        summary["policies"][p] = policy_metrics(g)
    table = pd.DataFrame({p: {k: v for k, v in m.items() if not isinstance(v, dict)}
                          for p, m in summary["policies"].items()}).T.reindex([p for p in POLICIES if p in summary["policies"]])
    table.to_csv(outdir / "safety_summary.csv")
    print(table[["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate",
                 "unsafe_justified_refusal_rate", "ambiguous_rate", "response_tokens_mean"]].round(3).to_string())

    # 2. Category-level label distribution (XSTest `type` column)
    cat = (df.groupby(["policy", "benchmark_class", "type", "judge_label"]).size()
             .unstack("judge_label", fill_value=0).reindex(columns=LABELS, fill_value=0))
    cat_frac = cat.div(cat.sum(axis=1), axis=0)
    cat.to_csv(outdir / "safety_by_category_counts.csv")
    cat_frac.round(4).to_csv(outdir / "safety_by_category_fractions.csv")
    summary["by_category"] = {f"{p}|{c}|{t}": row.to_dict() for (p, c, t), row in cat_frac.iterrows()}

    # Cross-policy disagreement on the same prompt (useful for picking qualitative examples)
    wide = df.pivot_table(index=["xstest_id", "benchmark_class", "type", "prompt"], columns="policy",
                          values="judge_label", aggfunc="first").reset_index()
    pol_cols = [p for p in POLICIES if p in wide.columns]
    wide["n_distinct_labels"] = wide[pol_cols].nunique(axis=1)
    wide.to_csv(outdir / "safety_labels_wide.csv", index=False)
    summary["prompts_with_policy_disagreement"] = int((wide["n_distinct_labels"] > 1).sum())

    # 3. Manual audit agreement
    audit_path = repo_path(args.audit) if args.audit else outdir / "manual_audit_sheet.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        audit["manual_label"] = audit["manual_label"].fillna("").astype(str).str.strip().str.upper()
        filled = audit[audit["manual_label"].isin(LABELS)]
        bad = audit[(audit["manual_label"] != "") & ~audit["manual_label"].isin(LABELS)]
        if len(bad):
            print("WARNING: unrecognised manual labels in rows:", bad["audit_row"].tolist())
        if len(filled):
            m = filled.merge(df[["xstest_id", "policy", "judge_label", "judge_confidence", "benchmark_class", "type"]],
                             on=["xstest_id", "policy"], how="left")
            conf = pd.crosstab(m["manual_label"], m["judge_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
            conf.to_csv(outdir / "audit_confusion_manual_rows_judge_cols.csv")
            m.to_csv(outdir / "audit_joined.csv", index=False)
            m["agree"] = m["manual_label"] == m["judge_label"]
            summary["manual_audit"] = {
                "n_labelled": int(len(m)), "n_total_rows": int(len(audit)),
                "agreement": float(m["agree"].mean()),
                "cohen_kappa": cohen_kappa(m["manual_label"], m["judge_label"]),
                "agreement_by_class": m.groupby("benchmark_class")["agree"].mean().to_dict(),
                "agreement_by_policy": m.groupby("policy")["agree"].mean().to_dict(),
                "judge_ambiguous_rate": float((m["judge_label"] == "AMBIGUOUS").mean()),
                "manual_ambiguous_rate": float((m["manual_label"] == "AMBIGUOUS").mean()),
                "judge_confidence_when_agree": float(m.loc[m["agree"], "judge_confidence"].mean()) if m["agree"].any() else None,
                "judge_confidence_when_disagree": float(m.loc[~m["agree"], "judge_confidence"].mean()) if (~m["agree"]).any() else None,
                "confusion_manual_rows_judge_cols": conf.to_dict(orient="index"),
                "disagreement_rows": m.loc[~m["agree"], ["audit_row", "xstest_id", "policy", "type", "manual_label",
                                                        "judge_label"]].to_dict(orient="records"),
            }
            # Policy comparison recomputed with the manual labels on the audited subset
            mm = m.rename(columns={"judge_label": "ai_label"}).assign(judge_label=m["manual_label"])
            mm["response_tokens"] = 0
            summary["manual_audit"]["manual_label_rates_on_subset"] = {
                p: {k: v for k, v in policy_metrics(g).items() if k.endswith("_rate")} for p, g in mm.groupby("policy")}
            print(f"Manual audit: n={len(m)} agreement={summary['manual_audit']['agreement']:.3f} "
                  f"kappa={summary['manual_audit']['cohen_kappa']:.3f}")
            print(conf.to_string())
        else:
            print("Audit sheet found but no manual labels filled yet:", audit_path)
    else:
        print("No audit sheet yet. Run: python -m task4_safety.make_audit_sheet")

    from common.logging_utils import save_json as _save
    _save(outdir / "safety_summary.json", summary)
    print("Saved", outdir / "safety_summary.json")


if __name__ == "__main__":
    main()
