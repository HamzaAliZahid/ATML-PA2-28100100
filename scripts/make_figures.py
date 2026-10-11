"""Build report tables (CSV) and figures (PNG) from saved results.

Run after the experiments: python -m scripts.make_figures
Every number comes from a file under results/; missing results are skipped with a message.
Outputs: report/tables/*.csv and report/figures/*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.data import read_jsonl, repo_path

R = repo_path("results")
TAB = repo_path("report/tables")
FIG = repo_path("report/figures")
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load(p):
    p = R / p
    if not p.exists():
        print("  (missing)", p.relative_to(repo_path(".")))
        return None
    return json.loads(p.read_text())


def save_table(df, name):
    df.to_csv(TAB / f"{name}.csv")
    print(f"  table  {name}.csv\n{df.round(4).to_string()}\n")


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figure {name}.png")


def dpo_row(d):
    p, g = d["preference"]["overall"], d.get("generation", {})
    wl = d.get("word_limit") or {}
    return {
        "pref_acc": p["preference_accuracy"], "dpo_loss": p["dpo_loss"], "margin": p["margin_mean"],
        "reward": g.get("reward", {}).get("mean"), "kl_token": g.get("kl_token_mean"), "kl_seq": g.get("kl_seq_mean"),
        "len_mean": g.get("response_length", {}).get("mean"), "len_std": g.get("response_length", {}).get("std"),
        "wordlimit_compliance": wl.get("compliance"),
    }


def task1():
    print("Task 1")
    rows = {}
    for name, f in [("sft", "eval_sft.json"), ("standard (1 epoch)", "eval_standard.json"),
                    ("beta=0.03 (short)", "eval_beta0p03.json"), ("beta=0.10 (short)", "eval_beta0p1.json"),
                    ("beta=0.30 (short)", "eval_beta0p3.json")]:
        d = load(f"task1_dpo/{f}")
        if d:
            rows[name] = dpo_row(d)
    if rows:
        save_table(pd.DataFrame(rows).T, "t1_dpo_summary")
    ls = load("task1_dpo/length_study_summary.json")
    if ls:
        recs = {}
        for cond, d in ls["conditions"].items():
            for s, m in d["preference"]["by_stratum"].items():
                recs[(cond, s)] = {"pref_acc": m["preference_accuracy"], "margin": m["margin_mean"], "n": m["n"]}
        df = pd.DataFrame(recs).T
        save_table(df, "t1_length_strata")
        piv = df["pref_acc"].unstack(0)
        fig, ax = plt.subplots(figsize=(5, 3))
        piv.plot.bar(ax=ax)
        ax.set_ylabel("held-out preference accuracy"); ax.set_xlabel("")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        save_fig(fig, "t1_length_strata")
        g = {c: {"len_mean": (d.get("generation") or {}).get("response_length", {}).get("mean"),
                 "wordlimit_compliance": (d.get("word_limit") or {}).get("compliance"),
                 "words_over_limit": (d.get("word_limit") or {}).get("mean_words_over_limit")}
             for c, d in ls["conditions"].items()}
        save_table(pd.DataFrame(g).T, "t1_length_generation")
        save_table(pd.DataFrame(ls["data_profile"]["length_balanced_train"]).T, "t1_data_profile_balanced")
        print("  standard-train length profile:", ls["data_profile"].get("standard_train"))
    for run in ["standard", "beta0p03", "beta0p1", "beta0p3", "length_balanced"]:
        p = R / f"task1_dpo/train_{run}_log.jsonl"
        if p.exists():
            log = pd.DataFrame(read_jsonl(p))
            fig, axs = plt.subplots(1, 3, figsize=(10, 2.6))
            for ax, k in zip(axs, ["loss", "preference_accuracy", "dpo_margin_mean"]):
                ax.plot(log["step"], log[k]); ax.set_title(k); ax.set_xlabel("optimizer step")
            save_fig(fig, f"t1_train_{run}")


def rl_traj(task, run, keys, name):
    p = R / f"{task}/train_{run}_log.jsonl"
    if not p.exists():
        print("  (missing)", p)
        return
    log = pd.DataFrame(read_jsonl(p))
    n = len(keys)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 5))
    for ax, k in zip(axs.flat, keys):
        ax.plot(log["update"], log[k], marker=".")
        ax.set_title(k, fontsize=9); ax.set_xlabel("update")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    save_fig(fig, name)
    log.drop(columns=[c for c in ["epochs", "prompt_ids"] if c in log], errors="ignore").to_csv(TAB / f"{name}.csv", index=False)


def heldout_table(task, names, out):
    rows = {}
    for label, f in names:
        d = load(f"{task}/eval_{f}.json")
        if d:
            rows[label] = {"reward": d["reward"]["mean"], "kl_token": d["kl_token_mean"], "kl_seq": d["kl_seq_mean"],
                           "entropy_full": d["full_entropy"], "entropy_sample": d["sample_entropy"],
                           "len_mean": d["response_length"]["mean"], "len_std": d["response_length"]["std"],
                           "truncated": d["truncated_fraction"]}
    if rows:
        save_table(pd.DataFrame(rows).T, out)


def task2():
    print("Task 2")
    rl_traj("task2_ppo", "standard",
            ["reward_raw_mean", "kl_ref_token", "policy_loss", "value_loss", "entropy", "policy_grad_norm",
             "clip_fraction", "response_length", "explained_variance"], "t2_ppo_standard_trajectory")
    s = load("task2_ppo/train_standard_summary.json")
    if s:
        print(f"  PPO standard: wall-clock {s['wall_clock_s']/60:.1f} min, peak VRAM {s['peak_vram_gb']:.2f} GB")
    heldout_table("task2_ppo", [("midpoint", "midpoint"), ("standard", "standard")], "t2_ppo_heldout")
    c = load("task2_ppo/clipping_cached_study.json")
    if c:
        rows = {e: {"step0_clip_frac": v["clip_fraction"], "step0_affected": v["affected_token_fraction"],
                    "final_clip_frac": c["trajectories"][e][-1]["clip_fraction"],
                    "final_affected": c["trajectories"][e][-1]["affected_token_fraction"],
                    "final_abs_log_ratio": c["trajectories"][e][-1]["abs_log_ratio_mean"]}
                for e, v in c["step0_midpoint_vs_cached_old"]["per_eps"].items()}
        save_table(pd.DataFrame(rows).T, "t2_clip_cached")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for e, tr in c["trajectories"].items():
            ax.plot([t["step"] for t in tr], [t["affected_token_fraction"] for t in tr], marker="o", label=f"eps={e}")
        ax.set_xlabel("probe step on cached batch"); ax.set_ylabel("affected-token fraction"); ax.legend()
        save_fig(fig, "t2_clip_cached")
    f = load("task2_ppo/clipping_forks_summary.json")
    if f:
        rows = {k: {"eps": v["clip_epsilon"], "reward": v["heldout"]["reward"]["mean"], "kl_token": v["heldout"]["kl_token_mean"],
                    "len_mean": v["heldout"]["response_length"]["mean"], **v["stability"]} for k, v in f.items()}
        save_table(pd.DataFrame(rows).T, "t2_clip_forks")
    k = load("task2_ppo/kl_study_summary.json")
    if k:
        rows = {n: {"beta_kl": v["kl_beta"], "reward": v["heldout"]["reward"]["mean"], "kl_token": v["heldout"]["kl_token_mean"],
                    "entropy": v["heldout"]["full_entropy"], "len_mean": v["heldout"]["response_length"]["mean"],
                    "len_std": v["heldout"]["response_length"]["std"]} for n, v in k.items()}
        save_table(pd.DataFrame(rows).T, "t2_kl_study")
        keys = ["reward_raw_mean", "kl_ref_token", "entropy", "response_length"]
        fig, axs = plt.subplots(1, 4, figsize=(12, 2.8))
        for ax, key in zip(axs, keys):
            for n, v in k.items():
                ax.plot(range(1, len(v["trajectory"][key]) + 1), v["trajectory"][key], marker=".", label=f"beta={v['kl_beta']}")
            ax.set_title(key, fontsize=9); ax.set_xlabel("update")
        axs[0].legend(fontsize=7)
        save_fig(fig, "t2_kl_trajectories")


def task3():
    print("Task 3")
    rl_traj("task3_grpo", "standard",
            ["reward_mean", "kl_ref_token", "group_reward_std_mean", "uninformative_group_fraction",
             "policy_loss", "grad_norm", "entropy", "response_length"], "t3_grpo_standard_trajectory")
    s = load("task3_grpo/train_standard_summary.json")
    if s:
        print(f"  GRPO standard: wall-clock {s['wall_clock_s']/60:.1f} min, peak VRAM {s['peak_vram_gb']:.2f} GB")
    heldout_table("task3_grpo", [("midpoint", "midpoint"), ("standard", "standard"),
                                 ("canonical (short)", "norm_grpo"), ("Dr. GRPO (short)", "norm_dr_grpo")], "t3_grpo_heldout")
    g = load("task3_grpo/group_size_study.json")
    if g:
        rows = {}
        for kk, v in g["by_k"].items():
            rows[(f"K={kk}", "all")] = {k2: v[k2] for k2 in ["n_groups", "informative_group_rate", "mean_within_group_reward_std",
                                                            "advantage_variance", "centered_reward_variance"]}
            rows[(f"K={kk}", "all")]["near_uninf(0.1*std)"] = v["near_uninformative_fraction"]["0.1"]
            for b, m in v["by_difficulty"].items():
                rows[(f"K={kk}", b)] = {k2: m[k2] for k2 in ["n_groups", "informative_group_rate", "mean_within_group_reward_std",
                                                            "advantage_variance", "centered_reward_variance"]}
                rows[(f"K={kk}", b)]["near_uninf(0.1*std)"] = m["near_uninformative_fraction"]["0.1"]
        save_table(pd.DataFrame(rows).T, "t3_group_size")
    n = load("task3_grpo/normalization_study.json")
    if n:
        rows = {}
        for lt, v in n["conditions"].items():
            for b, m in v["length_conditioned"].items():
                if m.get("n"):
                    rows[(lt, b)] = {"n": m["n"], "grad_norm": m["grad_norm_mean"], "per_token_weight": m["per_token_weight_mean"],
                                     "abs_adv": m["abs_advantage_mean"], "grad_norm_per_abs_adv": m["grad_norm_per_abs_adv"]}
        save_table(pd.DataFrame(rows).T, "t3_norm_length_conditioned")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for lt, v in n["conditions"].items():
            ax.plot(range(1, len(v["train_response_length_trajectory"]) + 1), v["train_response_length_trajectory"], marker=".", label=lt)
        ax.set_xlabel("update"); ax.set_ylabel("mean train response length"); ax.legend()
        save_fig(fig, "t3_norm_length")


def task4():
    print("Task 4")
    p = R / "task4_safety/safety_summary.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        cols = ["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "unsafe_justified_refusal_rate",
                "ambiguous_rate", "response_tokens_mean"]
        save_table(df[cols], "t4_safety_calibration")
        fig, ax = plt.subplots(figsize=(6, 3))
        df[cols[:4]].plot.bar(ax=ax)
        ax.set_ylabel("rate"); ax.legend(fontsize=7)
        save_fig(fig, "t4_safety_calibration")
    s = load("task4_safety/safety_summary.json")
    if s and "manual_audit" in s:
        a = s["manual_audit"]
        print(f"  audit: n={a['n_labelled']} agreement={a['agreement']:.3f} kappa={a['cohen_kappa']:.3f}")
        save_table(pd.DataFrame(a["confusion_manual_rows_judge_cols"]).T, "t4_audit_confusion")


def task5():
    print("Task 5")
    p = R / "task5_feedback/feedback_comparison.csv"
    if p.exists():
        save_table(pd.read_csv(p, index_col=0), "t5_in_vs_out_domain")
    p = R / "task5_feedback/diagnostic_table.csv"
    if p.exists():
        save_table(pd.read_csv(p).set_index(["pair", "mechanism"]), "t5_diagnostics")
    d = load("task5_feedback/diagnostic_study.json")
    if d:
        print("  S_reason:", d["S_reason"], " S_outcome:", d["S_outcome"])
        save_table(pd.DataFrame(d["per_variant_mean_signal"]).T, "t5_per_variant_signal")


if __name__ == "__main__":
    for f in (task1, task2, task3, task4, task5):
        try:
            f()
        except Exception as exc:  # keep going so one missing piece doesn't block the rest
            print(f"  ERROR in {f.__name__}: {exc!r}")
    print("Done ->", TAB, FIG)
