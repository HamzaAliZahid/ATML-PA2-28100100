from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


# Controlled pairs: (diagnostically better, worse, what the pair tests)
PAIRS = [
    ("clean_correct", "corrupt_reasoning_correct_final", "reasoning_only"),   # S_reason
    ("clean_correct", "good_reasoning_wrong_final", "outcome_only"),          # S_outcome
    ("clean_correct", "persuasive_filler_correct", "persuasive_filler"),      # verbosity / style
    ("clean_correct", "gold_distractor_wrong_final", "gold_distractor"),      # parser robustness / outcome
    ("good_reasoning_wrong_final", "gold_distractor_wrong_final", "distractor_vs_wrong_final"),
]


def verifier_pref(a_row, b_row):
    ra = exact_reward(a_row["response"], str(a_row["gold_final"]))
    rb = exact_reward(b_row["response"], str(b_row["gold_final"]))
    return ("A" if ra > rb else "B" if rb > ra else "TIE"), ra, rb


def judge_pref(judge, problem, a, b):
    """Query the judge in both presentation orders; combine into a single preference.

    Each order returns A/B/TIE for (first, second). Order 2 is mapped back to (a, b).
    Combined: both prefer a -> A; both prefer b -> B; otherwise TIE (inconsistent or tied).
    """
    p1 = judge.compare(problem, a, b)
    p2 = {"A": "B", "B": "A", "TIE": "TIE"}[judge.compare(problem, b, a)]
    if p1 == p2:
        combined = p1
    elif "TIE" in (p1, p2):
        combined = p1 if p2 == "TIE" else p2  # one tie + one decision -> lean to the decision
    else:
        combined = "TIE"  # order-dependent flip
    score = {"A": 1.0, "B": 0.0, "TIE": 0.5}
    return combined, p1, p2, (score[p1] + score[p2]) / 2


def rates(prefs):
    n = len(prefs)
    return {
        "n": n,
        "better_rate": float(np.mean([p == "A" for p in prefs])) if n else float("nan"),
        "tie_rate": float(np.mean([p == "TIE" for p in prefs])) if n else float("nan"),
        "wrong_preference_rate": float(np.mean([p == "B" for p in prefs])) if n else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")

    pair_recs = []
    variant_scores = defaultdict(lambda: {"verifier": [], "rlaif_group_reward": []})
    for pid, v in sorted(groups.items(), key=lambda x: x[0]):
        problem = v["clean_correct"]["question"]
        for better, worse, kind in PAIRS:
            vp, ra, rb = verifier_pref(v[better], v[worse])
            jp, j1, j2, jscore = judge_pref(judge, problem, v[better]["response"], v[worse]["response"])
            pair_recs.append({"problem_id": pid, "pair": kind, "better": better, "worse": worse,
                              "verifier_pref": vp, "verifier_better": ra, "verifier_worse": rb,
                              "judge_pref": jp, "judge_order1": j1, "judge_order2_mapped": j2,
                              "judge_order_consistent": j1 == j2, "judge_mean_score": jscore,
                              "better_len": len(v[better]["response"]), "worse_len": len(v[worse]["response"])})
        # Per-variant absolute signals: verifier reward and the direct-RLAIF group reward
        # (normalized pairwise win rate among the 5 variants, exactly the training-time reward form).
        order = sorted(EXPECTED_VARIANTS)
        gr = judge.group_rewards(problem, [v[k]["response"] for k in order])
        for k, g in zip(order, gr):
            variant_scores[k]["verifier"].append(exact_reward(v[k]["response"], str(v[k]["gold_final"])))
            variant_scores[k]["rlaif_group_reward"].append(g)
            # sanity check against the staff-provided expectation
            exp = v[k].get("expected_exact_reward")
            if exp is not None and float(exp) != variant_scores[k]["verifier"][-1]:
                print(f"WARNING: verifier disagrees with expected_exact_reward for {pid}/{k}")
        print(f"  problem {pid} done", flush=True)

    write_jsonl(outdir / "diagnostic_pairs.jsonl", pair_recs)
    by_pair = {}
    for kind in [p[2] for p in PAIRS]:
        rs = [r for r in pair_recs if r["pair"] == kind]
        by_pair[kind] = {
            "verifier": rates([r["verifier_pref"] for r in rs]),
            "ai_judge": rates([r["judge_pref"] for r in rs]),
            "ai_judge_order1_only": rates([r["judge_order1"] for r in rs]),
            "ai_judge_order_consistency": float(np.mean([r["judge_order_consistent"] for r in rs])),
            "ai_judge_mean_score_for_better": float(np.mean([r["judge_mean_score"] for r in rs])),
            "verifier_judge_agreement": float(np.mean([r["verifier_pref"] == r["judge_pref"] for r in rs])),
        }
    summary = {
        "n_problems": len(groups),
        "pairs": {k: {"better": b, "worse": w} for b, w, k in PAIRS},
        "judge_combination_rule": judge_pref.__doc__.strip(),
        "by_pair": by_pair,
        "S_reason": {m: by_pair["reasoning_only"][m]["better_rate"] for m in ["verifier", "ai_judge"]},
        "S_outcome": {m: by_pair["outcome_only"][m]["better_rate"] for m in ["verifier", "ai_judge"]},
        "S_outcome_incl_distractor": {
            m: float(np.mean([r[f"{'verifier' if m == 'verifier' else 'judge'}_pref"] == "A" for r in pair_recs
                              if r["pair"] in ("outcome_only", "gold_distractor")])) for m in ["verifier", "ai_judge"]},
        "filler_susceptibility_ai_judge": by_pair["persuasive_filler"]["ai_judge"]["wrong_preference_rate"],
        "per_variant_mean_signal": {k: {m: float(np.mean(x)) for m, x in d.items()} for k, d in variant_scores.items()},
    }
    save_json(outdir / "diagnostic_study.json", summary)
    for kind, d in by_pair.items():
        print(f"{kind:28s} verifier better/tie/wrong = {d['verifier']['better_rate']:.2f}/{d['verifier']['tie_rate']:.2f}/"
              f"{d['verifier']['wrong_preference_rate']:.2f}   judge = {d['ai_judge']['better_rate']:.2f}/"
              f"{d['ai_judge']['tie_rate']:.2f}/{d['ai_judge']['wrong_preference_rate']:.2f}")
    print("Saved", outdir / "diagnostic_study.json")


if __name__ == "__main__":
    main()
