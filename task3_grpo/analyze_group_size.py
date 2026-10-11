from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, perm=None):
    """Return K-sized groups while keeping total cached completions fixed.

    Partitioning rule (documented): every prompt keeps all of its 8 cached completions, and they are
    split into 8/K disjoint groups of K consecutive completions in generation_index order
    (K=8: {0..7}; K=4: {0..3},{4..7}; K=2: {0,1},{2,3},...). Every condition therefore uses exactly the
    same total generations (n_prompts * 8) and the same completions; only how they are grouped changes.
    This is the equal-generation analogue of "K completions for each of 8/K times as many prompts".
    An optional `perm` (list of index permutations per prompt) randomizes which completions share a group.
    """
    if 8 % k != 0:
        raise ValueError(f"K={k} must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        order = list(range(8))
        if perm is not None:
            order = perm[pid]
        for g in range(8 // k):
            members = [comps[j] for j in order[g * k:(g + 1) * k]]
            groups.append({"prompt_id": pid, "group_in_prompt": g,
                           "rewards": [float(m["reward"]) for m in members]})
    return groups


def group_metrics(groups, tol):
    """Informative-group rate, within-group reward std, and variance of the group-relative signal."""
    stds = np.array([np.std(g["rewards"]) for g in groups])
    informative = stds > tol
    advs, centered = [], []
    for g, s in zip(groups, stds):
        r = np.asarray(g["rewards"])
        advs.extend(((r - r.mean()) / (s + tol)).tolist())     # same formula as the fixed GRPO helper
        centered.extend((r - r.mean()).tolist())
    advs, centered = np.asarray(advs), np.asarray(centered)
    return {
        "n_groups": len(groups),
        "n_completions": int(sum(len(g["rewards"]) for g in groups)),
        "informative_group_rate": float(informative.mean()),
        "uninformative_group_fraction": float(1 - informative.mean()),
        "mean_within_group_reward_std": float(stds.mean()),
        "median_within_group_reward_std": float(np.median(stds)),
        # variance of the normalized advantage over all completions (0 for uninformative groups)
        "advantage_variance": float(advs.var()),
        # variance of the mean-centred (un-normalized) relative signal r - mean_group(r)
        "centered_reward_variance": float(centered.var()),
        "mean_abs_advantage": float(np.abs(advs).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))
    tol = float(cfg.get("uninformative_tol", 1e-6))
    ks = [int(k) for k in cfg["group_sizes"]]

    # Prompt difficulty: mean reward over all 8 cached completions of the prompt; bins are the
    # terciles of that value across cached prompts (hard = lowest mean reward).
    diff = {pid: float(np.mean([c["reward"] for c in comps[:8]])) for pid, comps in by_prompt.items()}
    q1, q2 = np.quantile(list(diff.values()), [1 / 3, 2 / 3])
    def bin_of(pid):
        d = diff[pid]
        return "hard" if d <= q1 else ("medium" if d <= q2 else "easy")
    bins = {pid: bin_of(pid) for pid in by_prompt}

    # Relative tolerances: RM rewards are continuous, so exact-zero std is rare. We also report the
    # fraction of groups whose std falls below a fraction of the global reward std ("near-uninformative").
    all_r = np.array([c["reward"] for comps in by_prompt.values() for c in comps[:8]])
    global_std = float(all_r.std())
    rel_tols = [0.1, 0.25]

    out = {"partition_rule": regroup_equal_generation_budget.__doc__.strip(),
           "difficulty_rule": "prompt mean reward over its 8 cached completions; tercile bins (hard<=q1<medium<=q2<easy)",
           "difficulty_thresholds": {"q1": float(q1), "q2": float(q2)},
           "n_prompts": len(by_prompt), "total_generations": int(len(by_prompt) * 8),
           "global_reward_std": global_std, "tol": tol, "by_k": {}}
    rng = np.random.default_rng(int(cfg["seed"]))
    n_boot = 200
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k)
        res = group_metrics(groups, tol)
        res["near_uninformative_fraction"] = {
            str(rt): float(np.mean([np.std(g["rewards"]) < rt * global_std for g in groups])) for rt in rel_tols}
        res["by_difficulty"] = {}
        for b in ["hard", "medium", "easy"]:
            gb = [g for g in groups if bins[g["prompt_id"]] == b]
            m = group_metrics(gb, tol)
            m["near_uninformative_fraction"] = {
                str(rt): float(np.mean([np.std(g["rewards"]) < rt * global_std for g in gb])) for rt in rel_tols}
            m["n_prompts"] = int(sum(1 for p in bins.values() if p == b))
            res["by_difficulty"][b] = m
        # Robustness: random assignment of completions to groups (bootstrap over partitions).
        boot = []
        for _ in range(n_boot):
            perm = {pid: rng.permutation(8).tolist() for pid in by_prompt}
            gm = group_metrics(regroup_equal_generation_budget(by_prompt, k, perm), tol)
            gm["near"] = float(np.mean([np.std(g["rewards"]) < rel_tols[0] * global_std
                                        for g in regroup_equal_generation_budget(by_prompt, k, perm)]))
            boot.append(gm)
        res["random_partition_bootstrap"] = {
            key: {"mean": float(np.mean([b_[key] for b_ in boot])), "std": float(np.std([b_[key] for b_ in boot]))}
            for key in ["informative_group_rate", "mean_within_group_reward_std", "advantage_variance",
                        "centered_reward_variance", "near"]}
        out["by_k"][str(k)] = res
        print(f"K={k}: groups={res['n_groups']} informative={res['informative_group_rate']:.3f} "
              f"mean_std={res['mean_within_group_reward_std']:.3f} adv_var={res['advantage_variance']:.3f} "
              f"near-uninformative(0.1)={res['near_uninformative_fraction']['0.1']:.3f}")
    path = repo_path(cfg["results_dir"]) / "group_size_study.json"
    save_json(path, out)
    print("Saved", path)


if __name__ == "__main__":
    main()
