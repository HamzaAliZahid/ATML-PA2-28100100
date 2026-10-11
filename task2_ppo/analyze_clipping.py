from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import load_json, save_json, set_seed, append_jsonl
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer
from common.rl_utils import cast_trainable_to_fp32, disable_dropout
from task2_ppo.ppo import (
    clip_affected_fraction,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def prompt_lookup(cfg):
    table = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            table.setdefault(str(r.get("prompt_id")), r)
    return table


def build_cached_batch(cfg, rows, tokenizer):
    """Rebuild token sequences for the cached rollouts and right-pad all per-token tensors.

    The response is re-tokenized from text; the EOS token is appended when the cached rollout
    terminated with EOS. Rows whose re-tokenized length does not match the cached log-prob
    length are truncated to the shorter length and reported.
    """
    table = prompt_lookup(cfg)
    items, mismatches = [], 0
    for r in rows:
        prow = table[str(r["prompt_id"])]
        p_ids = tokenizer.apply_chat_template(prompt_messages(prow), tokenize=True, add_generation_prompt=True)
        r_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("terminated_with_eos", False):
            r_ids = r_ids + [tokenizer.eos_token_id]
        n_cache = int(r["old_logprobs"].numel())
        if len(r_ids) != n_cache:
            mismatches += 1
        n = min(len(r_ids), n_cache)
        items.append({"prompt_ids": p_ids, "response_ids": r_ids[:n], "n": n, "row": r})

    T = max(it["n"] for it in items)
    B = len(items)
    def pad(key):
        out = torch.zeros(B, T)
        for i, it in enumerate(items):
            out[i, : it["n"]] = it["row"][key][: it["n"]].float()
        return out
    mask = torch.zeros(B, T)
    for i, it in enumerate(items):
        mask[i, : it["n"]] = 1.0
    reward = torch.tensor([float(it["row"].get("effective_terminal_reward", it["row"].get("raw_terminal_reward", 0.0)))
                           for it in items])
    return {
        "items": items, "mask": mask, "old": pad("old_logprobs"), "ref": pad("ref_logprobs"),
        "values": pad("values") if "values" in rows[0] else None, "reward": reward,
        "tokenization_mismatches": mismatches,
    }


def policy_logprobs(policy, tokenizer, items, T, device, idx, grad=False):
    """Response-token log-probs for items[idx], right-padded to T."""
    out = torch.zeros(len(idx), T, device=device)
    for j, i in enumerate(idx):
        it = items[i]
        ids = torch.tensor([it["prompt_ids"] + it["response_ids"]], device=device)
        attn = torch.ones_like(ids)
        rid = torch.tensor([it["response_ids"]], device=device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lp, _ = response_token_logprobs(policy, ids, attn, len(it["prompt_ids"]), rid)
        out[j, : it["n"]] = lp[0]
    return out


def ratio_diagnostics(new, old, adv, mask, eps_list):
    ratio = torch.exp(new - old)
    surr_unclipped = masked_mean(ratio * adv, mask)
    out = {"ratio_mean": float(masked_mean(ratio, mask)), "ratio_std": float(ratio[mask.bool()].std()),
           "abs_log_ratio_mean": float(masked_mean((new - old).abs(), mask)),
           "unclipped_surrogate": float(surr_unclipped), "per_eps": {}}
    for e in eps_list:
        clipped = torch.minimum(ratio * adv, ratio.clamp(1 - e, 1 + e) * adv)
        out["per_eps"][str(e)] = {
            "clip_fraction": float(masked_mean(((ratio < 1 - e) | (ratio > 1 + e)).float(), mask)),
            "affected_token_fraction": float(clip_affected_fraction(ratio, adv, mask, e)),
            "clipped_surrogate": float(masked_mean(clipped, mask)),
            "surrogate_gap": float(surr_unclipped - masked_mean(clipped, mask)),
        }
    return out


def cached_batch_study(cfg, args):
    """Immediate geometric effect of epsilon on one fixed cached batch.

    (a) ratio of the supplied midpoint policy vs the cached old log-probs (step 0), evaluated for every eps;
    (b) for each eps separately, `steps` PPO optimizer steps on the same batch from the same midpoint,
        recording clip / affected-token fractions and surrogates before every step.
    """
    set_seed(int(cfg["seed"]))
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    tok = load_tokenizer(cfg["base_model"])
    batch = build_cached_batch(cfg, rows, tok)
    eps_list = [float(e) for e in cfg["clip_values"]]
    mask = batch["mask"]
    print(f"Cached rollouts: {len(rows)}; tokenization mismatches: {batch['tokenization_mismatches']}")

    # Advantages with the same shaping/GAE/whitening as the continuation loop, from cached quantities.
    with torch.no_grad():
        rewards = shaped_rewards(batch["reward"], batch["old"], batch["ref"], mask, float(cfg["kl_beta"]))
        if batch["values"] is None:
            raise ValueError("cache has no values; cannot compute GAE")
        adv, returns = compute_gae(rewards, batch["values"] * mask, mask,
                                   gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
        adv_n = normalize_advantages(adv, mask)

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    cast_trainable_to_fp32(policy)
    device = next(policy.parameters()).device
    params = [p for p in policy.parameters() if p.requires_grad]
    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    T = mask.shape[1]
    B = mask.shape[0]
    mask_d, old_d, adv_d = mask.to(device), batch["old"].to(device), adv_n.to(device)

    new0 = policy_logprobs(policy, tok, batch["items"], T, device, list(range(B)))
    step0 = ratio_diagnostics(new0, old_d, adv_d, mask_d, eps_list)
    print("step-0 ratio diagnostics:", {k: v for k, v in step0.items() if k != "per_eps"})

    micro = int(args.micro_batch)
    # Learning rate for the probe steps. Default = training lr; a larger value (e.g. --lr 3e-5) can be
    # used to make the ratio spread visible within a few steps. Report whichever you use.
    lr = float(args.lr) if args.lr else float(cfg["policy_learning_rate"])
    trajectories = {}
    for e in eps_list:
        with torch.no_grad():  # restore the identical midpoint for every eps
            for n, p in policy.named_parameters():
                if n in init_state:
                    p.copy_(init_state[n])
        opt = torch.optim.AdamW(params, lr=lr)
        traj = []
        for s in range(int(args.steps) + 1):
            with torch.no_grad():
                cur = policy_logprobs(policy, tok, batch["items"], T, device, list(range(B)))
            d = ratio_diagnostics(cur, old_d, adv_d, mask_d, [e])
            traj.append({"step": s, **{k: v for k, v in d.items() if k != "per_eps"}, **d["per_eps"][str(e)]})
            if s == int(args.steps):
                break
            opt.zero_grad(set_to_none=True)
            n_tok = float(mask_d.sum())
            for start in range(0, B, micro):
                idx = list(range(start, min(start + micro, B)))
                lp = policy_logprobs(policy, tok, batch["items"], T, device, idx, grad=True)
                loss, _, _ = ppo_policy_loss(lp, old_d[idx], adv_d[idx], mask_d[idx], eps=e)
                # re-weight micro-batch token means into one full-batch token mean
                (loss * mask_d[idx].sum() / n_tok).backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
            opt.step()
        trajectories[str(e)] = traj
        print(f"eps={e}: final clip={traj[-1]['clip_fraction']:.3f} affected={traj[-1]['affected_token_fraction']:.3f}")
    clear_gpu(policy)
    return {
        "n_rollouts": len(rows), "n_valid_tokens": int(mask.sum()),
        "tokenization_mismatches": batch["tokenization_mismatches"],
        "advantage": "KL-shaped (beta_kl=config) terminal effective reward, GAE with cached values, whitened",
        "steps_per_eps": int(args.steps), "learning_rate": lr,
        "step0_midpoint_vs_cached_old": step0,
        "trajectories": trajectories,
    }


def stability_stats(log_path):
    logs = read_jsonl(log_path)
    g = np.array([r["policy_grad_norm"] for r in logs])
    kl_old = np.array([r["approx_kl_old"] for r in logs])
    rmax = np.array([r["ratio_max"] for r in logs])
    reward = np.array([r["reward_raw_mean"] for r in logs])
    kl_ref = np.array([r["kl_ref_token"] for r in logs])
    return {
        "policy_grad_norm_mean": float(g.mean()), "policy_grad_norm_std": float(g.std()),
        "policy_grad_norm_max": float(g.max()),
        "approx_kl_old_mean": float(kl_old.mean()), "approx_kl_old_max": float(kl_old.max()),
        "ratio_max_overall": float(rmax.max()),
        "clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in logs])),
        "reward_std_over_updates": float(reward.std()),
        "kl_ref_increase": float(kl_ref[-1] - kl_ref[0]),
        "value_loss_mean": float(np.mean([r["value_loss"] for r in logs])),
    }


def run_forks(cfg, config_path, eps_list, skip_existing):
    from task2_ppo.continue_train import run_ppo
    from common.rl_eval import evaluate_rl_policy
    out = {}
    for e in eps_list:
        tag = f"clip_eps{e:g}".replace(".", "p")
        adapter = f"outputs/task2_ppo/{tag}"
        if not (skip_existing and (repo_path(adapter) / "adapter_config.json").exists()):
            run_ppo(config_path, output=adapter, updates=int(cfg["fork_updates"]), clip_epsilon=e, run_name=tag)
        ev = evaluate_rl_policy(cfg, adapter, tag, skip_existing=skip_existing)
        out[tag] = {"clip_epsilon": e, "heldout": ev,
                    "train_summary": load_json(repo_path(cfg["results_dir"]) / f"train_{tag}_summary.json"),
                    "stability": stability_stats(repo_path(cfg["results_dir"]) / f"train_{tag}_log.jsonl")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--part", choices=["cached", "forks", "all"], default="all")
    ap.add_argument("--steps", type=int, default=4, help="optimizer steps per eps on the cached batch")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, help="probe-step learning rate (default: config policy_learning_rate)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    eps_list = [float(e) for e in cfg["clip_values"]]
    print("Required epsilon values:", eps_list)
    if args.part in ("cached", "all"):
        res = cached_batch_study(cfg, args)
        save_json(results_dir / "clipping_cached_study.json", res)
        print("Saved", results_dir / "clipping_cached_study.json")
    if args.part in ("forks", "all"):
        res = run_forks(cfg, args.config, eps_list, args.skip_existing)
        save_json(results_dir / "clipping_forks_summary.json", res)
        print("Saved", results_dir / "clipping_forks_summary.json")


if __name__ == "__main__":
    main()
