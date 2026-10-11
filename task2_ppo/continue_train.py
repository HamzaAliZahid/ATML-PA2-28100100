from __future__ import annotations

import argparse
from pathlib import Path

import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import clear_gpu, reference_mode
from common.rl_utils import (
    cast_trainable_to_fp32,
    disable_dropout,
    grad_norm,
    masked_full_entropy,
    peak_vram_gb,
    reset_peak_vram,
)
from task2_ppo.ppo import (
    clip_affected_fraction,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)


def value_head(value_model):
    if hasattr(value_model, "score"):
        return value_model.score
    if hasattr(value_model, "classifier"):
        return value_model.classifier
    raise RuntimeError("Could not locate scalar value head")


def response_values(value_model, sequences, attention_mask, prompt_width, n_steps):
    """V(s_t) for each response position t (state = prompt + tokens before a_t)."""
    # Run only the transformer body (Qwen2Model), not the full classifier: the full model would
    # also apply the (fp32) score head to fp16 hidden states internally.
    inner = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    body = getattr(inner, inner.base_model_prefix)
    out = body(input_ids=sequences, attention_mask=attention_mask, return_dict=True, use_cache=False)
    hidden = out.last_hidden_state[:, prompt_width - 1: prompt_width - 1 + n_steps]
    return value_head(value_model)(hidden.float()).squeeze(-1).float()


def prompt_schedule(n_prompts: int, updates: int, per_update: int, seed: int):
    """Fixed prompt order shared by the standard run and every fork."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_prompts).tolist()
    need = updates * per_update
    while len(perm) < need:
        perm += rng.permutation(n_prompts).tolist()
    return [perm[u * per_update:(u + 1) * per_update] for u in range(updates)]


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # Deterministic on-policy ratios (dropout off) and fp32 trainable params for AdamW.
    disable_dropout(policy)
    disable_dropout(value_model)
    cast_trainable_to_fp32(policy)
    cast_trainable_to_fp32(value_model)
    for p in value_head(value_model).parameters():
        p.data = p.data.float()

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value_model = bundle["policy"], bundle["value_model"]
    tok = bundle["tokenizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    rows = bundle["prompt_rows"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    pparams = [p for p in policy.parameters() if p.requires_grad]
    vparams = [p for p in value_model.parameters() if p.requires_grad]

    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    vcoef = float(cfg["value_coef"])
    max_norm = float(cfg["max_grad_norm"])
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    per_update = int(cfg["prompts_per_update"])
    n_updates = int(cfg["updates"])
    schedule = prompt_schedule(len(rows), n_updates, per_update, int(cfg["seed"]))

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}_log.jsonl"
    roll_path = results_dir / f"train_{run_name}_rollouts.jsonl"
    for pth in (log_path, roll_path):
        if pth.exists():
            pth.unlink()

    print(f"[PPO:{run_name}] updates={n_updates} eps={eps} beta_kl={beta_kl} prompts/update={per_update}", flush=True)
    reset_peak_vram()
    t0 = time.perf_counter()
    total_gen_tokens = 0

    for u in range(n_updates):
        tu = time.perf_counter()
        batch_rows = [rows[i] for i in schedule[u]]
        prompts = [prompt_messages(r) for r in batch_rows]

        # ---- 1. on-policy rollout from the current policy (= pi_old for this update)
        gen = batch_generate(policy, tok, prompts,
                             max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(cfg["max_response_length"]),
                             temperature=float(cfg["generation"]["temperature"]),
                             top_p=float(cfg["generation"]["top_p"]),
                             do_sample=True)
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rid, mask = gen["response_ids"], gen["response_mask"]
        T = rid.shape[1]
        total_gen_tokens += int(mask.sum().item())

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
            old_values = response_values(value_model, seq, attn, pw, T) * mask
            raw_reward = score_reward_pairs(rm, rm_tok, prompts, gen["responses"],
                                            max_length=int(cfg["reward_max_length"])).float()
            eos = torch.tensor(gen["terminated_with_eos"], device=raw_reward.device)
            task_reward = raw_reward - penalty * (~eos).float()

            # ---- 2. KL-shaped token rewards, GAE and returns (computed once, held fixed across epochs)
            rewards = shaped_rewards(task_reward.to(old_logp.device), old_logp, ref_logp, mask, beta_kl)
            adv, returns = compute_gae(rewards, old_values, mask, gamma=gamma, lam=lam)
            adv_n = normalize_advantages(adv, mask)
            ev_den = torch.var(returns[mask.bool()])
            explained_var = float(1 - torch.var((returns - old_values)[mask.bool()]) / ev_den.clamp_min(1e-8))

        # ---- 3. PPO epochs over the rollout batch
        ep_logs = []
        for ep in range(int(cfg["ppo_epochs"])):
            new_logp, logits = response_token_logprobs(policy, seq, attn, pw, rid)
            ploss, ratio, clipfrac = ppo_policy_loss(new_logp, old_logp, adv_n, mask, eps=eps)
            entropy = masked_full_entropy(logits.detach(), mask)
            del logits
            popt.zero_grad(set_to_none=True)
            ploss.backward()
            pgn = grad_norm(pparams)
            torch.nn.utils.clip_grad_norm_(pparams, max_norm)
            popt.step()

            new_values = response_values(value_model, seq, attn, pw, T)
            vloss = value_mse_loss(new_values, returns, mask)
            vopt.zero_grad(set_to_none=True)
            (vcoef * vloss).backward()
            vgn = grad_norm(vparams)
            torch.nn.utils.clip_grad_norm_(vparams, max_norm)
            vopt.step()

            with torch.no_grad():
                log_r = new_logp.detach() - old_logp
                approx_kl = masked_mean(torch.exp(log_r) - 1 - log_r, mask)
            ep_logs.append({
                "policy_loss": float(ploss.item()), "value_loss": float(vloss.item()),
                "clip_fraction": float(clipfrac.item()),
                "affected_fraction": float(clip_affected_fraction(ratio, adv_n, mask, eps).item()),
                "ratio_mean": float(masked_mean(ratio, mask).item()),
                "ratio_max": float(ratio[mask.bool()].max().item()),
                "ratio_min": float(ratio[mask.bool()].min().item()),
                "approx_kl_old": float(approx_kl.item()),
                "policy_grad_norm": pgn, "value_grad_norm": vgn, "entropy": entropy,
            })

        lengths = mask.sum(-1).tolist()
        rec = {
            "update": u + 1,
            "prompt_ids": [r.get("source_index") for r in batch_rows],
            "reward_raw_mean": float(raw_reward.mean().item()),
            "reward_task_mean": float(task_reward.mean().item()),
            "kl_ref_token": float(sampled_kl(old_logp, ref_logp, mask).item()),
            "kl_ref_seq": float(((old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "sample_entropy": float(sample_entropy(old_logp, mask).item()),
            "response_length": float(np.mean(lengths)),
            "eos_fraction": float(eos.float().mean().item()),
            "value_mean": float(masked_mean(old_values, mask).item()),
            "return_mean": float(masked_mean(returns, mask).item()),
            "value_at_start": float(old_values[:, 0].mean().item()),
            "explained_variance": explained_var,
            "epochs": ep_logs,
            "update_time_s": time.perf_counter() - tu,
            "elapsed_s": time.perf_counter() - t0,
            "generated_tokens_total": total_gen_tokens,
        }
        # Epoch-averaged copies of the per-epoch diagnostics for easy plotting.
        for k in ep_logs[0]:
            rec[k] = float(np.mean([e[k] for e in ep_logs]))
        append_jsonl(log_path, rec)
        for r, resp, rw, L in zip(batch_rows, gen["responses"], raw_reward.tolist(), lengths):
            append_jsonl(roll_path, {"update": u + 1, "source_index": r.get("source_index"),
                                     "response": resp, "reward": rw, "length": L})
        print(f"[PPO:{run_name}] {u+1}/{n_updates} R={rec['reward_raw_mean']:.3f} KL={rec['kl_ref_token']:.4f} "
              f"pl={rec['policy_loss']:.4f} vl={rec['value_loss']:.4f} clip={rec['clip_fraction']:.3f} "
              f"H={rec['entropy']:.3f} len={rec['response_length']:.0f} t={rec['update_time_s']:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    value_model.save_pretrained(str(out) + "_value")
    summary = {
        "run_name": run_name, "output": str(out), "updates": n_updates,
        "clip_epsilon": eps, "kl_beta": beta_kl, "ppo_epochs": int(cfg["ppo_epochs"]),
        "prompts_per_update": per_update, "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "max_response_length": int(cfg["max_response_length"]), "seed": int(cfg["seed"]),
        "generated_tokens_total": total_gen_tokens,
        "wall_clock_s": elapsed, "peak_vram_gb": peak_vram_gb(),
        "train_log": str(log_path),
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    print(f"[PPO:{run_name}] done in {elapsed/60:.1f} min, peak VRAM {summary['peak_vram_gb']:.2f} GB", flush=True)
    clear_gpu(policy, value_model, rm)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
