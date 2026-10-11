from __future__ import annotations

import argparse

import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from common.rl_utils import (
    cast_trainable_to_fp32,
    disable_dropout,
    masked_full_entropy,
    peak_vram_gb,
    reset_peak_vram,
)
from task2_ppo.continue_train import prompt_schedule
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_stats,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    disable_dropout(policy)
    cast_trainable_to_fp32(policy)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tok, opt = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    rows = bundle["prompt_rows"]
    params = trainable_parameters(policy)
    K = int(cfg["num_generations"])
    per_update = int(cfg["prompts_per_update"])
    n_updates = int(cfg["updates"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    Lmax = int(cfg["max_completion_length"])
    tol = float(cfg.get("uninformative_tol", 1e-6))
    schedule = prompt_schedule(len(rows), n_updates, per_update, int(cfg["seed"]))

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}_log.jsonl"
    comp_path = results_dir / f"train_{run_name}_completions.jsonl"
    for pth in (log_path, comp_path):
        if pth.exists():
            pth.unlink()

    print(f"[GRPO:{run_name}] updates={n_updates} K={K} loss_type={loss_type} eps={eps} beta={beta}", flush=True)
    reset_peak_vram()
    t0 = time.perf_counter()
    total_gen_tokens = 0

    for u in range(n_updates):
        tu = time.perf_counter()
        batch_rows = [rows[i] for i in schedule[u]]
        prompts, group_ids = [], []
        for g, r in enumerate(batch_rows):
            for _ in range(K):
                prompts.append(prompt_messages(r))
                group_ids.append(g)
        gen = batch_generate(policy, tok, prompts,
                             max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=Lmax,
                             temperature=float(cfg["generation"]["temperature"]),
                             top_p=float(cfg["generation"]["top_p"]),
                             do_sample=True)
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rid, rmask = gen["response_ids"], gen["response_mask"]
        total_gen_tokens += int(rmask.sum().item())
        device = seq.device
        gids = torch.tensor(group_ids, device=device)

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
            rewards = score_reward_pairs(rm, rm_tok, prompts, gen["responses"],
                                         max_length=int(cfg.get("reward_max_length", 1280))).float().to(device)
            adv = group_relative_advantages(rewards, gids)
            stds, informative = group_reward_stats(rewards, gids, tol=tol)
        token_mask = rmask
        if bool(cfg.get("mask_truncated_completions", True)):
            token_mask = mask_truncated_sequences(rmask, gen["truncated"])

        # One gradient step per update (policy_epochs=1 => old == current policy, ratio = 1 at the step).
        # Sequences are processed one at a time; each contributes its exact share of the full-batch loss
        # (policy term: 1/N_seq; KL term: its tokens / all valid tokens), so the accumulated gradient equals
        # the batch gradient, and we obtain a per-completion gradient norm for the normalization study.
        N = len(prompts)
        n_valid_all = float(token_mask.sum().item())
        per_seq = []
        for epoch in range(int(cfg.get("policy_epochs", 1))):
            opt.zero_grad(set_to_none=True)
            total_loss, total_pol, total_kl, total_clip = 0.0, 0.0, 0.0, 0.0
            ent_vals = []
            for i in range(N):
                n_i = float(token_mask[i].sum().item())
                rec_i = {"idx": i, "group": group_ids[i], "length": int(rmask[i].sum().item()),
                         "truncated": bool(gen["truncated"][i]), "reward": float(rewards[i]),
                         "advantage": float(adv[i]), "grad_norm": 0.0, "masked": n_i == 0}
                if n_i == 0:
                    if epoch == 0:
                        per_seq.append(rec_i)
                    continue
                before = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                new_lp, logits = response_token_logprobs(policy, seq[i:i+1], attn[i:i+1], pw, rid[i:i+1])
                ent_vals.append((masked_full_entropy(logits.detach(), token_mask[i:i+1]), n_i))
                del logits
                loss_i, d = grpo_policy_loss(new_lp, old_logp[i:i+1], adv[i:i+1], token_mask[i:i+1],
                                             ref_logp[i:i+1], eps, beta, loss_type=loss_type,
                                             max_completion_length=Lmax)
                # For one sequence, loss_i = policy_term_i + beta * kl_i. In the full batch the policy
                # term is averaged over N sequences and the KL over all valid tokens, so reweight:
                kl_i = _kl_tensor(new_lp, ref_logp[i:i+1], token_mask[i:i+1])
                pol_part = loss_i - beta * kl_i  # == policy_term_i, differentiable
                weighted = pol_part / N + beta * kl_i * (n_i / n_valid_all)
                weighted.backward()
                gsq = 0.0
                for p, b in zip(params, before):
                    if p.grad is None:
                        continue
                    diff = p.grad.detach() - (b if b is not None else 0.0)
                    gsq += float(diff.float().pow(2).sum().item())
                rec_i["grad_norm"] = gsq ** 0.5
                rec_i["per_token_weight"] = abs(float(adv[i])) / (n_i if loss_type == "grpo" else Lmax)
                total_loss += float(weighted.item())
                total_pol += float(pol_part.item()) / N
                total_kl += float(d["sampled_kl"]) * n_i / n_valid_all
                total_clip += float(d["clip_fraction"]) * n_i / n_valid_all
                if epoch == 0:
                    per_seq.append(rec_i)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
            opt.step()

        lengths = [int(x) for x in rmask.sum(-1).tolist()]
        ent = sum(e * n for e, n in ent_vals) / max(sum(n for _, n in ent_vals), 1.0)
        rec = {
            "update": u + 1,
            "prompt_ids": [r.get("source_index") for r in batch_rows],
            "reward_mean": float(rewards.mean()),
            "reward_std_all": float(rewards.std(unbiased=False)),
            "group_reward_std_mean": float(np.mean(stds)),
            "uninformative_group_fraction": float(1 - np.mean(informative)),
            "kl_ref_token": float(sampled_kl(old_logp, ref_logp, rmask)),
            "kl_ref_seq": float(((old_logp - ref_logp) * rmask).sum(-1).mean()),
            "k3_kl_loss_term": total_kl,
            "policy_loss": total_pol,
            "loss": total_loss,
            "grad_norm": gn,
            "clip_fraction": total_clip,
            "entropy": ent,
            "response_length": float(np.mean(lengths)),
            "response_length_std": float(np.std(lengths)),
            "truncated_fraction": float(np.mean(gen["truncated"])),
            "masked_completions": int(sum(r["masked"] for r in per_seq)),
            "length_grad_corr": _corr([r["length"] for r in per_seq if not r["masked"]],
                                      [r["grad_norm"] for r in per_seq if not r["masked"]]),
            "update_time_s": time.perf_counter() - tu,
            "elapsed_s": time.perf_counter() - t0,
            "generated_tokens_total": total_gen_tokens,
        }
        append_jsonl(log_path, rec)
        for r, resp in zip(per_seq, gen["responses"]):
            append_jsonl(comp_path, {"update": u + 1, "loss_type": loss_type,
                                     "source_index": batch_rows[r["group"]].get("source_index"),
                                     **r, "response": resp})
        print(f"[GRPO:{run_name}] {u+1}/{n_updates} R={rec['reward_mean']:.3f} gstd={rec['group_reward_std_mean']:.3f} "
              f"uninf={rec['uninformative_group_fraction']:.2f} KL={rec['kl_ref_token']:.4f} gn={gn:.3f} "
              f"H={ent:.3f} len={rec['response_length']:.0f} t={rec['update_time_s']:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    summary = {
        "run_name": run_name, "output": str(out), "updates": n_updates, "loss_type": loss_type,
        "num_generations": K, "prompts_per_update": per_update, "clip_epsilon": eps, "kl_beta": beta,
        "learning_rate": float(cfg["learning_rate"]), "max_completion_length": Lmax,
        "mask_truncated_completions": bool(cfg.get("mask_truncated_completions", True)),
        "seed": int(cfg["seed"]), "generated_tokens_total": total_gen_tokens,
        "wall_clock_s": elapsed, "peak_vram_gb": peak_vram_gb(), "train_log": str(log_path),
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    print(f"[GRPO:{run_name}] done in {elapsed/60:.1f} min, peak VRAM {summary['peak_vram_gb']:.2f} GB", flush=True)
    clear_gpu(policy, rm)
    return str(out)


def _kl_tensor(new_lp, ref_lp, mask):
    """Same k3 KL estimator as grpo_policy_loss (masked token mean), differentiable."""
    from common.metrics import masked_mean
    lr = ref_lp - new_lp
    return masked_mean(torch.exp(lr) - lr - 1.0, mask)


def _corr(a, b):
    from common.metrics import safe_corr
    return safe_corr(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
