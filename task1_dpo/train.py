from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
import json
import time

from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import cast_trainable_to_fp32, grad_norm, peak_vram_gb, reset_peak_vram
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_overlong(rows, tokenizer, max_length, min_response_tokens=64):
    """Drop pairs whose prompt leaves < min_response_tokens of room (logged; same rule for every condition)."""
    keep = []
    for r in rows:
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True))
        if n <= max_length - min_response_tokens:
            keep.append(r)
    print(f'[DPO] filtered {len(rows) - len(keep)} / {len(rows)} pairs with prompt > {max_length - min_response_tokens} tokens', flush=True)
    return keep


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows = filter_overlong(read_jsonl(path), tokenizer, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    cast_trainable_to_fp32(model)
    gen = torch.Generator()
    gen.manual_seed(int(cfg["seed"]))  # identical data order for every condition
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=gen,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = int(cfg.get("epochs", 1))
    params = trainable_parameters(model)
    device = next(model.parameters()).device

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    total_micro = len(loader) * epochs
    total_steps = (total_micro + accum - 1) // accum
    print(f"[DPO:{run_name}] rows={len(bundle['rows'])} beta={beta} micro_batches={total_micro} "
          f"optimizer_steps={total_steps}", flush=True)

    reset_peak_vram()
    t0 = time.perf_counter()
    step, micro = 0, 0
    window = []  # diagnostics accumulated within one optimizer step
    optimizer.zero_grad(set_to_none=True)
    model.train()

    def flush_step():
        nonlocal step, window
        gn = grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "examples_seen": micro * int(cfg["batch_size"]), "beta": beta,
               "grad_norm": gn, "elapsed_s": time.perf_counter() - t0}
        for k in window[0]:
            rec[k] = float(sum(w[k] for w in window) / len(window))
        append_jsonl(log_path, rec)
        if step % 5 == 0 or step == total_steps:
            print(f"[DPO:{run_name}] step {step}/{total_steps} loss={rec['loss']:.4f} "
                  f"acc={rec['preference_accuracy']:.3f} margin={rec['dpo_margin_mean']:.3f} gn={gn:.3f}", flush=True)
        window = []

    for epoch in range(epochs):
        for chosen, rejected in loader:
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}
            # Reference log-probs: same network with the LoRA adapter disabled (frozen SFT policy).
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            pol_c, _, mask_c = response_sequence_logprobs(model, chosen)
            pol_r, _, mask_r = response_sequence_logprobs(model, rejected)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()
            micro += 1
            window.append({
                "loss": float(loss.item()),
                **{k: float(v.item()) for k, v in diag.items()},
                "chosen_len": float(mask_c.sum(-1).float().mean().item()),
                "rejected_len": float(mask_r.sum(-1).float().mean().item()),
            })
            if micro % accum == 0:
                flush_step()
        if window:  # leftover partial accumulation at the end of the epoch
            flush_step()

    elapsed = time.perf_counter() - t0
    model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "output": str(output),
        "dataset": dataset_path or cfg["paths"]["dpo_standard_train"],
        "n_examples": len(bundle["rows"]),
        "beta": beta,
        "epochs": epochs,
        "optimizer_steps": step,
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "seed": int(cfg["seed"]),
        "lora": cfg["lora"],
        "wall_clock_s": elapsed,
        "peak_vram_gb": peak_vram_gb(),
        "train_log": str(log_path),
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    print(f"[DPO:{run_name}] done in {elapsed/60:.1f} min, saved adapter to {output}", flush=True)
    clear_gpu(model, optimizer)
    return str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
