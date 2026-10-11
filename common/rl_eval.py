"""Common held-out evaluation protocol for the online RL tasks (PPO/GRPO).

Every condition is evaluated on the same fixed prompt subset (first `eval_prompts` rows of
data/rl_prompt_pool_eval.jsonl), the same decoding settings, the same seed and the same
generation cap, so results are directly comparable within a task.
"""
from __future__ import annotations

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.rl_utils import evaluate_generations, save_eval


def eval_rows(cfg):
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    return rows[: int(cfg.get("eval_prompts", 64))]


def evaluate_rl_policy(cfg, adapter: str | None, name: str, skip_existing: bool = False):
    out_dir = repo_path(cfg["results_dir"])
    path = out_dir / f"eval_{name}.json"
    if skip_existing and path.exists():
        print("Reusing", path)
        return load_json(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    metrics, records = evaluate_generations(
        policy, tokenizer, rm, rm_tok, eval_rows(cfg),
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg.get("eval_max_response_length", 768)),
        batch_size=int(cfg.get("eval_batch_size", 4)),
        seed=int(cfg["seed"]),
        temperature=float(cfg["generation"]["temperature"]),
        top_p=float(cfg["generation"]["top_p"]),
        do_sample=bool(cfg["generation"]["do_sample"]),
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
    )
    save_eval(out_dir, name, metrics, records, extra={"adapter": adapter})
    clear_gpu(policy, rm)
    metrics["adapter"] = adapter
    return metrics
