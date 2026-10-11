from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.models import clear_gpu
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
        print(f"  {policy_name}: {min(start + batch_size, len(df))}/{len(df)}", flush=True)
    clear_gpu(model)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", choices=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    for name in args.policies or list(policy_specs(cfg)):
        path = outdir / f"generated_{name}.jsonl"
        if args.skip_existing and path.exists():
            print("Reusing", path)
            continue
        adapter = policy_specs(cfg)[name]
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing frozen {name} adapter: {adapter}")
        # Greedy decoding, common max length, fixed CSV order (generate_for_policy iterates the CSV in order).
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(path, records)
        print(f"Saved {len(records)} responses -> {path}")
        clear_gpu()


if __name__ == "__main__":
    main()
