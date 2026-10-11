"""Print the schema (keys, types, shapes) of every course data/cache file.

Run: python -m scripts.inspect_schemas
"""
from __future__ import annotations

import json

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path


def short(v, n=80):
    if isinstance(v, torch.Tensor):
        return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, (list, tuple)):
        inner = short(v[0], 40) if v else ""
        return f"{type(v).__name__}[len={len(v)}] first={inner}"
    if isinstance(v, dict):
        return "dict{" + ", ".join(f"{k}: {short(x, 30)}" for k, x in list(v.items())[:8]) + "}"
    s = repr(v)
    return s[:n] + ("..." if len(s) > n else "")


cfg = load_yaml("configs/base.yaml")
for key, path in cfg["paths"].items():
    p = repo_path(path)
    print(f"\n===== {key}: {path}")
    if not p.exists():
        print("  (missing)")
        continue
    if p.is_dir():
        print("  dir:", sorted(x.name for x in p.iterdir()))
        cfgf = p / "adapter_config.json"
        if cfgf.exists():
            c = json.loads(cfgf.read_text())
            print("  adapter:", {k: c.get(k) for k in ["base_model_name_or_path", "r", "lora_alpha", "target_modules", "task_type"]})
        continue
    if p.suffix == ".jsonl":
        rows = read_jsonl(p)
        print(f"  rows={len(rows)}")
        for k, v in rows[0].items():
            print(f"  {k}: {short(v)}")
        for k in ["stratum", "length_stratum", "variant_type", "benchmark_class", "type"]:
            if k in rows[0]:
                print(f"  counts[{k}]:", pd.Series([r.get(k) for r in rows]).value_counts().to_dict())
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        print(f"  rows={len(df)} cols={list(df.columns)}")
        print(df.head(2).to_string()[:600])
    elif p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu", weights_only=False)
        print(f"  type={type(obj).__name__} len={len(obj) if hasattr(obj, '__len__') else '-'}")
        first = obj[0] if isinstance(obj, list) else obj
        if isinstance(first, dict):
            for k, v in first.items():
                print(f"  {k}: {short(v)}")
