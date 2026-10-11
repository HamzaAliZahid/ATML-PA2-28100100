from __future__ import annotations

import argparse

from common.data import load_yaml
from common.rl_eval import evaluate_rl_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT/reference policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_rl_policy(cfg, adapter, args.name)


if __name__ == "__main__":
    main()
