#!/usr/bin/env bash
# GPU 0: Task 1 (DPO) -> Task 4 (safety; needs the standard PPO/GRPO adapters from gpu1.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
source run/common.sh

step t1_train_standard python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
step t1_eval_standard  python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
step t1_eval_sft       python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft

# Task 4 as early as possible: the manual audit needs your time.
step t4_gen_sft_dpo    python -m task4_safety.generate_responses --config configs/feedback.yaml --policies sft dpo --skip-existing
wait_for outputs/task2_ppo/standard/adapter_config.json
wait_for outputs/task3_grpo/standard/adapter_config.json
step t4_gen_ppo_grpo   python -m task4_safety.generate_responses --config configs/feedback.yaml --policies ppo grpo --skip-existing
step t4_audit_sheet    python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
step t4_judge          python -m task4_safety.judge_responses --config configs/feedback.yaml
step t4_aggregate      python -m task4_safety.evaluate_safety --config configs/feedback.yaml

step t1_beta_sweep     python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-existing
step t1_length_study   python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-train
echo "GPU0 queue finished."
