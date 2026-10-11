#!/usr/bin/env bash
# GPU 1: Task 2 (PPO) + Task 3 (GRPO) standard runs first (Task 4 waits for them), then ablations, then Task 5.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
source run/common.sh

step t2_train_standard python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
step t3_train_standard python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
step t3_group_size     python -m task3_grpo.analyze_group_size --config configs/grpo.yaml   # cache only, fast

step t2_eval_standard  python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
step t2_eval_midpoint  python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
step t3_eval_standard  python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
step t3_eval_midpoint  python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint

step t2_clip_cached    python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part cached
step t2_clip_forks     python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part forks --skip-existing
step t2_kl_forks       python -m task2_ppo.ablate_kl --config configs/ppo.yaml --skip-existing
step t3_normalization  python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing

step t5_gsm            python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm --skip-existing
step t5_transfer       python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer --skip-existing
step t5_diagnostics    python -m task5_feedback.score_perturbations --config configs/feedback.yaml
step t5_compare        python -m task5_feedback.compare_feedback --config configs/feedback.yaml
echo "GPU1 queue finished."
