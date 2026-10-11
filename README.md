# ATML PA2: LLM Post-Training

Name: Hamza Ali Zahid

Roll Number: 28100100

This is my code for ATML PA2. It covers DPO, PPO, GRPO, safety calibration on XSTest, and the RLVR vs RLAIF comparison. It is built on the course starter repository:
<https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>

## Use of AI

- **Code and README.** These were written with coding assistance from Claude (Anthropic), as the assignment's coding-assistance policy allows.
- **Report.** The PDF report was written entirely by me. No generative AI was used for its language, interpretation, or analysis.
- **Manual labels.** The Task 4 manual audit labels in `results/task4_safety/manual_audit_sheet.csv` were assigned by me.
- **Other sources.** No external repository code was copied. I used the course starter code, configs, loaders, metric helpers, the safety judge, and the RLVR verifier and RLAIF judge as provided, except for the three defect fixes listed below. Libraries: PyTorch, Transformers, PEFT, bitsandbytes, pandas, matplotlib.

## Setup

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets    # pinned course data, caches and checkpoints from Hugging Face
python -m scripts.validate_assets
```

Datasets, model weights and trained adapters (`data/`, `cached/`, `checkpoints/`, `outputs/`) are not committed. The small result files in `results/` are committed. All settings are in `configs/*.yaml`. The seed is 6304 everywhere.

## Run everything

The reported results come from one unattended run on Kaggle (2× T4 16 GB, about 5 hours):

```bash
python run/kaggle_all.py
```

This runs `run/gpu0.sh` (Tasks 1 and 4) and `run/gpu1.sh` (Tasks 2, 3 and 5) in parallel, then `scripts/make_figures.py`. Each finished step leaves a marker in `results/.done/`, so a rerun skips finished work. To run the queues by hand, use `CUDA_VISIBLE_DEVICES=0 bash run/gpu0.sh` and `CUDA_VISIBLE_DEVICES=1 bash run/gpu1.sh`.

## Commands per task

**Task 1: DPO**
```bash
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml     # beta = 0.03 / 0.10 / 0.30, first 600 pairs
python -m task1_dpo.analyze_length --config configs/dpo.yaml     # length-balanced model + per-stratum eval
```

**Task 2: PPO**
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part cached   # cached batch, eps = 0.05 / 0.2 / 0.5
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part forks    # 8-update forks per eps
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml                 # beta_KL = 0 / 0.1 / 0.2
```

**Task 3: GRPO**
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml   # K = 2 / 4 / 8 from the cache
python -m task3_grpo.compare_normalization --config configs/grpo.yaml   # canonical GRPO vs Dr. GRPO
```

**Task 4: Safety (XSTest)**
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml
python -m task4_safety.judge_responses    --config configs/feedback.yaml
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml   # after the manual labels are filled
```

**Task 5: RLVR vs RLAIF**
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.compare_feedback    --config configs/feedback.yaml
```

**Tables and figures**
```bash
python -m scripts.make_figures      # writes report/tables/*.csv and report/figures/*.png
```

## Starter defects fixed

| File | Defect | Fix |
|---|---|---|
| `task1_dpo/dpo.py` | The logit used `policy_margin + ref_margin` | Changed to `policy_margin - ref_margin` |
| `task2_ppo/ppo.py` | The surrogate used `torch.maximum`, so clipping never limited the update | Changed to `torch.minimum` |
| `task3_grpo/grpo.py` | Advantages were normalized over the whole batch, ignoring `group_ids` | Changed to normalize within each prompt's group |

## Choices and limitations

- **Reference policy and KL.** The reference policy is the frozen base model, obtained by disabling the LoRA adapter. KL uses the course's sampled-response estimator. I report it as a per-token mean and as a per-response sum.
- **Held-out generation subsets.** These were chosen to fit the compute budget, and every condition uses the same subset.
  - DPO generation: the first 64 prompts of `dpo_standard_eval`, plus the 10 word-limit prompts (greedy decoding).
  - PPO/GRPO: the first 64 prompts of `rl_prompt_pool_eval`.
  - Settings for both: temperature 0.7, top-p 0.9, seed 6304.
  - Preference accuracy uses all 300 standard held-out pairs and all 246 length-stratified pairs.
- **DPO filtering.** I drop pairs whose prompt leaves fewer than 64 tokens of the 768-token limit. That removed 57 of 1,500 standard training pairs. The same rule applies to every DPO run.
- **Forks.** All PPO and GRPO forks start from the same supplied midpoint and use the same fixed prompt order. Each fork runs 8 updates. The DPO β runs use 600 pairs, a shorter budget than the standard run.
- **Dropout.** Dropout is disabled during PPO/GRPO updates, so the importance ratio is exactly 1 before the first step.
- **GRPO masking.** Completions that hit the length cap are masked out of the GRPO loss, as the config specifies.
- **Group-size study.** Each prompt's 8 cached completions are split into 8/K groups, so every K uses the same total generations.
- **Manual audit.** I labelled 60 fixed XSTest IDs, 30 SAFE and 30 UNSAFE, each paired with one policy's response (15 per policy), without seeing the AI labels.
- **Task 5.** Decoding is greedy with a 512-token cap. In the pairwise judge, a win counts 1, a tie 0.5 and a loss 0. Diagnostic pairs are judged in both orders.