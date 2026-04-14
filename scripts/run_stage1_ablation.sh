#!/usr/bin/env bash
set -euo pipefail

# Stage-1 ablation runner (depth/head/dropout).
# Run from repository root:
#   bash scripts/run_stage1_ablation.sh
#
# Notes:
# 1) test.py needs a ckpt path. Replace <best>.ckpt for each experiment.
# 2) If you only want training, comment out test/plot blocks.

python train.py --conf_dir config/ablation_s1_base.yaml
python train.py --conf_dir config/ablation_s1_l3_d015.yaml
python train.py --conf_dir config/ablation_s1_l4_d020.yaml
python train.py --conf_dir config/ablation_s1_l2_h16_d015.yaml

# Example test commands (edit ckpt file names first):
# python test.py --conf_dir config/ablation_s1_base.yaml --ckpt_path Exps/Ablation_S1_base/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s1_l3_d015.yaml --ckpt_path Exps/Ablation_S1_l3_d015/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s1_l4_d020.yaml --ckpt_path Exps/Ablation_S1_l4_d020/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s1_l2_h16_d015.yaml --ckpt_path Exps/Ablation_S1_l2_h16_d015/checkpoints/<best>.ckpt

# Example plot commands (after test):
# python scripts/plot_experiment.py --exp Exps/Ablation_S1_base
# python scripts/plot_experiment.py --exp Exps/Ablation_S1_l3_d015
# python scripts/plot_experiment.py --exp Exps/Ablation_S1_l4_d020
# python scripts/plot_experiment.py --exp Exps/Ablation_S1_l2_h16_d015
