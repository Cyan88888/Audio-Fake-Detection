#!/usr/bin/env bash
set -euo pipefail

# Stage-3 (augmentation + regularization) on top of stage-2 best setup.
# Run from repository root:
#   bash scripts/run_stage3_aug_reg.sh

python train.py --conf_dir config/ablation_s3_aug_time_mask.yaml
python train.py --conf_dir config/ablation_s3_aug_time_mask_ls005.yaml
python train.py --conf_dir config/ablation_s3_aug_time_mask_ls005_clip1.yaml

# Replace <best>.ckpt for each experiment before running tests:
# python test.py --conf_dir config/ablation_s3_aug_time_mask.yaml --ckpt_path Exps/Ablation_S3_aug_time_mask/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s3_aug_time_mask_ls005.yaml --ckpt_path Exps/Ablation_S3_aug_time_mask_ls005/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s3_aug_time_mask_ls005_clip1.yaml --ckpt_path Exps/Ablation_S3_aug_time_mask_ls005_clip1/checkpoints/<best>.ckpt

# Plot after test:
# python scripts/plot_experiment.py --exp Exps/Ablation_S3_aug_time_mask
# python scripts/plot_experiment.py --exp Exps/Ablation_S3_aug_time_mask_ls005
# python scripts/plot_experiment.py --exp Exps/Ablation_S3_aug_time_mask_ls005_clip1
