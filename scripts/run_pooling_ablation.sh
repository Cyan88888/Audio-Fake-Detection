#!/usr/bin/env bash
set -euo pipefail

# Pooling ablation on top of stage-2 best architecture.
# Run from repository root:
#   bash scripts/run_pooling_ablation.sh

python train.py --conf_dir config/ablation_pool_attn.yaml
python train.py --conf_dir config/ablation_pool_mean.yaml
python train.py --conf_dir config/ablation_pool_max.yaml
python train.py --conf_dir config/ablation_pool_meanmax.yaml

# After training, replace <best>.ckpt for each:
# python test.py --conf_dir config/ablation_pool_attn.yaml --ckpt_path Exps/Ablation_Pool_attn/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_pool_mean.yaml --ckpt_path Exps/Ablation_Pool_mean/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_pool_max.yaml --ckpt_path Exps/Ablation_Pool_max/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_pool_meanmax.yaml --ckpt_path Exps/Ablation_Pool_meanmax/checkpoints/<best>.ckpt

# Plot:
# python scripts/plot_experiment.py --exp Exps/Ablation_Pool_attn
# python scripts/plot_experiment.py --exp Exps/Ablation_Pool_mean
# python scripts/plot_experiment.py --exp Exps/Ablation_Pool_max
# python scripts/plot_experiment.py --exp Exps/Ablation_Pool_meanmax
