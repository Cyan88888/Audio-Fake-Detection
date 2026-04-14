#!/usr/bin/env bash
set -euo pipefail

# Stage-2 positional encoding ablation
# Run from repository root:
#   bash scripts/run_stage2_posenc.sh

python train.py --conf_dir config/ablation_s2_pe_sine.yaml
python train.py --conf_dir config/ablation_s2_pe_learnable.yaml

# After training, replace <best>.ckpt and run:
# python test.py --conf_dir config/ablation_s2_pe_sine.yaml --ckpt_path Exps/Ablation_S2_pe_sine/checkpoints/<best>.ckpt
# python test.py --conf_dir config/ablation_s2_pe_learnable.yaml --ckpt_path Exps/Ablation_S2_pe_learnable/checkpoints/<best>.ckpt
#
# Then plot:
# python scripts/plot_experiment.py --exp Exps/Ablation_S2_pe_sine
# python scripts/plot_experiment.py --exp Exps/Ablation_S2_pe_learnable
