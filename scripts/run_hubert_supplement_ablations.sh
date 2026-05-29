#!/usr/bin/env bash
# 一键串行：HuBERT 补充消融（训练 + 自动选最优 ckpt 测试 + 汇总）
# 用法（仓库根目录）:
#   conda activate safeear
#   nohup bash scripts/run_hubert_supplement_ablations.sh > Exps/_logs/supplement_ablations/nohup.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-Exps/_logs/supplement_ablations}"
mkdir -p "$LOG_DIR"

CONFIGS=(
  config/ablation_supplement/no_pe.yaml
  config/ablation_supplement/heads8.yaml
  config/ablation_supplement/layers1.yaml
  config/ablation_supplement/layers3.yaml
  config/ablation_supplement/bilstm.yaml
)

pick_best_ckpt() {
  local exp_path="$1"
  "$PYTHON" - <<PY
from pathlib import Path
import re
import sys

exp = Path("${exp_path}")
ckpt_dir = exp / "checkpoints"
if not ckpt_dir.is_dir():
    sys.exit(1)
cands = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
if not cands:
    sys.exit(1)

def score(p: Path) -> float:
    if p.name.startswith("last"):
        return 1e9
    # Do not use [\d.]+ — it would swallow the dot before ".ckpt" (e.g. 0.0008.)
    m = re.search(r"val_eer=([0-9]+(?:\.[0-9]+)?)", p.name)
    if m:
        return float(m.group(1))
    return 1e8

print(str(min(cands, key=score)))
PY
}

run_one() {
  local conf="$1"
  local tag
  tag="$(basename "$conf" .yaml)"
  local log_train="${LOG_DIR}/${tag}_train.log"
  local log_test="${LOG_DIR}/${tag}_test.log"

  local exp_path
  exp_path="$("$PYTHON" - <<PY
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load("${conf}")
print(os.path.normpath(os.path.join(str(cfg.exp.dir), str(cfg.exp.name))))
PY
)"

  if [[ -f "${exp_path}/test_results.json" ]]; then
    echo "[SKIP] ${conf} — test_results.json exists"
    return 0
  fi

  echo ""
  echo "========== [$(date '+%F %T')] TRAIN ${conf} =========="
  "$PYTHON" train.py --conf_dir "$conf" 2>&1 | tee "$log_train"

  local ckpt
  if ! ckpt="$(pick_best_ckpt "$exp_path")"; then
    echo "[ERROR] No checkpoint under ${exp_path}/checkpoints" | tee -a "$log_test"
    echo "[WARN] Skip TEST for ${conf}, continue next experiment." | tee -a "$log_test"
    return 0
  fi
  echo "[INFO] Using checkpoint: ${ckpt}"

  echo "========== [$(date '+%F %T')] TEST ${conf} =========="
  "$PYTHON" test.py --conf_dir "$conf" --ckpt_path "$ckpt" 2>&1 | tee "$log_test"
}

echo "[START] Hubert supplement ablations at $(date '+%F %T')"
echo "[INFO] Logs: ${LOG_DIR}"

for conf in "${CONFIGS[@]}"; do
  run_one "$conf" || echo "[WARN] run_one failed for ${conf}, continue."
done

echo ""
echo "========== [$(date '+%F %T')] AGGREGATE RESULTS =========="
"$PYTHON" scripts/aggregate_test_results.py --exps-dir Exps --out-dir Exps/_summary 2>&1 | tee "${LOG_DIR}/aggregate.log"

echo "[DONE] All supplement runs finished at $(date '+%F %T')"
echo "See: Exps/_summary/all_test_results.md (rows: Ablation_Hubert_supp_*)"
