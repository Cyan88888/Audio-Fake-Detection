#!/usr/bin/env bash
# 批量绘制 Exps 下所有混淆矩阵 CSV（*_confusion_matrix.csv），输出到 figures/cm_<相对路径>.png
# 用法：在项目根目录执行 ./scripts/plot_all_confusion_matrices.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "Exps" ]]; then
  echo "Exps/ 不存在，请在项目根目录运行或先完成测试。" >&2
  exit 1
fi

mkdir -p figures

# 你可以通过环境变量覆盖 DPI（默认 220）
DPI="${DPI:-220}"

while IFS= read -r -d '' f; do
  rel="${f#Exps/}"
  safe="${rel//\//_}"
  base="${safe%.csv}"
  out="figures/cm_${base}.png"
  # 标题更短更适合 PPT（需要完整路径可自行改回 ${rel}）
  title="${base}"
  python scripts/plot_confusion_matrix.py --csv "$f" --title "$title" --out "$out" --dpi "$DPI"
done < <(find Exps -name '*confusion_matrix.csv' -print0 2>/dev/null | sort -z)

echo "Done. PNGs under figures/"
