#!/usr/bin/env bash
# 一键启动 AudioForgeryDet Web 检测服务并在本机浏览器中打开页面。
#
# 默认模型：Exps/Search_PoolMax_S3_ls002_hubert/checkpoints/epoch=4-val_eer=0.0009.ckpt
# 环境变量（新前缀 SPOOFDET_*，仍兼容 SAFEAR_*）：
#   SPOOFDET_DEVICE / SPOOFDET_CKPT / SPOOFDET_FEAT / SPOOFDET_HUBERT
#   SPOOFDET_WEB_HOST / SPOOFDET_WEB_PORT / SPOOFDET_WEB_FIXED_THRESHOLD_SPOOF
#   PYTHON                             解释器，默认 python3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate spoofdet 2>/dev/null || conda activate safeear 2>/dev/null || true
  fi
fi

DEFAULT_CKPT="${ROOT}/Exps/Search_PoolMax_S3_ls002_hubert/checkpoints/epoch=4-val_eer=0.0009.ckpt"
export SPOOFDET_CKPT="${SPOOFDET_CKPT:-${SAFEAR_CKPT:-${DEFAULT_CKPT}}}"
export SPOOFDET_FEAT="${SPOOFDET_FEAT:-${SAFEAR_FEAT:-hubert}}"
export SPOOFDET_HUBERT="${SPOOFDET_HUBERT:-${SAFEAR_HUBERT:-${ROOT}/model_zoos/hubert_base_ls960.pt}}"
export SPOOFDET_WEB_FIXED_THRESHOLD_SPOOF="${SPOOFDET_WEB_FIXED_THRESHOLD_SPOOF:-${SAFEAR_WEB_FIXED_THRESHOLD_SPOOF:-0.5}}"

PYTHON="${PYTHON:-python3}"
HOST="${SPOOFDET_WEB_HOST:-${SAFEAR_WEB_HOST:-0.0.0.0}}"
PORT="${SPOOFDET_WEB_PORT:-${SAFEAR_WEB_PORT:-8080}}"

if [[ -z "${SPOOFDET_DEVICE:-}${SAFEAR_DEVICE:-}" ]] && ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export SPOOFDET_DEVICE=cpu
fi

HEALTH_WAIT_SEC=180
if [[ "${SPOOFDET_DEVICE:-${SAFEAR_DEVICE:-}}" == "cpu" ]]; then
  HEALTH_WAIT_SEC=600
fi
BASE_URL="http://127.0.0.1:${PORT}/"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]]; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

open_browser() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  "${PYTHON}" -c "import webbrowser; webbrowser.open('${url}')" 2>/dev/null || true
}

wait_for_health() {
  local max_wait="${1:-180}"
  echo "等待服务就绪（含模型加载，最多 ${max_wait}s）..."
  "${PYTHON}" - "${HEALTH_URL}" "${max_wait}" <<'PY'
import json, sys, time, urllib.request
url, max_wait = sys.argv[1], int(sys.argv[2])
for _ in range(max_wait):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status != 200:
                time.sleep(1)
                continue
            body = json.loads(r.read().decode())
            if body.get("status") == "ok":
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
sys.exit(1)
PY
}

echo "项目目录: ${ROOT}"
echo "监听: ${HOST}:${PORT}（浏览器将打开 ${BASE_URL}）"
echo "模型 CKPT: ${SPOOFDET_CKPT}"
echo "特征前端: ${SPOOFDET_FEAT}  HuBERT: ${SPOOFDET_HUBERT}"
echo "推理设备: ${SPOOFDET_DEVICE:-${SAFEAR_DEVICE:-（自动：有 GPU 用 cuda，否则 cpu）}}"
echo "按 Ctrl+C 停止服务"
echo ""

"${PYTHON}" -m uvicorn web.api:app --host "${HOST}" --port "${PORT}" &
UVICORN_PID=$!

if wait_for_health "${HEALTH_WAIT_SEC}"; then
  echo "服务已就绪，正在打开浏览器..."
  open_browser "${BASE_URL}"
else
  echo "警告: 在超时内未能连上 ${HEALTH_URL}，仍尝试打开浏览器。" >&2
  open_browser "${BASE_URL}"
fi

wait "${UVICORN_PID}"
