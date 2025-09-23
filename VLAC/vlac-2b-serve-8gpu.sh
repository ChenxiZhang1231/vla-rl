#!/usr/bin/env bash
set -euo pipefail

# ===== 可配置参数 =====
MODEL_DIR="/inspire/ssd/project/robotsimulation/public/huggingface_models/VLAC"
HOST="0.0.0.0"
BASE_PORT=8000               # 最终端口 = BASE_PORT + GPU_ID
NUM_GPUS=8                   # 启动 8 个进程：GPU 0..7
SERVED_NAME="judge"
INFER_BACKEND="pt"           # 你之前使用的是 pt

# 线程/性能相关（可选，常见优化）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 日志目录
LOG_DIR="./logs_vlac_serve"
mkdir -p "${LOG_DIR}"

# 简单端口占用检测
is_port_free () {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"
  else
    ! netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$"
  fi
}

start_one () {
  local gpu_id="$1"
  local port=$((BASE_PORT + gpu_id))
  local log="${LOG_DIR}/serve_gpu${gpu_id}.log"

  if ! is_port_free "${port}"; then
    echo "[WARN] Port ${port} is busy, skip GPU ${gpu_id}."
    return
  fi

  echo "[INFO] Launching GPU ${gpu_id} on port ${port} ..."
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  swift deploy \
    --model "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${port}" \
    --infer_backend "${INFER_BACKEND}" \
    --served_model_name "${SERVED_NAME}" \
    > "${log}" 2>&1 &

  echo "[INFO]   -> log: ${log}   pid: $!"
}

# ===== 主循环：GPU 0..NUM_GPUS-1 =====
for gid in $(seq 0 $((NUM_GPUS-1))); do
  start_one "${gid}"
  # 如需避免同时加载造成的 I/O 抖动，可加一点点延时：
  sleep 0.5
done

echo "[DONE] Started ${NUM_GPUS} serve processes on ports ${BASE_PORT}..$((BASE_PORT+NUM_GPUS-1))"
