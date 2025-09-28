#!/usr/bin/env bash
# super_simple_parallel.sh
# 每个子库后台起一个 Python 进程并行跑，最后 wait

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-utils/droid/droid_to_json.py}"

DROID_ROOT="${DROID_ROOT:-/inspire/ssd/project/robotsimulation/public/data/droid_raw/1.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/droid_unclip}"

LABS=( AUTOLab GuptaLab IPRL PennPAL RAIL RPL WEIRD CLVR ILIAD IRIS RAD REAL TRI )
# LABS=( TRI )

FPS="${FPS:-10}"
# CLIP_LEN="${CLIP_LEN:--1}"
STRIDE="${STRIDE:-20}"
CRF="${CRF:-20}"
PRESET="${PRESET:-veryfast}"
JOBS="${JOBS:-64}"
IMG_W="${IMG_W:-1280}"
IMG_H="${IMG_H:-720}"

LOG_DIR="${LOG_DIR:-./logs_unclip}"
mkdir -p "$LOG_DIR"

ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] 并行启动 ${#LABS[@]} 个子库（全部后台跑，最后统一 wait）"
echo "脚本: $SCRIPT_PATH"
echo

pids=()

for LAB in "${LABS[@]}"; do
  SRC_DIR="${DROID_ROOT}/${LAB}"
  OUT_DIR="${OUTPUT_ROOT}/${LAB}"
  LOG_FILE="${LOG_DIR}/${LAB}.log"

  if [[ ! -d "$SRC_DIR" ]]; then
    echo "[$(ts)] [跳过] 不存在: $SRC_DIR"
    continue
  fi
  mkdir -p "$OUT_DIR"

  (
    echo "[$(ts)] ===== 开始 $LAB ====="
    "$PYTHON_BIN" "$SCRIPT_PATH" \
      --scene_root "$SRC_DIR" \
      --output_dir "$OUT_DIR" \
      --fps "$FPS" \
      --stride "$STRIDE" \
      --crf "$CRF" \
      --preset "$PRESET" \
      --jobs "$JOBS" \
      --image_width "$IMG_W" \
      --image_height "$IMG_H"
    echo "[$(ts)] ===== 完成 $LAB ====="
  ) >"$LOG_FILE" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "[$(ts)] 启动 $LAB, PID=$pid，日志: $LOG_FILE"
done

echo "[$(ts)] 等待所有子任务收尾..."
rc_all=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc_all=1
  fi
done

echo "[$(ts)] 全部完成 ✅"
exit "$rc_all"
