#!/usr/bin/env bash
# batch_droid_to_json.sh
# 批量运行 droid_to_json.py 覆盖多个实验室子库

set -euo pipefail

########################################
#            配置区（可改）            #
########################################

# Python 解释器与脚本路径
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-utils/droid/droid_to_json.py}"

# DROID 原始根目录版本（子库在其下）
DROID_ROOT="${DROID_ROOT:-/inspire/ssd/project/robotsimulation/public/data/droid_raw/1.0.1}"

# 输出根目录（每个子库会落到 ${OUTPUT_ROOT}/${LAB}）
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/droid}"

# 处理的子库列表（确保这些目录都在 ${DROID_ROOT} 下）
LABS=(
  AUTOLab GuptaLab IPRL PennPAL RAIL RPL WEIRD CLVR ILIAD IRIS RAD REAL TRI
)

# 转码 / 切片参数（等价于 main() 里的 argparse 默认/自定义）
FPS="${FPS:-10}"
CLIP_LEN="${CLIP_LEN:-30}"
STRIDE="${STRIDE:-5}"
CRF="${CRF:-20}"
PRESET="${PRESET:-veryfast}"
JOBS="${JOBS:-128}"
IMG_W="${IMG_W:-1280}"
IMG_H="${IMG_H:-720}"

# 日志目录
LOG_DIR="${LOG_DIR:-./logs}"

########################################
#              运行逻辑                #
########################################

mkdir -p "${LOG_DIR}"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] 开始批处理，共 ${#LABS[@]} 个子库"
echo "脚本: ${SCRIPT_PATH}"
echo "源根: ${DROID_ROOT}"
echo "输出: ${OUTPUT_ROOT}"
echo "参数: fps=${FPS}, clip_len=${CLIP_LEN}, stride=${STRIDE}, crf=${CRF}, preset=${PRESET}, jobs=${JOBS}, image=${IMG_W}x${IMG_H}"
echo

# 逐个子库处理
for LAB in "${LABS[@]}"; do
  SRC_DIR="${DROID_ROOT}/${LAB}"
  OUT_DIR="${OUTPUT_ROOT}/${LAB}"
  LOG_FILE="${LOG_DIR}/${LAB}.log"

  echo "[$(ts)] >>> 准备处理 ${LAB}"
  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[$(ts)] [跳过] 未找到目录: ${SRC_DIR}" | tee -a "${LOG_FILE}"
    continue
  fi

  # 提前建立输出目录
  mkdir -p "${OUT_DIR}"

  echo "[$(ts)] 运行：${LAB} -> ${OUT_DIR} （日志：${LOG_FILE}）"
  {
    echo "[$(ts)] ===== 开始 ${LAB} ====="
    "${PYTHON_BIN}" "${SCRIPT_PATH}" \
      --scene_root "${SRC_DIR}" \
      --output_dir "${OUT_DIR}" \
      --fps "${FPS}" \
      --clip_len "${CLIP_LEN}" \
      --stride "${STRIDE}" \
      --crf "${CRF}" \
      --preset "${PRESET}" \
      --jobs "${JOBS}" \
      --image_width "${IMG_W}" \
      --image_height "${IMG_H}"
    echo "[$(ts)] ===== 完成 ${LAB} ====="
  } 2>&1 | tee "${LOG_FILE}"

  echo "[$(ts)] <<< 完成 ${LAB}"
  echo
done

echo "[$(ts)] 全部完成 ✅ 日志位于: ${LOG_DIR}"
