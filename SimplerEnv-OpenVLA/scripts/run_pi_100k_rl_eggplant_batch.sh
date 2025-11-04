#!/usr/bin/env bash
set -euo pipefail

model_name=pi05

tasks=(
  bridge_eggplant.sh
  # drawer_variant_agg.sh
  # drawer_visual_matching.sh
  # move_near_variant_agg.sh
  # move_near_visual_matching.sh
  # pick_coke_can_variant_agg.sh
  # pick_coke_can_visual_matching.sh
  # put_in_drawer_variant_agg.sh
  # put_in_drawer_visual_matching.sh
)

# 底模 ckpt（不变）

# 外层遍历的“加载用/合并后” ckpt
ckpt_path=/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/100000
load_ckpt_paths=(
  "/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/100000"
  "/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/200000"
)

# 统一的 tag 后缀（steps）
suffixes=(1 2 3 4 5)

# 可选：你的 tag 前缀（自定义，便于筛选）
tag_prefix="bridge_ck5_100k_200k_rl_pi05_fs"

action_ensemble_temp=0.0
device=0
CUDA_VISIBLE_DEVICES=0
DISPLAY=""

for load_ckpt_path in "${load_ckpt_paths[@]}"; do
  ckpt_tag="$(basename "${load_ckpt_path%.*}")"   # e.g., step49

  for task in "${tasks[@]}"; do
    task_base="${task%.sh}"                       # e.g., bridge_carrot

    for s in "${suffixes[@]}"; do
      # 让 tag 显式包含 task 名，避免不同 task 产生同名 tag
      tag="${tag_prefix}_${task_base}_${s}steps_${ckpt_tag}"

      echo "============================"
      echo "CKPT : $load_ckpt_path"
      echo "TASK : $task_base"
      echo "TAG  : $tag"
      echo "============================"
    
      # 结果目录层级：ckpt/tag按task分组，进一步避免覆盖
      logging_dir="results/${ckpt_tag}/${task_base}/${tag}"
      mkdir -p "$logging_dir"

      echo "🚀 running $task ..."
      bash "scripts/$task" \
        "$ckpt_path" "$model_name" "$action_ensemble_temp" "$logging_dir" "$device" "$load_ckpt_path"

      echo "🚀 task finished. Calculating metrics for this tag..."
      python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root "$logging_dir" \
        >>"$logging_dir/total.metrics"
    done
  done
done
