#!/bin/bash

# 设置 -e 选项，如果任何一个命令失败，脚本将立即退出
set -e

LOG_DIR="logs_black"
mkdir -p "$LOG_DIR"

echo "Starting all 5 processing jobs in parallel..."

python libero_to_json.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_spatial_no_noops \
  --libero_task_suite=libero_spatial \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM/libero_spatial \
  --jobs=4 &> "$LOG_DIR/libero_spatial.log" &

python libero_to_json.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_goal_no_noops \
  --libero_task_suite=libero_goal \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM/libero_goal \
  --jobs=4 &> "$LOG_DIR/libero_goal.log" &

python libero_to_json.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_object_no_noops \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM/libero_object \
  --jobs=4 &> "$LOG_DIR/libero_object.log" &

python libero_to_json.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_10_no_noops \
  --libero_task_suite=libero_10 \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM/libero_10 \
  --jobs=4 &> "$LOG_DIR/libero_10.log" &

python libero_to_json.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_90_no_noops \
  --libero_task_suite=libero_90 \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM/libero_90 \
  --jobs=4 &> "$LOG_DIR/libero_90.log" &

# 等待所有后台任务完成
wait

echo "All jobs have completed successfully."