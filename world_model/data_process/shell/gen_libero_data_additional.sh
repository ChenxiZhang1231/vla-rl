#!/bin/bash

# 设置 -e 选项，如果任何一个命令失败，脚本将立即退出
set -e
export MUJOCO_GL="osmesa"
LOG_DIR="logs_black_wxyz_addi_fixedbug"
mkdir -p "$LOG_DIR"

echo "Starting all 5 processing jobs in parallel..."

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_1 \
  --libero_task_suite=libero_spatial \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_spatial1 \
  --jobs=4 &> "$LOG_DIR/libero_spatial1.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_2 \
  --libero_task_suite=libero_spatial \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_spatial2 \
  --jobs=4 &> "$LOG_DIR/libero_spatial2.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_3 \
  --libero_task_suite=libero_spatial \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_spatial3 \
  --jobs=4 &> "$LOG_DIR/libero_spatial3.log" &

wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_1 \
  --libero_task_suite=libero_goal \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_goal1 \
  --jobs=4 &> "$LOG_DIR/libero_goal1.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_2 \
  --libero_task_suite=libero_goal \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_goal2 \
  --jobs=4 &> "$LOG_DIR/libero_goal2.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_3 \
  --libero_task_suite=libero_goal \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_goal3 \
  --jobs=4 &> "$LOG_DIR/libero_goal3.log" &

wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_1 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_object1 \
  --jobs=4 &> "$LOG_DIR/libero_object1.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_2 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_object2 \
  --jobs=4 &> "$LOG_DIR/libero_object2.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_3 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_object3 \
  --jobs=4 &> "$LOG_DIR/libero_object3.log" &

wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_1 \
  --libero_task_suite=libero_10 \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_101 \
  --jobs=4 &> "$LOG_DIR/libero_101.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_2 \
  --libero_task_suite=libero_10 \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_102 \
  --jobs=4 &> "$LOG_DIR/libero_102.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_3 \
  --libero_task_suite=libero_10 \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-FIXEDBUG/libero_103 \
  --jobs=4 &> "$LOG_DIR/libero_103.log" &

wait

echo "All jobs have completed successfully."