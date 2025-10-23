#!/bin/bash

# 设置 -e 选项，如果任何一个命令失败，脚本将立即退出
set -e
export MUJOCO_GL="osmesa"
LOG_DIR="logs_black_wxyz_addi_vla_adapter"
mkdir -p "$LOG_DIR"

echo "Starting all 5 processing jobs in parallel..."

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Spatial_1 \
#   --libero_task_suite=libero_spatial \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_spatial1 \
#   --jobs=4 &> "$LOG_DIR/libero_spatial1.log" &


# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Spatial_2 \
#   --libero_task_suite=libero_spatial \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_spatial2 \
#   --jobs=4 &> "$LOG_DIR/libero_spatial2.log" &


# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Spatial_3 \
#   --libero_task_suite=libero_spatial \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_spatial3 \
#   --jobs=4 &> "$LOG_DIR/libero_spatial3.log" &


# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Spatial_4 \
#   --libero_task_suite=libero_spatial \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_spatial4 \
#   --jobs=4 &> "$LOG_DIR/libero_spatial4.log" &


# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Spatial_5 \
#   --libero_task_suite=libero_spatial \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_spatial5 \
#   --jobs=4 &> "$LOG_DIR/libero_spatial5.log" &
# wait

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Long_1 \
#   --libero_task_suite=libero_10 \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_101 \
#   --jobs=4 &> "$LOG_DIR/libero_101.log" &


# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Long_2 \
#   --libero_task_suite=libero_10 \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_102 \
#   --jobs=4 &> "$LOG_DIR/libero_102.log" &



# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Long_3 \
#   --libero_task_suite=libero_10 \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_103 \
#   --jobs=4 &> "$LOG_DIR/libero_103.log" &



# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Long_4 \
#   --libero_task_suite=libero_10 \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_104 \
#   --jobs=4 &> "$LOG_DIR/libero_104.log" &



# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Long_5 \
#   --libero_task_suite=libero_10 \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_105 \
#   --jobs=4 &> "$LOG_DIR/libero_105.log" &

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Goal_1 \
#   --libero_task_suite=libero_goal \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_goal1 \
#   --jobs=4 &> "$LOG_DIR/libero_goal1.log" &

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Goal_2 \
#   --libero_task_suite=libero_goal \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_goal2 \
#   --jobs=4 &> "$LOG_DIR/libero_goal2.log" &

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Goal_3 \
#   --libero_task_suite=libero_goal \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_goal3 \
#   --jobs=4 &> "$LOG_DIR/libero_goal3.log" &

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Goal_4 \
#   --libero_task_suite=libero_goal \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_goal4 \
#   --jobs=4 &> "$LOG_DIR/libero_goal4.log" &

# python libero_to_json_addi.py \
#   --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Goal_5 \
#   --libero_task_suite=libero_goal \
#   --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_goal5 \
#   --jobs=4 &> "$LOG_DIR/libero_goal5.log" &

# wait

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Object_1 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_object1 \
  --jobs=4 &> "$LOG_DIR/libero_object1.log" &

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Object_2 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_object2 \
  --jobs=4 &> "$LOG_DIR/libero_object2.log" &

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Object_3 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_object3 \
  --jobs=4 &> "$LOG_DIR/libero_object3.log" &

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Object_4 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_object4 \
  --jobs=4 &> "$LOG_DIR/libero_object4.log" &

python libero_to_json_addi.py \
  --dataset_dir=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional-VLA-Adapter/Object_5 \
  --libero_task_suite=libero_object \
  --output_dir=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_object5 \
  --jobs=4 &> "$LOG_DIR/libero_object5.log" &

echo "All jobs have completed successfully."