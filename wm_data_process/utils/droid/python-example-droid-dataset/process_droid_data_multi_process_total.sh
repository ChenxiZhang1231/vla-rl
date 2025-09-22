#!/bin/bash

# 定义数据集名称数组
datasets=("TRI" "CLVR" "GuptaLab" "ILIAD" "IPRL" "IRIS" "PennPAL" "RAD" "RAIL" "REAL" "RPL" "WEIRD" "AUTOLab")

# 遍历每个数据集并顺序处理
for dataset in "${datasets[@]}"; do
  echo "Processing dataset: $dataset"
  python src/process_raw_data_multi_process.py \
    --scene data/1.0.1/"$dataset" \
    --save_path data/droid_processed_data_debug/ \
    --urdf franka_description/panda.urdf \
    --num_threads 20


done
