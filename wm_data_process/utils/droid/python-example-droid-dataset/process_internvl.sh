#!/bin/bash

# 定义输出目录
OUTPUT_DIR="data/processed_jsonl_folder_5hist_2si"

# 并行处理每个场景
for SCENE in AUTOLab CLVR GuptaLab ILIAD IPRL IRIS PennPAL RAD RAIL REAL RPL TRI; do
    echo "Processing scene: $SCENE"
    python process_for_internvl.py --scene "$SCENE" --output "$OUTPUT_DIR/$SCENE" &
done

# 等待所有后台任务完成
wait

echo "Parallel processing completed for all scenes."
