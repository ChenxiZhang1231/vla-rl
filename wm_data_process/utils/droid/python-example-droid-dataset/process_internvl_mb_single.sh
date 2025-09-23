#!/bin/bash

# 定义输出目录
OUTPUT_DIR="data/processed_jsonl_folder_20ws_10bs"
SCENE="AUTOLab"

echo "Processing scene: $SCENE"
python process_for_internvl_mb.py --scene "$SCENE" --output "$OUTPUT_DIR/$SCENE"



echo "Parallel processing completed for all scenes."
