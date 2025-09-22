# #!/bin/bash

# # 定义输出目录
# OUTPUT_DIR="data/processed_jsonl_folder_20ws_10bs"

# # 并行处理每个场景
# for SCENE in AUTOLab CLVR GuptaLab ILIAD IPRL IRIS PennPAL RAD RAIL REAL RPL TRI; do
#     echo "Processing scene: $SCENE"
#     python process_for_internvl_mb.py --scene "$SCENE" --output "$OUTPUT_DIR/$SCENE" &
# done

# # 等待所有后台任务完成
# wait

# echo "Parallel processing completed for all scenes."

#!/bin/bash

# 定义输出目录
OUTPUT_DIR="data/processed_jsonl_folder_20ws_10bs"

# 定义可用 GPU 数量
NUM_GPUS=8

# 场景列表
SCENES=("AUTOLab" "CLVR" "GuptaLab" "ILIAD" "IPRL" "IRIS" "PennPAL" "RAD" "RAIL" "REAL" "RPL" "TRI")

# 处理每个场景，分配到不同 GPU
INDEX=0
for SCENE in "${SCENES[@]}"; do
    GPU_ID=$((INDEX % NUM_GPUS))  # 轮流分配 GPU
    echo "Processing scene: $SCENE on GPU $GPU_ID"
    
    # 运行任务，并将其分配到指定 GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python process_for_internvl_mb.py --scene "$SCENE" --output "$OUTPUT_DIR/$SCENE" &

    ((INDEX++))  # 增加索引
done

# 等待所有后台任务完成
wait

echo "Parallel processing completed for all scenes."

