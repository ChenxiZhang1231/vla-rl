#!/bin/bash

MODEL_NAME="/inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-32B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-32b-sft-lora-baseline/checkpoint-75 \
    --model-base $MODEL_NAME  \
    --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-32b-sft-lora-baseline/qwen2.5-vl-32b-sft-lora-baseline-merge-75step \
    --safe-serialization
