#!/bin/bash

MODEL_NAME="/inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-r128-bridge-4tasks/checkpoint-222 \
    --model-base $MODEL_NAME  \
    --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-r128-bridge-4tasks-lora-merge/checkpoint-222 \
    --safe-serialization

# python src/merge_lora_weights.py \
#     --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/checkpoint-171 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-171step \
#     --safe-serialization


# python src/merge_lora_weights.py \
#     --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/checkpoint-342 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-342step \
#     --safe-serialization


# python src/merge_lora_weights.py \
#     --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/checkpoint-513 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-513step \
#     --safe-serialization


# python src/merge_lora_weights.py \
#     --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/checkpoint-684 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-684step \
#     --safe-serialization

# python src/merge_lora_weights.py \
#     --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/checkpoint-855 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-64tokens/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-855step \
#     --safe-serialization
