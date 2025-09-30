#!/bin/bash
# find . -type f -name "*_optim_states.pt" -exec rm {} \;
# lrm_wm2

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

MODEL_NAME="/inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR='work_dirs/qwen2.5-vl-7b-sft-full-baseline-5k-64tokens-10ep'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH=src:$PYTHONPATH
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  src/train/train_sft.py \
    --use_liger False \
    --lora_enable False \
    --use_dora False \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path shell/data/train_rm_5k.json \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --image_min_pixels $((64 * 28 * 28)) \
    --image_max_pixels $((64 * 28 * 28)) \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 171 \
    --save_total_limit 10 \
    --dataloader_num_workers 4