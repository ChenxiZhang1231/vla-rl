set -x

# GPUS=${GPUS:-8}
# BATCH_SIZE=${BATCH_SIZE:-128}
# PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GPUS=8
BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/smolvla-0.5b-ft_expert-bf16-20ep-ur-real-bs16-kb'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/ft_smolvla_ur_real.py \
  --model_name_or_path "/SSD_DISK/users/zhangjiahui/SimpleVLA-RL/hugg_models/smolvla_base" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/ur_real_jsonl.json" \
  --overwrite_output_dir True \
  --drop_path_rate 0.0 \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 2210 \
  --save_total_limit 10 \
  --learning_rate 5e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  # 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"