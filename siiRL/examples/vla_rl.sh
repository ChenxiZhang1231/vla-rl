set -x

ray stop --force

export NCCL_DEBUG=WARN 
export WANDB_API_KEY='f7f09bc0f061da63632aca9baae8551ecc66e66a'
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
# export NCCL_TIMEOUT=86400          # 设置 24 小时超时（防止误杀）
# export NCCL_ASYNC_ERROR_HANDLING=0 # 禁用异步错误处理（避免超时自动终止）

# export PATH=$PATH:/opt/conda/envs/openvla-oft/bin/gcc

# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PROJECT_DIR=/inspire/hdd/global_user/liao-liao/workspace/siiRL-VLA
PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME='long_test_8+8GPU' #'MODIFIED YOURSELF e.g. vla-lib10_model10j_lr10_tmp16_nsample8_clip08-128_batch64_ppominibs128_node2' 

VIDEO_EMBEDDING_MODEL_PATH="/inspire/hdd/global_user/liao-liao/models/vitg-384.pt"

# SFT_MODEL_PATH="/inspire/hdd/global_user/liao-liao/models/Haozhan72/Openvla-oft-SFT-libero10-traj1"
# CKPT_PATH="$PROJECT_DIR/ckpt/libero_long"
# DATASET_NAME="libero_10"

# For openvla-oft Libero-Long traj1 SFT or traj all SFT models can be find in https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86
# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
SFT_MODEL_PATH="/inspire/hdd/global_user/liao-liao/models/Sylvest/libero_goal_0823"
CKPT_PATH="$PROJECT_DIR/ckpt/libero_goal"
DATASET_NAME="libero_goal"

VLA_NAME="openvla-oft"

ALIGN_PATH="$PROJECT_DIR/align.json"

export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_MASTER_ADDR=$MASTER_ADDR

NUM_GPUS=$N_GPUS_PER_NODE
# # If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES=$NNODES

export PROJECT_NAME=siirl_vla_ppo_${DATASET_NAME}_${MUJOCO_GL}
export EXPERIMENT_NAME=siirl_vla_ppo_${VLA_NAME}_${DATASET_NAME}_${NUM_NODES}nodes

# 启动Ray集群（仅在多节点时需要）
start_ray_cluster() {
    if [ "$NNODES" -gt 1 ]; then
        # 多节点环境，主节点启动Ray head，从节点启动Ray worker
        if [ "$NODE_RANK" = "0" ]; then
            ray start --head --port=$RAY_MASTER_PORT --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus $N_GPUS_PER_NODE
        else
            sleep 10
            ray start --address=$RAY_MASTER_ADDR:$RAY_MASTER_PORT --num-gpus $N_GPUS_PER_NODE --block
        fi
    fi
}

# 启动训练（根据环境判断）
start_training() {
    # 只需在主节点（rank0）或单节点上启动
    if [ "$NNODES" -eq 1 ] || [ "$NODE_RANK" = "0" ]; then
        # data.n_samples=8  原始  data.train_batch_size=64 log_prob_micro_batch_size=32

        HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
            data.task_suite_name=$DATASET_NAME \
            data.num_trials_per_task=50 \
            data.n_samples=8 \
            data.filter_accuracy=True \
            data.accuracy_lower_bound=0.1 \
            data.accuracy_upper_bound=0.9 \
            data.oversample_factor=1 \
            data.train_batch_size=64 \
            data.val_batch_size=496 \
            data.max_prompt_length=256 \
            data.max_response_length=128 \
            actor_rollout_ref.model.path=$SFT_MODEL_PATH \
            actor_rollout_ref.model.vla=$VLA_NAME \
            actor_rollout_ref.model.action_token_len=7 \
            actor_rollout_ref.model.action_chunks_len=8 \
            actor_rollout_ref.actor.optim.lr=5e-6 \
            actor_rollout_ref.actor.optim.warmup_style=constant \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.ppo_micro_batch_size=8 \
            actor_rollout_ref.actor.use_dynamic_bsz=False \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.grad_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.actor.grad_clip=1 \
            actor_rollout_ref.actor.clip_ratio_high=0.28 \
            actor_rollout_ref.actor.clip_ratio_low=0.2 \
            actor_rollout_ref.actor.num_images_in_input=1 \
            actor_rollout_ref.actor.traj_mini_batch_size=16 \
            actor_rollout_ref.model.enable_gradient_checkpointing=False \
            actor_rollout_ref.model.use_remove_padding=False \
            actor_rollout_ref.actor.entropy_coeff=0. \
            actor_rollout_ref.rollout.num_images_in_input=1 \
            actor_rollout_ref.rollout.val_micro_batch_size=8 \
            actor_rollout_ref.rollout.temperature=1.6 \
            actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
            actor_rollout_ref.rollout.micro_batch_size=1 \
            actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
            actor_rollout_ref.rollout.model_family=openvla \
            actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
            actor_rollout_ref.rollout.num_steps_wait=10 \
            actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
            actor_rollout_ref.rollout.center_crop=True \
            actor_rollout_ref.rollout.max_prompt_length=512 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=hf \
            actor_rollout_ref.rollout.embedding_model_path=$VIDEO_EMBEDDING_MODEL_PATH \
            actor_rollout_ref.rollout.embedding_model_offload=True \
            actor_rollout_ref.rollout.embedding_enable_fp16=True \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
            actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=False \
            algorithm.kl_ctrl.kl_coef=0.00 \
            trainer.logger=['console','wandb'] \
            trainer.project_name=$PROJECT_NAME \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
            trainer.n_gpus_per_node=$NUM_GPUS \
            trainer.nnodes=$NUM_NODES \
            trainer.save_freq=4 \
            trainer.test_freq=4 \
            trainer.total_epochs=100 \
            trainer.val_only=False \
            algorithm.adv_estimator=grpo \
            algorithm.adv_params.verifier_gamma=1.0 \
            algorithm.adv_params.reward_model_gamma=1.0 \
            trainer.runtime_env=$ALIGN_PATH \
            trainer.wandb_mode=offline \
            trainer.val_before_train=True $@
    fi
}


# 启动Ray集群（仅在多节点环境下启动）
start_ray_cluster

# 启动训练任务（仅在rank0或单节点时启动）
start_training
