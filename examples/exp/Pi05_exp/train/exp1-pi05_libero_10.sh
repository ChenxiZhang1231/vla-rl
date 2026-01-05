set -x
# ============================================================================
# Pi05 + Libero 环境训练脚本
# ============================================================================
# 使用说明：
# 1. 确保 LIBERO 环境已安装: pip install libero-libero
# 2. 确保数据路径正确
# 3. 根据需要调整 GPU 数量和其他超参数
# ============================================================================

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export ACCELERATE_USE_FSDP=true
export FSDP_CPU_RAM_EFFICIENT_LOADING=true
export MUJOCO_GL="egl"

# ============================================================================
# 项目配置
# ============================================================================
PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME='exp1-pi05_libero_10'

# ============================================================================
# 模型路径配置
# ============================================================================
# Pi05 SFT 模型路径 - 请替换为你自己的模型路径
SFT_MODEL_PATH="/inspire/hdd/project/robotsimulation/public/models/openpi05/pi05_libero/pi05_libero_0104/200000"

# 检查点保存路径
CKPT_PATH="work_dirs/$PROJECT_NAME/$EXPERIMENT_NAME"

# ============================================================================
# Libero 数据集配置
# ============================================================================
# DATASET_NAME 可选值:
#   - libero_10      (Libero-Long, 10个任务)
#   - libero_90      (90个任务)
#   - libero_spatial (10个空间推理任务)
#   - libero_object (10个物体操作任务)
#   - libero_goal    (10个目标导向任务)
DATASET_NAME="libero_10"

# Libero 原始数据路径 (用于加载demo数据)
# 如果为空字符串 "", 则直接从libero库获取初始状态
DATASET_PATH="/inspire/ssd/project/robotsimulation/public/data/LIBERO-datasets"

# ============================================================================
# VLA 模型配置
# ============================================================================
VLA_NAME="pi05"  # 使用 pi05 模型

# ============================================================================
# 训练资源配置
# ============================================================================
NUM_GPUS=8
NUM_NODES=1

# Ray 运行时环境配置 (可选)
ALIGN_PATH="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/align.json"

# ============================================================================
# World Model 配置 (可选 - 如果使用 World Model 则取消注释)
# ============================================================================
# DIT_PATH="/path/to/your/world_model.pt"
# VAE_FOLDER="/path/to/your/vae_folder"

# ============================================================================
# 执行训练命令
# ============================================================================
HYDRA_FULL_ERROR=1 python -m verl_vla.trainer.main_ppo \
    # ==================== 数据配置 ====================
    data.task_suite_name=$DATASET_NAME \
    data.libero_raw_data_dir=$DATASET_PATH \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=32 \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    # ==================== 模型配置 ====================
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=5 \
    # ==================== Actor 优化配置 ====================
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.params=full \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.dlogp_clamp=True \
    actor_rollout_ref.actor.dlogp_clamp_max=4.0 \
    actor_rollout_ref.actor.dlogp_clamp_min=-4.0 \
    actor_rollout_ref.actor.kl_loss_type=kl_ffp \
    actor_rollout_ref.actor.k_baseline_eta=0.1 \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    algorithm.kl_ctrl.kl_coef=0.04 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.actor.unnorm_key=$DATASET_NAME \
    # ==================== Rollout 配置 ====================
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
    actor_rollout_ref.rollout.reward_type=env \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_gpus_per_node=$NUM_GPUS \
    # ==================== Ref 策略配置 ====================
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.vla=$VLA_NAME \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.ref.fsdp_config.model_dtype=float32 \
    # ==================== 奖励模型配置 ====================
    # 奖励类型可选: rule (使用环境完成标志), vlm_serve (使用VLM判断)
    reward_model.type=rule \
    reward_model.return_env_score=True \
    reward_model.unnorm_key=$DATASET_NAME \
    # 如果使用 VLM 奖励，取消下面的注释并注释掉上面的 rule 配置
    # reward_model.type=vlm_serve \
    # reward_model.vlm_input_num_frames=30 \
    # reward_model.vote_n=5 \
    # reward_model.vote_m=3 \
    # reward_model.temperature=0.6 \
    # reward_model.top_p=0.9 \
    # reward_model.return_env_score=False \
    # reward_model.unnorm_key=$DATASET_NAME \
    # ==================== World Model 配置 (可选) ====================
    # actor_rollout_ref.world_model.dit_path=$DIT_PATH \
    # actor_rollout_ref.world_model.vae_folder=$VAE_FOLDER \
    # actor_rollout_ref.world_model.num_sampling_step=10 \
    # actor_rollout_ref.world_model.use_cuda_graphs=True \
    # actor_rollout_ref.world_model.fsdp_config.model_dtype=bfloat16 \
    # actor_rollout_ref.world_model.use_history=True \
    # actor_rollout_ref.world_model.history_video_length=60 \
    # actor_rollout_ref.world_model.unnorm_key=$DATASET_NAME \
    # ==================== 训练器配置 ====================
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5000 \
    trainer.total_epochs=1000 \
    trainer.val_only=False \
    ray_init.num_cpus=32 \
    # ==================== 算法配置 ====================
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=False \
    2>&1 | tee -a "${EXPERIMENT_NAME}.log"
