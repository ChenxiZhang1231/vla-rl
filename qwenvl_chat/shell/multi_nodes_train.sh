
nvidia-smi 
source /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/miniconda3/etc/profile.d/conda.sh
conda activate lrm_wm2

cd /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat

export NNODES=${PET_NNODES}
export NRANK=${PET_NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

export GPUS=8
export BATCH_SIZE=32 # for torchrun, total batchsize = BATCH_SIZE * NNODES
export PER_DEVICE_BATCH_SIZE=1
bash shell/train_debug.sh