
nvidia-smi 
cd /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/shell

export NNODES=${PET_NNODES}
export NRANK=${PET_NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

export GPUS=2  # for debug, 2gpus per nodes
export BATCH_SIZE=32. # for torchrun, total batchsize = BATCH_SIZE * NNODES
export PER_DEVICE_BATCH_SIZE=1
bash train_debug.sh