# internvl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# vllm serve /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-full-baseline-5k-64tokens-10ep/checkpoint-1710 \
vllm serve /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-full-libero-total-wm-bs128 \
  --served-model-name "judge" \
  --port 18901 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 8 \
  --distributed-executor-backend mp \
  --disable-log-requests
