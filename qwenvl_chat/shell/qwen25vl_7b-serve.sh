vllm serve /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-513step \
    --port 18901 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --served-model-name "judge" \
    --disable-log-requests