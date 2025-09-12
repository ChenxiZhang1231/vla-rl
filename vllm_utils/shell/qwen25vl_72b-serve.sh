vllm serve /inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --served-model-name "judge" \
    --disable-log-requests