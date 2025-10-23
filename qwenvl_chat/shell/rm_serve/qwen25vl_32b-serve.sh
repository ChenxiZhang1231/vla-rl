# vllm serve work_dirs/qwen2.5-vl-32b-sft-lora-baseline/qwen2.5-vl-32b-sft-lora-baseline-merge-60step \
vllm serve /inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-32B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --served-model-name "judge" \
    --disable-log-requests