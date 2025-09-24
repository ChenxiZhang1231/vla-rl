
python tool/merge_lora.py \
    --input_path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/checkpoint-513 \
    --model-base $MODEL_NAME  \
    --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-513step \
    --safe-serialization


python src/merge_lora_weights.py \
    --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/checkpoint-684 \
    --model-base $MODEL_NAME  \
    --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-684step \
    --safe-serialization

python src/merge_lora_weights.py \
    --model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/checkpoint-855 \
    --model-base $MODEL_NAME  \
    --save-model-path /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/qwen2.5-vl-7b-sft-lora-baseline-5k-r128-merge-855step \
    --safe-serialization
