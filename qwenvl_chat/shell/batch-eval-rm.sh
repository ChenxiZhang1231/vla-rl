

OUTPUT_DIR="work_dirs/sft-7b-lora128-5k-64tokens"

TAG="855steps_vote5_pass3-64tokens"
python utils/eval_reward_model.py --output_dir $OUTPUT_DIR --tag $TAG