

OUTPUT_DIR="work_dirs/sft-7b-lora128-5k-ref"

TAG="855steps_vote5_pass3"
python utils/eval_reward_model_ref.py --output_dir $OUTPUT_DIR --tag $TAG