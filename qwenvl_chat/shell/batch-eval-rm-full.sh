

OUTPUT_DIR="work_dirs/sft-7b-full-5k-64tokens"

TAG="1710steps_vote5_pass3-64tokens"
python utils/eval_reward_model.py --output_dir $OUTPUT_DIR --tag $TAG