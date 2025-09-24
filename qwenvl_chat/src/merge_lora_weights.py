import argparse
from utils import get_model_name_from_path, load_pretrained_model

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map='cpu')
    model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/qwenvl_chat/work_dirs/qwen2.5-vl-7b-sft-lora-baseline-5k-r128/checkpoint-171")
    parser.add_argument("--model-base", type=str, default="/inspire/ssd/project/robotsimulation/public/huggingface_models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--save-model-path", type=str, default="debug")
    parser.add_argument("--safe-serialization", action='store_true')

    args = parser.parse_args()

    merge_lora(args)