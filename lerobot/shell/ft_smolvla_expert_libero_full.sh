export CUDA_VISIBLE_DEVICES=0
cd src/lerobot 
python scripts/train.py \
  --policy.path=/inspire/ssd/project/robotsimulation/public/huggingface_models/smolvla_base \
  --dataset.repo_id=/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot/libero_full_lerobot \
  --batch_size=64 \
  --steps=7000 \
  --output_dir=outputs/train/debug \
  --job_name=debug \
  --log_freq=1 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false