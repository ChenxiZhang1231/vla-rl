# uv run scripts/compute_norm_stats.py --config-name pi05_bridge

python examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_bridge \
    --checkpoint_dir /inspire/ssd/project/robotsimulation/public/huggingface_models/pi05_base \
    --output_path /inspire/ssd/project/robotsimulation/public/huggingface_models/pi05_base_torch