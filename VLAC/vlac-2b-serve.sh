CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model /inspire/ssd/project/robotsimulation/public/huggingface_models/VLAC \
    --host 0.0.0.0 \
    --port 8000 \
    --infer_backend pt \
    --served_model_name judge