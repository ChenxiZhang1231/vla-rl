# lrm_wm_vla_adapter
data_name=bridge_orig

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --vla_path /inspire/ssd/project/robotsimulation/public/data/openvla-7b \
    --data_root_dir /inspire/ssd/project/robotsimulation/public/data/bridge \
    --dataset_name $data_name \
    --run_root_dir outputs \
    --use_l1_regression False \
    --use_diffusion False \
    --use_flow True \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --use_lora True \
    --image_aug True \
    --num_steps_before_decay 400000 \
    --max_steps 400005 \
    --save_freq 20000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 8 \
    --image_aug True \
    --grad_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --run_id_note OpenVLA-OFT--brdige-repeat--$current_time \
    # > logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &