# lrm_cu126_main
data_name=bridge_orig

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
--vlm_path /inspire/ssd/project/robotsimulation/public/data/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir /inspire/ssd/project/robotsimulation/public/data/bridge \
--dataset_name $data_name \
--run_root_dir outputs_ck10 \
--use_film False \
--num_images_in_input 1 \
--use_proprio False \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 400000 \
--max_steps 400005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--use_flow True \
--grad_accumulation_steps 1 \
--learning_rate 1e-4 \
--lora_rank 64 \
--use_pro_version True \
--run_id_note VLA-Adapter--brdige--$current_time \
# > logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &