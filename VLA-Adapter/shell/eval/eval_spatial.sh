CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio False \
  --num_images_in_input 1 \
  --use_film False \
  --pretrained_checkpoint /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+libero_spatial_no_noops+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops----140000_chkpt \
  --task_suite_name libero_spatial \
  --use_pro_version True \
#   > eval_logs/Spatial--chkpt.log 2>&1 &