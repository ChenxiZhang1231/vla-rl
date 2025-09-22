CUDA_VISIBLE_DEVICES=6 python src/process_raw_data_multi_thread.py \
  --scene data/1.0.1/TRI/6 \
  --save_path data/droid_processed_data/TRI/ \
  --urdf franka_description/panda.urdf \
  --num_threads 20