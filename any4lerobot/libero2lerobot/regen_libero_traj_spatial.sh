# export MUJOCO_GL=osmesa
python libero_utils/regenerate_libero_dataset.py \
    --resolution 256 \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /inspire/ssd/project/robotsimulation/public/data/LIBERO-datasets/libero_spatial \
    --libero_target_dir /inspire/ssd/project/robotsimulation/public/data/LIBERO-datasets/libero_spatial_no_noops