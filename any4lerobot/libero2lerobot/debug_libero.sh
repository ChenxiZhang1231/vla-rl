export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=osmesa && export PYOPENGL_PLATFORM=osmesa
python libero_utils/regenerate_libero_dataset.py \
    --resolution 256 \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /inspire/ssd/project/robotsimulation/public/data/LIBERO-datasets/libero_spatial \
    --libero_target_dir debug