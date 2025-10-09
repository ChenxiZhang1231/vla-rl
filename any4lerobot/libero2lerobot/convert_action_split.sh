export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths /inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_spatial_no_noops  \
    --output-path /inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Spatial \
    --executor local \
    --tasks-per-job 5 \
    --workers 100 \
    --use_delta_action


python libero_h5.py \
    --src-paths /inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_10_no_noops  \
    --output-path /inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Long \
    --executor local \
    --tasks-per-job 5 \
    --workers 100 \
    --use_delta_action


python libero_h5.py \
    --src-paths /inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_goal_no_noops \
    --output-path /inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Goal \
    --executor local \
    --tasks-per-job 5 \
    --workers 100 \
    --use_delta_action

python libero_h5.py \
    --src-paths /inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed/libero_object_no_noops  \
    --output-path /inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Object \
    --executor local \
    --tasks-per-job 5 \
    --workers 100 \
    --use_delta_action
