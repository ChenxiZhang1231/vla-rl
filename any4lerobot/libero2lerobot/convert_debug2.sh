export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths  /inspire/ssd/project/robotsimulation/public/data/LIBERO-datasets/libero_spatial_no_noops \
    --output-path /inspire/ssd/project/robotsimulation/public/data/Debug-dataset2 \
    --executor local \
    --tasks-per-job 3 \
    --workers 20
