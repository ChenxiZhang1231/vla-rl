export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset/libero_10_no_noops /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset/libero_90_no_noops /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset/libero_spatial_no_noops /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset/libero_goal_no_noops /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset/libero_object_no_noops  \
    --output-path /SSD_DISK/users/zhangjiahui/LIBERO/libero/dataset \
    --executor local \
    --tasks-per-job 3 \
    --workers 20
