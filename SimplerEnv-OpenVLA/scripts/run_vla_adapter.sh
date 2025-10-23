model_name=vla-adapter
tasks=(
    bridge.sh
    # drawer_variant_agg.sh
    # drawer_visual_matching.sh
    # move_near_variant_agg.sh
    # move_near_visual_matching.sh
    # pick_coke_can_variant_agg.sh
    # pick_coke_can_visual_matching.sh

    # put_in_drawer_variant_agg.sh
    # put_in_drawer_visual_matching.sh
)

# ckpts=(
#     /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+bridge_orig+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--brdige----400000_chkpt # or a local path
# )
# tag=bridge_ck5_400k

CKPT_PATH="${1:-}"
TAG="${2:-}"
DEVICE="${3:-0}"


if [[ -z "$CKPT_PATH" || -z "$TAG" ]]; then
  echo "Usage: $0 <ckpt_path> <tag> <device_id> [action_ensemble_temp]"
  exit 1
fi

ckpts=(
    "$CKPT_PATH"
)

logging_dir="results/$(basename "$TAG")"
mkdir -p "$logging_dir"


action_ensemble_temp=0.0
for ckpt_path in ${ckpts[@]}; do
    echo "==> Running ckpt: $ckpt_path"
    echo "==> Tag         : $TAG"
    echo "==> Device      : $DEVICE"

    mkdir -p $logging_dir
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        echo "🚀 running $task ..."
        device=$DEVICE"
        bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
