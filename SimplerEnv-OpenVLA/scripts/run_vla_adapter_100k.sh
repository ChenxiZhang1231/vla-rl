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

ckpts=(
    /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+bridge_orig+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--brdige----100000_chkpt # or a local path
)
tag=bridge_ck5_100k

action_ensemble_temp=0.0
for ckpt_path in ${ckpts[@]}; do
    # 🤗 NOTE: set hf cache to avoid confilcts
    # base_dir=$(dirname $ckpt_path)
    # export HF_MODULES_CACHE=$base_dir/hf_cache/modules
    # mkdir -p $HF_MODULES_CACHE
    # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  
    logging_dir=results/$(basename $tag)

    mkdir -p $logging_dir
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        echo "🚀 running $task ..."
        device=1
        bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
