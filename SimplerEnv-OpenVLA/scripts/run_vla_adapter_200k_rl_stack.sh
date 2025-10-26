model_name=vla-adapter
tasks=(
    # bridge_carrot.sh
    bridge_stack.sh
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
#     /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+bridge_orig+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--brdige----200000_chkpt # or a local path
# )
# tag=bridge_ck5_200k_rl_carrot_9steps
ckpt_path=/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+bridge_orig+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--brdige----200000_chkpt
tags=(
    bridge_ck5_200k_rl_stack_filter_29steps_repeat1
    bridge_ck5_200k_rl_stack_filter_29steps_repeat2
    bridge_ck5_200k_rl_stack_filter_29steps_repeat3
    bridge_ck5_200k_rl_stack_filter_29steps_repeat4
    bridge_ck5_200k_rl_stack_filter_29steps_repeat5
)

load_ckpt_path="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/stack-filter/step29.pt"

action_ensemble_temp=0.0
for tag in ${tags[@]}; do
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
        device=2
        bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device $load_ckpt_path
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
