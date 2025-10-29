
declare -a experiments=(
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_9" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step9.pt"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_19" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step19.pt"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_29" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step29.pt"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_39" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step39.pt"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_49" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step49.pt"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/SimpleVLA-RL/exp2-vla_adapter_wm_kl-full-fixedbug-faster-bridge-eggplant/actor/global_step_59" "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts/bridge/eggplant-fa/step59.pt"
)

num_experiments=${#experiments[@]}

for (( i=0; i<${num_experiments}; i+=2 )); do
    model_path=${experiments[i]}
    merge_path=${experiments[i+1]}

    bash examples/exp/VLA-Adapter_exp/merge_ckpts/merge_bridge_batch.sh "${model_path}" "${merge_path}"
    ray stop --force
    echo "############################################################"
    echo "##                                                        ##"
    echo "##  实验 ${merge_path} 已完成"
    echo "##                                                        ##"
    echo "############################################################"
    echo ""
    echo ""
done

echo "所有实验已完成！"