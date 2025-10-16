#!/bin/bash

declare -a experiments=(
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_9" "wm_spatial_1" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_1" "libero_spatial"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_19" "wm_spatial_2" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_2" "libero_spatial"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_9" "wm_spatial_3" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Spatial_3" "libero_spatial"
    # "" "wm_goal_1" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_1" "libero_goal"
    # "" "wm_goal_2" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_2" "libero_goal"
    # "" "wm_goal_3" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Goal_3" "libero_goal"
    # "" "wm_object_1" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_1" "libero_object"
    # "" "wm_object_2" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_2" "libero_object"
    # "" "wm_object_3" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Object_3" "libero_object"
    "" "wm_long_1" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_1" "libero_10"
    "" "wm_long_2" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_2" "libero_10"
    "" "wm_long_3" "/inspire/ssd/project/robotsimulation/public/data/LIBERO-Processed-Additional/Long_3" "libero_10"
)


num_experiments=${#experiments[@]}

for (( i=0; i<${num_experiments}; i+=4 )); do
    load_model_path=${experiments[i]}
    experiment_name=${experiments[i+1]}
    wm_save_path=${experiments[i+2]}
    data_name=${experiments[i+3]}

    echo "############################################################"
    echo "##                                                        ##"
    echo "##  正在开始实验: ${experiment_name}"
    echo "##  加载模型路径: ${load_model_path}"
    echo "##                                                        ##"
    echo "############################################################"

    bash examples/debug/rollout_rm_dataset_smolvlarl.sh "${load_model_path}" "${experiment_name}" "${wm_save_path}" "${data_name}"

    echo "############################################################"
    echo "##                                                        ##"
    echo "##  实验 ${experiment_name} 已完成"
    echo "##                                                        ##"
    echo "############################################################"
    echo ""
    echo ""
done

echo "所有实验已完成！"