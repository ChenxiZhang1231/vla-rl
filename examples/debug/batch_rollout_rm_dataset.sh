#!/bin/bash

# declare -a experiments=(
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_9" "rm_train2"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_19" "rm_train3"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_29" "rm_train4"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_39" "rm_train5"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_49" "rm_train6"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_59" "rm_train7"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_69" "rm_train8"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_79" "rm_train9"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_89" "rm_train10"
#     "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/actor/global_step_99" "rm_train11"
# )

declare -a experiments=(
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/actor/global_step_19" "rm_train12"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/actor/global_step_29" "rm_train13"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/actor/global_step_39" "rm_train14"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/actor/global_step_79" "rm_train15"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/actor/global_step_109" "rm_train16"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-10-sii/actor/global_step_169" "rm_train17"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/actor/global_step_19" "rm_train18"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-14-whiten/actor/global_step_29" "rm_train19"
    # "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-15-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-15-whiten/actor/global_step_9" "rm_train20"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-15-whiten/SimpleVLA-RL/smolvla-bs32-n8-mb256-lr5e6-kl004-rm7b_5k-trainset-15-whiten/actor/global_step_19" "rm_train21"
)

num_experiments=${#experiments[@]}

for (( i=0; i<${num_experiments}; i+=2 )); do
    load_model_path=${experiments[i]}
    experiment_name=${experiments[i+1]}

    echo "############################################################"
    echo "##                                                        ##"
    echo "##  正在开始实验: ${experiment_name}"
    echo "##  加载模型路径: ${load_model_path}"
    echo "##                                                        ##"
    echo "############################################################"

    bash examples/debug/rollout_rm_dataset_smolvlarl.sh "${load_model_path}" "${experiment_name}"

    echo "############################################################"
    echo "##                                                        ##"
    echo "##  实验 ${experiment_name} 已完成"
    echo "##                                                        ##"
    echo "############################################################"
    echo ""
    echo ""
done

echo "所有实验已完成！"