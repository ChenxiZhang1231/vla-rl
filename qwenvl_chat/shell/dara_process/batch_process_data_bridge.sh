#!/bin/bash


declare -a experiments=(
    "/inspire/ssd/project/robotsimulation/public/data/rm_bridge_4tasks/predict2_video2world_2b_action_conditioned_finetuning_generation_bridge_2025-10-22_10-10-55/videos/00000/" "/inspire/ssd/project/robotsimulation/public/data/rm_for_wm_train_jsonl_bridge_4tasks/rm1"
    "/inspire/ssd/project/robotsimulation/public/data/rm_bridge_4tasks/predict2_video2world_2b_action_conditioned_finetuning_generation_bridge_2025-10-22_11-25-33/videos/00000/" "/inspire/ssd/project/robotsimulation/public/data/rm_for_wm_train_jsonl_bridge_4tasks/rm2"
    "/inspire/ssd/project/robotsimulation/public/data/rm_bridge_4tasks/predict2_video2world_2b_action_conditioned_finetuning_generation_bridge_2025-10-22_13-47-45/videos/00000/" "/inspire/ssd/project/robotsimulation/public/data/rm_for_wm_train_jsonl_bridge_4tasks/rm3"
    "/inspire/ssd/project/robotsimulation/public/data/rm_bridge_4tasks/predict2_video2world_2b_action_conditioned_finetuning_generation_bridge_2025-10-22_15-00-31/videos/00000/" "/inspire/ssd/project/robotsimulation/public/data/rm_for_wm_train_jsonl_bridge_4tasks/rm4"
)

num_experiments=${#experiments[@]}

# 循环启动所有实验，并将它们放入后台执行
for (( i=0; i<${num_experiments}; i+=2 )); do
    # 将循环体放在 (...) & 中，使其在子shell中后台运行
    (
        video_folder=${experiments[i]}
        output_dir=${experiments[i+1]}

        echo "############################################################"
        echo "##                                                        ##"
        echo "##  正在开始实验: ${video_folder}"
        echo "##  加载模型路径: ${output_dir}"
        echo "##                                                        ##"
        echo "############################################################"

        # 运行Python脚本
        python utils/process_rollout_video_2_jsonl_bridge.py "--video_folder=${video_folder}" "--output_dir=${output_dir}"

        echo "############################################################"
        echo "##                                                        ##"
        echo "##  实验 ${output_dir} 已完成"
        echo "##                                                        ##"
        echo "############################################################"
        echo ""
        echo ""
    ) &
done

# 等待所有后台启动的子进程（实验）完成
wait

echo "所有实验已完成！"