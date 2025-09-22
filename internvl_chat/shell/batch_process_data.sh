#!/bin/bash

# 定义实验的视频文件夹和输出目录
declare -a experiments=(
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train2" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train2"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train3" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train3"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train4" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train4"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train5" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train5"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train6" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train6"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train7" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train7"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train8" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train8"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train9" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train9"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train10" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train10"
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train11" "/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train11"
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
        python tools/process_rollout_video_2_jsonl.py "--video_folder=${video_folder}" "--output_dir=${output_dir}"

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