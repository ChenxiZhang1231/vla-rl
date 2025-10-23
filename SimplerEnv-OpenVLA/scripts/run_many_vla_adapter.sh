#!/usr/bin/env bash
set -euo pipefail

RUNNER="scripts/run_vla_adapter.sh"   # 指向上面改造后的脚本
# DEVICE=0                        # 全局 GPU 号

declare -a experiments=(
    "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter/outputs/configs+bridge_orig+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--VLA-Adapter--brdige----200000_chkpt" "bridge_ck5_200k" "0"
)

num_experiments=${#experiments[@]}
if (( num_experiments % 3 != 0 )); then
  echo "experiments 数组必须成对：<ckpt> \"<tag>\""
  exit 2
fi

for (( i=0; i<${num_experiments}; i+=3 )); do
    ckpt_path=${experiments[i]}
    tag=${experiments[i+1]}
    device=${experiments[i+2]}

    echo "############################################################"
    echo "##  实验: ${tag}"
    echo "##  模型: ${ckpt_path}"
    echo "##  GPU : ${device}"
    echo "############################################################"

    bash "$RUNNER" "$ckpt_path" "$tag" "$device"

    echo "########## 实验 ${tag} 已完成 ##########"
    echo
done

echo "🎉 所有实验已完成！"
