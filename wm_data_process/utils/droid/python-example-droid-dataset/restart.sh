#!/bin/bash

# 设定运行时限为 1 小时（3600 秒）
TIME_LIMIT=3600

# 目标脚本路径
SCRIPT_PATH="./process_droid_data_multi_process_total.sh"

# 无限循环以定时重启
while true; do
  echo "Starting process_droid_data_multi_process_total.sh with a time limit of $TIME_LIMIT seconds..."

  # 使用 timeout 限制运行时间
  timeout $TIME_LIMIT bash "$SCRIPT_PATH"
  
  # 检查退出状态，如果失败则显示错误信息
  if [ $? -ne 0 ]; then
    echo "process_droid_data_multi_process_total.sh was interrupted or failed."
  fi
  
  echo "Waiting 10 seconds before restarting..."
  sleep 10  # 停止后等待 10 秒
  
  echo "Restarting process_droid_data_multi_process_total.sh..."
done
