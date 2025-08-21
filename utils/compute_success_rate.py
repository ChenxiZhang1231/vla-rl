import os
import re
from collections import defaultdict

def calculate_success_rates(directory_path):

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误：找不到目录 '{directory_path}'")
        return


    task_totals = defaultdict(int)
    task_successes = defaultdict(int)

    filename_pattern = re.compile(r"task=(.*?)_trial_\d+--success=(True|False)")

    for filename in os.listdir(directory_path):
        match = filename_pattern.search(filename)
        
        if match:
            task_name = match.group(1)  # 提取任务名称
            is_success = match.group(2) == 'True'  # 提取成功状态

            task_totals[task_name] += 1
            
            if is_success:
                task_successes[task_name] += 1

    if not task_totals:
        print(f"在目录 '{directory_path}' 中没有找到符合命名规则的文件。")
        return

    print("各任务成功率计算结果：")
    print("-" * 40)
    total_success_runs = 0
    for task_name in sorted(task_totals.keys()):
        total_runs = task_totals[task_name]
        success_runs = task_successes[task_name]
        
        success_rate = (success_runs / total_runs) * 100 if total_runs > 0 else 0
        total_success_runs += success_runs
        print(f"Task: {task_name}")
        print(f"   Success rate: {success_rate:.2f}% ({success_runs} / {total_runs})")
        print("-" * 40)
    print(f"Total Task:")
    print(f"  Success rate: {total_success_runs/500:.2f}% ({total_success_runs} / {500})")
    print("-" * 40)


target_directory = '/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/jasonzhang/vla-rl/rollouts/infer-smolvla_sft_full_eval_spatial'

calculate_success_rates(target_directory)