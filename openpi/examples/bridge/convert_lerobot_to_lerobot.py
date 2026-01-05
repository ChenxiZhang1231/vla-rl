#!/usr/bin/env python3
"""
从 TFDS 格式的 BRIDGE 原始数据集中筛选指定任务的 episodes，
并将结果保存为新的 LeRobot 数据集（使用 LeRobot API，图片格式）。

用法:
    python convert_lerobot_to_lerobot.py --task "put carrot on plate"
    python convert_lerobot_to_lerobot.py --task "PutCarrotOnPlateInScene"
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds


def main():
    parser = argparse.ArgumentParser(description='从 BRIDGE 原始数据集筛选指定任务的 episodes')
    parser.add_argument('--task', type=str, default='PutCarrotOnPlateInScene',
                        help='要筛选的任务名称 (默认: "PutCarrotOnPlateInScene")')
    parser.add_argument('--data_dir', type=str,
                        default='/inspire/ssd/project/robotsimulation/public/data/bridge',
                        help='TFDS 数据集根目录')
    parser.add_argument('--output', type=str,
                        default='/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/openpi1/bridge_filtered',
                        help='输出数据集路径')
    parser.add_argument('--image_writer_threads', type=int, default=10,
                        help='图片写入线程数')
    parser.add_argument('--image_writer_processes', type=int, default=5,
                        help='图片写入进程数')
    parser.add_argument('--raw_dataset_name', type=str, default='bridge_orig',
                        help='原始 TFDS 数据集名称')

    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"TFDS 数据目录: {args.data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"筛选任务: {args.task}")

    # 关键词配置：任务名必须同时包含的所有关键词
    keywords_candidates = {
        "PutCarrotOnPlateInScene": ["carrot", "plate", "on"],
    }

    # 加载原始 TFDS 数据集
    print(f"\n[1/4] 加载原始 TFDS 数据集 '{args.raw_dataset_name}'...")
    raw_dataset = tfds.load(args.raw_dataset_name, data_dir=args.data_dir, split="train")

    # 筛选 episodes
    print(f"\n[2/4] 筛选任务 '{args.task}' 的 episodes...")
    print(f"  关键词: {keywords_candidates.get(args.task, [args.task.lower()])}")

    # 先遍历一遍筛选符合条件的 episode（保存引用和任务描述）
    filtered_episodes = []
    total_count = 0

    keywords = keywords_candidates.get(args.task, [args.task.lower()])

    # max_episodes = 1100  # 最多扫描的 episode 数量

    for episode in raw_dataset:
        total_count += 1

        # # 达到最大扫描数量后停止
        # if total_count > max_episodes:
        #     break
        steps_iter = episode["steps"].as_numpy_iterator()

        try:
            first_step = next(steps_iter)
        except StopIteration:
            continue

        # 获取任务描述
        lang = first_step.get("language_instruction")
        if lang is None:
            continue

        if isinstance(lang, (bytes, bytearray)):
            lang = lang.decode()

        lang_lower = lang.lower()

        # 检查是否同时包含所有关键词
        if all(kw.lower() in lang_lower for kw in keywords):
            # 重新创建迭代器并保存第一个 step
            steps_list = [first_step] + list(steps_iter)
            filtered_episodes.append((steps_list, lang))

        if total_count % 100 == 0:
            print(f"  已扫描 {total_count} 个 episodes, 找到 {len(filtered_episodes)} 个匹配")

    print(f"  总共扫描 {total_count} 个 episodes")
    print(f"  找到 {len(filtered_episodes)} 个匹配的 episodes")

    if not filtered_episodes:
        print(f"  未找到匹配 '{args.task}' 的任务！")
        return

    # 清理输出目录
    if output_dir.exists():
        print(f"\n[3/4] 清理输出目录...")
        shutil.rmtree(output_dir)

    # 创建 LeRobot 数据集
    print(f"\n[3/4] 创建 LeRobot 数据集...")
    import sys
    sys.path.insert(0, "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/lerobot/src")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # BRIDGE 数据集有 4 个视角
    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir.parent,
        robot_type="widowx",
        fps=10,
        features={
            "image_0": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "image_1": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "image_2": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "image_3": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    # 处理每个 episode
    print(f"\n[4/4] 处理 episodes...")

    for new_idx, (steps_list, task_name) in enumerate(filtered_episodes):
        # 处理每个 step
        for step in steps_list:
            obs = step["observation"]
            dataset.add_frame(
                {
                    "image_0": obs["image_0"],
                    "image_1": obs["image_1"],
                    "image_2": obs["image_2"],
                    "image_3": obs["image_3"],
                    "state": obs["state"].astype(np.float32),
                    "action": step["action"].astype(np.float32),
                },
                task=task_name,
            )

        dataset.save_episode()

        # 清理 hf_dataset 防止 OOM
        dataset.hf_dataset = dataset.create_hf_dataset()

        if (new_idx + 1) % 10 == 0:
            print(f"  已处理 {new_idx + 1}/{len(filtered_episodes)} 个 episodes")

    print(f"  总共处理 {len(filtered_episodes)} 个 episodes")

    print("\n" + "="*50)
    print("完成！")
    print(f"输出目录: {output_dir}")
    print(f"  - Episodes: {len(filtered_episodes)}")
    print("="*50)


if __name__ == "__main__":
    main()
