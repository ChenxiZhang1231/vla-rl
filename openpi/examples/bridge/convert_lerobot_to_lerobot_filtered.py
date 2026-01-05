#!/usr/bin/env python3
"""
从 TFDS 格式的 BRIDGE 原始数据集中筛选多个指定任务的 episodes，
并将结果分别保存为新的 LeRobot 数据集（使用 LeRobot API，图片格式）。

支持四个任务:
    - PutCarrotOnPlateInScene
    - PutEggplantInBasketScene
    - PutSpoonOnTableClothInScene
    - StackGreenCubeOnYellowCubeBakedTexInScene

用法:
    python convert_lerobot_to_lerobot_filtered.py
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds
import re
from typing import List, Tuple, Dict, Any
import pyarrow.parquet as pq


def fix_list_to_sequence(task_dir: Path):
    """
    修改数据集 parquet 文件中的 List 类型为 Sequence 类型
    使其兼容 datasets 3.x
    """
    data_dir = task_dir / "data"
    if not data_dir.exists():
        return

    print(f"    修改 parquet metadata: List -> Sequence")

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            # 读取 parquet 文件
            pf = pq.ParquetFile(parquet_file)
            table = pf.read()

            # 获取原始 metadata
            metadata = table.schema.metadata
            if b"huggingface" in metadata:
                hf_meta = metadata[b"huggingface"].decode()
                # 把 List 改成 Sequence
                hf_meta_fixed = hf_meta.replace('"_type": "List"', '"_type": "Sequence"')
                metadata[b"huggingface"] = hf_meta_fixed.encode()

                # 创建新的 schema 并重新写入
                new_schema = table.schema.with_metadata(metadata)
                new_table = table.cast(new_schema)

                # 写入覆盖原文件
                pq.write_table(new_table, parquet_file, compression="snappy")


def normalize(s):
    """标准化字符串用于匹配"""
    if isinstance(s, bytes):
        s = s.decode(errors='ignore')
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def keyword_fuzzy_match_exact(text, keywords, task):
    """
    精确关键词匹配
    对于大多数任务: 需要同时包含所有关键词
    对于 StackGreenCubeOnYellowCubeBakedTexInScene: 只需包含任意一个关键词
    """
    text_n = normalize(text)
    if task == 'StackGreenCubeOnYellowCubeBakedTexInScene':
        # 只需包含任意一个关键词
        for kw in keywords:
            if kw in text_n:
                return True, 100.0, kw
        return False, 0.0, None
    else:
        # 需要同时包含所有关键词
        for kw in keywords:
            if kw not in text_n:
                return False, 0.0, kw
        return True, 100.0, keywords[0]


# 四个任务的关键词配置（使用 collect_trainset_rm.py 的配置）
candidates_exact = {
    "PutCarrotOnPlateInScene": ["carrot", "plate", "on"],
    "PutEggplantInBasketScene": ["eggplant"],
    "PutSpoonOnTableClothInScene": ["spoon", "cloth"],
    "StackGreenCubeOnYellowCubeBakedTexInScene": ["stack", "cube", "block"]
}


def main():
    parser = argparse.ArgumentParser(description='从 BRIDGE 原始数据集筛选多个任务的 episodes')
    parser.add_argument('--data_dir', type=str,
                        default='/inspire/ssd/project/robotsimulation/public/data/bridge',
                        help='TFDS 数据集根目录')
    parser.add_argument('--output_root', type=str,
                        default='/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/bridge_4tasks',
                        help='输出数据集根目录（四个任务将分别保存到此目录下的子文件夹）')
    parser.add_argument('--image_writer_threads', type=int, default=10,
                        help='图片写入线程数')
    parser.add_argument('--image_writer_processes', type=int, default=5,
                        help='图片写入进程数')
    parser.add_argument('--raw_dataset_name', type=str, default='bridge_orig',
                        help='原始 TFDS 数据集名称')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=list(candidates_exact.keys()),
                        help='要筛选的任务列表 (默认: 所有四个任务)')
    parser.add_argument('--max_episodes', type=int, default=2000,
                        help='最多扫描的 episode 数量 (默认: 2000)')

    args = parser.parse_args()

    output_root = Path(args.output_root)

    print(f"TFDS 数据目录: {args.data_dir}")
    print(f"输出根目录: {output_root}")
    print(f"筛选任务: {args.tasks}")

    # 加载原始 TFDS 数据集
    print(f"\n[1/4] 加载原始 TFDS 数据集 '{args.raw_dataset_name}'...")
    raw_dataset = tfds.load(args.raw_dataset_name, data_dir=args.data_dir, split="train")

    # 筛选 episodes
    print(f"\n[2/4] 筛选 episodes...")
    for task in args.tasks:
        print(f"  - {task}: {candidates_exact.get(task, [])}")

    # 为每个任务创建一个列表存储筛选结果
    filtered_episodes = {task: [] for task in args.tasks}
    total_count = 0
    task_counts = {task: 0 for task in args.tasks}

    # 遍历所有 episodes
    for episode in raw_dataset:
        total_count += 1

        # # 达到最大扫描数量后停止
        # if total_count > args.max_episodes:
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

        # 检查每个任务的匹配条件
        matched_tasks = []
        for task in args.tasks:
            keywords = candidates_exact.get(task, [])
            matched, score, kw = keyword_fuzzy_match_exact(lang, keywords, task)
            if matched:
                matched_tasks.append(task)

        # 如果匹配到任务，保存到对应的任务列表
        if matched_tasks:
            # 重新创建迭代器并保存所有 steps
            steps_list = [first_step] + list(steps_iter)

            for task in matched_tasks:
                filtered_episodes[task].append((steps_list, lang))
                task_counts[task] += 1

        if total_count % 100 == 0:
            print(f"  已扫描 {total_count} 个 episodes, " +
                  ", ".join([f"{task}: {task_counts[task]}" for task in args.tasks]))

    print(f"\n  总共扫描 {total_count} 个 episodes")
    print(f"  筛选结果:")
    for task in args.tasks:
        print(f"    - {task}: {len(filtered_episodes[task])} 个 episodes")

    # 检查是否有匹配结果
    empty_tasks = [task for task in args.tasks if not filtered_episodes[task]]
    if empty_tasks:
        print(f"\n  警告: 以下任务未找到匹配的 episodes: {empty_tasks}")

    if all(len(filtered_episodes[task]) == 0 for task in args.tasks):
        print(f"\n  未找到任何匹配的任务！")
        return

    # 导入 LeRobot
    import sys
    sys.path.insert(0, "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/lerobot/src")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 为每个任务创建独立的数据集
    print(f"\n[3/4] 创建 LeRobot 数据集...")

    for task in args.tasks:
        episodes = filtered_episodes[task]
        if not episodes:
            continue

        # 每个任务使用独立的临时父目录，避免 LeRobotDataset.create 的 root 目录冲突
        task_root = output_root / f"temp_{task}"
        if task_root.exists():
            shutil.rmtree(task_root)

        print(f"\n  处理任务: {task}")
        print(f"    Episodes: {len(episodes)}")
        print(f"    输出目录: {output_root / task}")

        # 创建 LeRobot 数据集
        # 使用与 config.py 中 LeRobotBridgeDataConfig 兼容的 key 格式
        dataset = LeRobotDataset.create(
            repo_id=task,
            root=task_root,
            fps=10,
            features={
                "observation.images.image_0": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.image_1": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.image_2": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.image_3": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
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
        print(f"    保存 episodes...")

        for new_idx, (steps_list, task_name) in enumerate(episodes):
            # 处理每个 step
            for step in steps_list:
                obs = step["observation"]
                dataset.add_frame(
                    {
                        "observation.images.image_0": obs["image_0"],
                        "observation.images.image_1": obs["image_1"],
                        "observation.images.image_2": obs["image_2"],
                        "observation.images.image_3": obs["image_3"],
                        "observation.state": obs["state"].astype(np.float32),
                        "action": step["action"].astype(np.float32),
                    },
                    task=task_name,
                )

            dataset.save_episode()

            # 清理 hf_dataset 防止 OOM
            dataset.hf_dataset = dataset.create_hf_dataset()

            if (new_idx + 1) % 10 == 0:
                print(f"      已处理 {new_idx + 1}/{len(episodes)} 个 episodes")

        print(f"    完成: {len(episodes)} 个 episodes")

        # 修改 parquet metadata: List -> Sequence (兼容 datasets 3.x)
        # 数据集实际保存在 task_root 下
        fix_list_to_sequence(task_root)

        # 将数据集移动到最终位置
        final_dir = output_root / task
        if final_dir.exists():
            shutil.rmtree(final_dir)
        # output_root 需要先存在
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.move(task_root, final_dir)

    print("\n" + "="*50)
    print("完成！")
    print(f"输出根目录: {output_root}")
    for task in args.tasks:
        count = len(filtered_episodes[task])
        if count > 0:
            print(f"  - {task}: {count} episodes -> {output_root / task}")
    print("="*50)


if __name__ == "__main__":
    main()
