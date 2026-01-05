"""
探索 BRIDGE 数据集的 tfrecord 格式
"""

import argparse
from pathlib import Path
import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter")
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform, EpisodicRLDSDataset

from rapidfuzz import fuzz
import re

def normalize(s):
    if isinstance(s, bytes):
        s = s.decode(errors='ignore')
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def keyword_fuzzy_match_exact(text, keywords, task):
    text_n = normalize(text)
    if task == 'StackGreenCubeOnYellowCubeBakedTexInScene':
        for kw in keywords:
            if kw in text_n:
                return True, 100.0, kw
        return False, 0.0, kw
    else:
        for kw in keywords:
            if kw not in text_n:
                return False, 0.0, kw
        return True, 100.0, kw

candidates_exact = {
  "PutCarrotOnPlateInScene": ["carrot", "plate", "on"],
  "PutEggplantInBasketScene": ["eggplant"],
  "PutSpoonOnTableClothInScene": ["spoon", "cloth"],
  "StackGreenCubeOnYellowCubeBakedTexInScene": ["stack", "cube", "block"]
}


def explore_tfrecord_structure(rlds_batch, ep_idx):
    """探索单个 episode 的结构"""
    print(f"\n{'='*60}")
    print(f"Episode {ep_idx}: Keys and Shapes")
    print(f"{'='*60}")

    # 递归打印结构
    def print_structure(d, prefix="", depth=0):
        if depth > 3:
            return
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"{prefix}{k}:")
                if isinstance(v, dict):
                    print_structure(v, prefix + "  ", depth + 1)
                elif isinstance(v, np.ndarray):
                    print(f"{prefix}  ndarray, shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, bytes):
                    try:
                        decoded = v.decode(errors='ignore')
                        print(f"{prefix}  bytes: {decoded[:100]}...")
                    except:
                        print(f"{prefix}  bytes (length={len(v)})")
                else:
                    print(f"{prefix}  {type(v).__name__}: {v}")
        elif isinstance(d, np.ndarray):
            print(f"{prefix}ndarray, shape={d.shape}, dtype={d.dtype}")

    print_structure(rlds_batch)

    # 详细打印每个顶层 key
    print(f"\n{'='*60}")
    print("Detailed Keys:")
    print(f"{'='*60}")

    for key in rlds_batch.keys():
        value = rlds_batch[key]
        print(f"\n[{key}]")
        if isinstance(value, dict):
            for sub_key in value.keys():
                sub_value = value[sub_key]
                print(f"  .{sub_key}")
                if isinstance(sub_value, np.ndarray):
                    print(f"    shape: {sub_value.shape}, dtype: {sub_value.dtype}")
                    if sub_value.ndim == 0:
                        print(f"    value: {sub_value.item()}")
                    elif sub_value.size < 10:
                        print(f"    value: {sub_value}")
                else:
                    print(f"    type: {type(sub_value)}")


def main():
    parser = argparse.ArgumentParser(description="Explore BRIDGE tfrecord format")
    parser.add_argument("--dataset_dirs", type=Path,
                        default='/inspire/ssd/project/robotsimulation/public/data/bridge/bridge_select_trajs/bridge',
                        help="the dataset root")
    parser.add_argument("--data_mix", type=str, default='bridge_orig')
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="number of episodes to explore")
    args = parser.parse_args()

    dataset = EpisodicRLDSDataset(
        str(args.dataset_dirs),
        args.data_mix,
        batch_transform=None,
        
        resize_resolution=(256,256),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=False,
    )

    dataset_length = dataset.dataset_length
    print(f"Total dataset length: {dataset_length}")

    ep_idx = 0
    for rlds_batch in dataset.dataset.as_numpy_iterator():
        if ep_idx >= args.num_episodes:
            break

        explore_tfrecord_structure(rlds_batch, ep_idx)
        ep_idx += 1


if __name__ == "__main__":
    main()
