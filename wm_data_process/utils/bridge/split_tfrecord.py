"""
Split BRIDGE dataset by task type and save as RLDS-format tfrecords.
Each task gets its own subdirectory with proper RLDS structure.
"""

import argparse
from pathlib import Path
import os
import json
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
import hashlib

import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter")
from prismatic.vla.datasets import EpisodicRLDSDataset

from rapidfuzz import fuzz
import re


# RLDS version
VERSION = "1.0.0"
DATASET_NAME = "bridge_dataset"  # Name in dataset_info.json, matches original


def normalize(s):
    if isinstance(s, bytes):
        s = s.decode(errors='ignore')
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def keyword_fuzzy_match(text, keywords, fuzz_threshold=70):
    text_n = normalize(text)
    for kw in keywords:
        if kw in text_n:
            return True, 100.0, kw
        score = fuzz.token_sort_ratio(text_n, kw)
        if score >= fuzz_threshold:
            return True, float(score), kw
    return False, 0.0, None


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


candidates = {
    "PutCarrotOnPlateInScene": ["carrot", "plate", "dish", "put carrot", "on plate", "place carrot"],
    "PutEggplantInBasketScene": ["eggplant", "aubergine", "basket", "put eggplant", "in basket", "move eggplant"],
    "PutSpoonOnTableClothInScene": ["spoon", "towel", "tablecloth", "napkin", "cloth", "place spoon", "on towel", "on cloth"],
    "StackGreenCubeOnYellowCubeBakedTexInScene": ["stack", "cube", "green cube", "yellow cube", "put green on yellow", "stack green"]
}

candidates_exact = {
    "PutCarrotOnPlateInScene": ["carrot", "plate", "on"],
    "PutEggplantInBasketScene": ["eggplant"],
    "PutSpoonOnTableClothInScene": ["spoon", "cloth"],
    "StackGreenCubeOnYellowCubeBakedTexInScene": ["stack", "cube", "block"]
}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, (list, np.ndarray)):
        value = np.array(value).tobytes()
    elif isinstance(value, str):
        value = value.encode('utf-8')
    elif not isinstance(value, bytes):
        value = str(value).encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, (list, np.ndarray)):
        value = np.array(value, dtype=np.float32).flatten().tolist()
    elif not isinstance(value, (list, np.ndarray)):
        value = [float(value)]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, (list, np.ndarray)):
        value = np.array(value, dtype=np.int64).flatten().tolist()
    elif not isinstance(value, (list, np.ndarray)):
        value = [int(value)]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bool_feature(value):
    """Returns a bool feature."""
    if isinstance(value, (list, np.ndarray)):
        value = [bool(v) for v in np.array(value).flatten()]
    else:
        value = [bool(value)]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in value]))


def serialize_trajectory(rlds_batch):
    """
    Serialize a complete trajectory to tf.train.Example.
    Preserves all original keys from the tfrecord.
    """
    features = {}

    # Process observation
    obs = rlds_batch['observation']

    # observation.image_primary: (T, 1, H, W, C)
    features['observation/image_primary'] = _bytes_feature(obs['image_primary'].tobytes())
    features['observation/image_wrist'] = _bytes_feature(obs['image_wrist'].tobytes())
    features['observation/proprio'] = _bytes_feature(obs['proprio'].tobytes())
    features['observation/timestep'] = _int64_feature(obs['timestep'].flatten())
    features['observation/pad_mask'] = _bool_feature(obs['pad_mask'].flatten())

    # observation.pad_mask_dict
    for key in obs['pad_mask_dict'].keys():
        features[f'observation/pad_mask_dict/{key}'] = _bool_feature(obs['pad_mask_dict'][key].flatten())

    # Process task
    task = rlds_batch['task']

    # task.language_instruction
    lang_instr = task['language_instruction'][0]
    if isinstance(lang_instr, bytes):
        lang_instr = lang_instr.decode('utf-8')
    features['task/language_instruction'] = _bytes_feature(lang_instr)

    # task.pad_mask_dict
    for key in task['pad_mask_dict'].keys():
        features[f'task/pad_mask_dict/{key}'] = _bool_feature(task['pad_mask_dict'][key].flatten())

    # task images and proprio
    features['task/image_primary'] = _bytes_feature(task['image_primary'].tobytes())
    features['task/image_wrist'] = _bytes_feature(task['image_wrist'].tobytes())
    features['task/proprio'] = _bytes_feature(task['proprio'].tobytes())
    features['task/timestep'] = _int64_feature(task['timestep'].flatten())

    # action
    features['action'] = _bytes_feature(rlds_batch['action'].tobytes())

    # dataset_name
    dataset_name = rlds_batch['dataset_name'][0]
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode('utf-8')
    features['dataset_name'] = _bytes_feature(dataset_name)

    # absolute_action_mask
    features['absolute_action_mask'] = _bool_feature(rlds_batch['absolute_action_mask'].flatten())

    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(trajectories, output_path):
    """Write a list of trajectories to a tfrecord file."""
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for traj in trajectories:
            example = serialize_trajectory(traj)
            writer.write(example.SerializeToString())


def create_dataset_info(version_dir, shard_lengths, total_bytes):
    """Create dataset_info.json for RLDS."""
    num_shards = len(shard_lengths)

    dataset_info = {
        "citation": "",
        "description": f"BRIDGE dataset filtered by task",
        "fileFormat": "tfrecord",
        "moduleName": "bridge_dataset.bridge_dataset_dataset_builder",
        "name": DATASET_NAME,
        "releaseNotes": {
            VERSION: "Initial release."
        },
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": str(total_bytes),
                "shardLengths": [str(n) for n in shard_lengths]
            }
        ],
        "version": VERSION
    }

    info_path = version_dir / f"dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return info_path


def main():
    parser = argparse.ArgumentParser(description="Split BRIDGE dataset by task and save as RLDS tfrecords.")
    parser.add_argument("--dataset_dirs", type=Path,
                        default='/inspire/ssd/project/robotsimulation/public/data/bridge',
                        help="the dataset root")
    parser.add_argument("--output_dir", type=Path,
                        default='/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/bridge_select_trajs',
                        help="processed output path")
    parser.add_argument("--data_mix", type=str, default='bridge_orig')
    parser.add_argument("--trajs_per_file", type=int, default=50,
                        help="number of trajectories per tfrecord file")
    args = parser.parse_args()

    # Create output directories for each task with RLDS structure
    # Structure: output_dir/task_name/data_mix/VERSION/
    task_version_dirs = {}
    for task_name in candidates_exact.keys():
        version_dir = args.output_dir / task_name / args.data_mix / VERSION
        version_dir.mkdir(parents=True, exist_ok=True)
        task_version_dirs[task_name] = version_dir

    # Load dataset
    dataset = EpisodicRLDSDataset(
        str(args.dataset_dirs),
        args.data_mix,
        batch_transform=None,
        resize_resolution=(256, 256),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=False,
    )
    dataset_length = dataset.dataset_length

    # Buffer for each task
    task_buffers = {task_name: [] for task_name in candidates_exact.keys()}
    task_counts = {task_name: 0 for task_name in candidates_exact.keys()}
    task_file_sizes = {task_name: [] for task_name in candidates_exact.keys()}  # Track file sizes
    task_shard_lengths = {task_name: [] for task_name in candidates_exact.keys()}  # Track trajs per shard

    # Statistics
    total_processed = 0
    skipped = 0

    print(f"Processing dataset, total episodes: {dataset_length}")

    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing episodes"):
        # Get task language instruction
        task_language = rlds_batch['task']['language_instruction'][0].decode()
        tnorm = normalize(task_language)

        # Use keyword_fuzzy_match_exact for filtering (original logic)
        fuzzy_res = {k: keyword_fuzzy_match_exact(tnorm, v, k) for k, v in candidates_exact.items()}
        fuzzy_any = [(k, v) for k, v in fuzzy_res.items() if v[0]]

        if len(fuzzy_any) == 0:
            skipped += 1
            continue

        # Get matched task name
        dataset_name = [fu[0] for fu in fuzzy_any]
        task_name = dataset_name[0]

        # Add to buffer
        task_buffers[task_name].append(rlds_batch)
        task_counts[task_name] += 1
        total_processed += 1

        # Write to file if buffer is full
        if len(task_buffers[task_name]) >= args.trajs_per_file:
            version_dir = task_version_dirs[task_name]
            file_idx = len(task_shard_lengths[task_name])
            filename = f"{DATASET_NAME}-train.tfrecord-{file_idx:05d}-of-?????"  # Will update later
            output_path = version_dir / filename

            write_tfrecord(task_buffers[task_name], output_path)

            # Get file size
            file_size = output_path.stat().st_size
            task_file_sizes[task_name].append(file_size)
            task_shard_lengths[task_name].append(len(task_buffers[task_name]))

            task_buffers[task_name] = []

    # Write remaining trajectories
    for task_name, buffer in task_buffers.items():
        if len(buffer) > 0:
            version_dir = task_version_dirs[task_name]
            file_idx = len(task_shard_lengths[task_name])
            output_path = version_dir / f"{DATASET_NAME}-train.tfrecord-{file_idx:05d}-of-?????"

            write_tfrecord(buffer, output_path)

            file_size = output_path.stat().st_size
            task_file_sizes[task_name].append(file_size)
            task_shard_lengths[task_name].append(len(buffer))

    # Rename files with correct shard count and create dataset_info.json
    for task_name in candidates_exact.keys():
        version_dir = task_version_dirs[task_name]
        num_shards = len(task_shard_lengths[task_name])

        if num_shards == 0:
            continue

        # Rename files with correct shard count
        for file_idx in range(num_shards):
            old_path = version_dir / f"{DATASET_NAME}-train.tfrecord-{file_idx:05d}-of-?????"
            new_path = version_dir / f"{DATASET_NAME}-train.tfrecord-{file_idx:05d}-of-{num_shards:05d}"
            if old_path.exists():
                old_path.rename(new_path)

        # Create dataset_info.json
        total_bytes = sum(task_file_sizes[task_name])
        create_dataset_info(version_dir, task_shard_lengths[task_name], total_bytes)

    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"Skipped (no match): {skipped}")
    print("\nPer-task statistics:")
    for task_name in candidates_exact.keys():
        num_files = len(task_shard_lengths[task_name])
        print(f"  {task_name}:")
        print(f"    Trajectories: {task_counts[task_name]}")
        print(f"    Files: {num_files}")
        print(f"    Output dir: {task_version_dirs[task_name]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
