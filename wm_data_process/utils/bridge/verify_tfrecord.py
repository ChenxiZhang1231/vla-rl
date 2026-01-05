"""
Verify the converted tfrecord files.
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np


def parse_tfrecord(serialized_example):
    """Parse a single trajectory from tfrecord."""
    features = {
        'observation/image_primary': tf.io.FixedLenFeature([], tf.string),
        'observation/image_wrist': tf.io.FixedLenFeature([], tf.string),
        'observation/proprio': tf.io.FixedLenFeature([], tf.string),
        'observation/timestep': tf.io.VarLenFeature(tf.int64),
        'observation/pad_mask': tf.io.VarLenFeature(tf.int64),
        'task/language_instruction': tf.io.FixedLenFeature([], tf.string),
        'task/image_primary': tf.io.FixedLenFeature([], tf.string),
        'task/image_wrist': tf.io.FixedLenFeature([], tf.string),
        'task/proprio': tf.io.FixedLenFeature([], tf.string),
        'task/timestep': tf.io.VarLenFeature(tf.int64),
        'action': tf.io.FixedLenFeature([], tf.string),
        'dataset_name': tf.io.FixedLenFeature([], tf.string),
        'absolute_action_mask': tf.io.VarLenFeature(tf.int64),
    }

    # Add pad_mask_dict features (task-specific)
    for prefix in ['observation', 'task']:
        for key in ['image_primary', 'image_wrist', 'proprio', 'timestep']:
            features[f'{prefix}/pad_mask_dict/{key}'] = tf.io.VarLenFeature(tf.int64)
            features[f'{prefix}/pad_mask_dict/language_instruction'] = tf.io.VarLenFeature(tf.int64)

    parsed = tf.io.parse_single_example(serialized_example, features)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Verify converted tfrecord files")
    parser.add_argument("--input_dir", type=Path,
                        default='/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/bridge_select_trajs',
                        help="directory containing task folders")
    parser.add_argument("--num_trajs", type=int, default=2,
                        help="number of trajectories to verify per file")
    args = parser.parse_args()

    # Find all tfrecord files
    tfrecord_files = list(args.input_dir.glob("**/*.tfrecord"))
    print(f"Found {len(tfrecord_files)} tfrecord files")

    for tfrecord_path in sorted(tfrecord_files):
        print(f"\n{'='*60}")
        print(f"Reading: {tfrecord_path}")
        print(f"{'='*60}")

        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        for i, raw_record in enumerate(dataset):
            if i >= args.num_trajs:
                break

            try:
                parsed = parse_tfrecord(raw_record)

                # Get language instruction
                lang_instr = parsed['task/language_instruction'].numpy().decode('utf-8')
                print(f"\n[{i}] Task: {lang_instr}")

                # Parse image_primary to get shape
                img_bytes = parsed['observation/image_primary'].numpy()
                # Assuming (T, 1, H, W, C) uint8
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                print(f"  observation/image_primary bytes: {len(img_bytes)}")

                # Parse action
                action_bytes = parsed['action'].numpy()
                action_array = np.frombuffer(action_bytes, dtype=np.float32)
                print(f"  action bytes: {len(action_bytes)}, shape hint: {action_array.shape}")

                # Parse timestep
                timestep = tf.sparse.to_dense(parsed['observation/timestep']).numpy()
                print(f"  timestep: {timestep[:5]}... (total {len(timestep)} steps)")

            except Exception as e:
                print(f"  Error parsing trajectory {i}: {e}")


if __name__ == "__main__":
    main()
