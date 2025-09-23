# Copyright 2025, Shanghai Innovation Institute.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the DeepScaleR dataset to parquet format
"""

import argparse
import json
import os

import datasets

from siirl.utils.extras.hdfs_io import copy, makedirs


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
        return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/deepscaler")
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", default=15)

    args = parser.parse_args()

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    instruction_following = "Let's think step by step and output the final within \\boxed{}."

    if args.source_dir == None:
        args.source_dir = data_source
    raw_dataset = datasets.load_dataset("json", data_files=args.source_dir)
    full_dataset = raw_dataset["train"]
    train_test_split_dataset = full_dataset.train_test_split(test_size=0.1, seed=args.seed)

    train_dataset = train_test_split_dataset["train"]
    test_dataset = train_test_split_dataset["test"]

    def make_map_fn(split_name):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            answer_raw = example.pop("answer")

            question = question_raw + " " + instruction_following
            solution = example.pop("solution")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }

            return data

        return process_fn

    processed_train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    processed_test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    processed_train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    processed_test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
