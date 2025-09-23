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
Preprocess the MM Eureka dataset to parquet format
"""

import argparse
import os

from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str)
    parser.add_argument("--output_dir", type=str, default="~/data/mm_eureka/")
    parser.add_argument("--dataset_name", type=str, default="mm_eureka")
    parser.add_argument("--nproc", type=int, default=16)
    parser.add_argument("--test_split", type=int, default=5, help="split percentage of test set")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    nproc = args.nproc

    instruct_prompt = "You should first thinks about the reasoning process in the mind and then provides the user with the answer."
    instruction_following = (
        r"You should first thinks about the reasoning process in the mind and then provides the user with the answer. "
        r"Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> "
        r"and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, "
        r"which means your output should start with <think> and end with </answer>."
    )

    test_split = args.test_split
    assert test_split > 0 and test_split < 100

    train_dataset = load_dataset("json", data_files=args.jsonl_file, split=f"train[:{1 - test_split}%]")
    test_dataset = load_dataset("json", data_files=args.jsonl_file, split=f"train[-{test_split}%:]")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            id = example.pop("id")
            conversations = example.pop("conversations")
            answer = example.pop("answer")
            image_urls = example.pop("image_urls")

            prompts = []
            for conv in conversations:
                if conv["role"] == "user":
                    if instruct_prompt not in conv["content"]:
                        conv["content"] = instruction_following + " " + conv["content"]
                    prompts.append(conv)
                # skip other roles such as "assistant", "system", etc.

            images = []
            for image_url in image_urls:
                with open(image_url, "rb") as f:
                    images.append({"path": image_url, "bytes": f.read()})

            data = {
                "data_source": dataset_name,
                "prompt": prompts,
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "id": id,
                    "split": split,
                    "index": idx,
                    "answer": answer,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=nproc)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=nproc)

    train_file = os.path.join(args.output_dir, "train.parquet")
    test_file = os.path.join(args.output_dir, "test.parquet")
    train_dataset.to_parquet(train_file)
    print(f"Write Done. train file: {train_file}")
    test_dataset.to_parquet(test_file)
    print(f"Write Done. test file: {test_file}")
