
import argparse
from pathlib import Path
import json
import os
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import uuid
from PIL import Image
import random 

from tqdm import tqdm
from collections import defaultdict

import sys
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)



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


def main():
    parser = argparse.ArgumentParser(description="Process dataset videos with synchronized metadata + 2D projections.")
    parser.add_argument("--dataset_dirs", type=Path, default='/inspire/ssd/project/robotsimulation/public/data/bridge', help="the dataset root")
    parser.add_argument("--output_dir", type=Path, default='/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/bridge_select_trajs', help="processed output path")
    parser.add_argument("--data_mix", type=str, default='bridge_orig')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    ep_idx = 0
    entry_list = []
    cnt = defaultdict(int) 
    cnt_joint = defaultdict(int) 
    group = defaultdict(list)
    group_task = defaultdict(list)
    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing episodes"):
        if ep_idx >= dataset_length:
            break
        ep_idx += 1
        
        task_language = rlds_batch['task']['language_instruction'][0].decode()
        tnorm = normalize(task_language)
        # fuzzy_res = {k: keyword_fuzzy_match(tnorm, v) for k, v in candidates.items()}
        fuzzy_res = {k: keyword_fuzzy_match_exact(tnorm, v, k) for k, v in candidates_exact.items()}
        fuzzy_any = [(k, v) for k,v in fuzzy_res.items() if v[0]]
        dataset_name = [fu[0] for fu in fuzzy_any]
        if len(dataset_name) == 0:
            continue
        for name in dataset_name:
            cnt[name] += 1
            group[name].append(task_language)
        if len(dataset_name) > 1:
            raise
            dataset_name_sort = sorted(dataset_name)
            dataset_name_joint = ""
            for name_sort in dataset_name_sort:
                dataset_name_joint += name_sort
            cnt_joint[dataset_name_joint] += 1
            
        img_path = f"images/{dataset_name[0]}_{str(uuid.uuid1())[:8]}.png"
        img_save_path = args.output_dir / img_path
        
        arr = rlds_batch['observation']['image_primary'][0][0] 
        arr = np.ascontiguousarray(arr)
        img = Image.fromarray(arr)
        os.makedirs(img_save_path.parent, exist_ok=True)
        img.save(img_save_path)
        
        entry = {
            "task_type": dataset_name,
            "task_language": [task_language],
            "init_state_path": [str(img_path)],
        }
        group_task[dataset_name[0]].append(entry)
        entry_list.append(entry)
        
        # if len(entry_list) > 10:
        #     break
    
    # entry_final = []
    # for name in group_task.keys():
    #     entry_list = group_task[name]
    #     entry_sample = random.sample(entry_list, 100)
    #     entry_final.extend(entry_sample)
        
    entry_final = entry_list
    entry_path = args.output_dir / 'dataset.jsonl'
    with open(entry_path, "w", encoding="utf-8") as f:
        for item in entry_final:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
