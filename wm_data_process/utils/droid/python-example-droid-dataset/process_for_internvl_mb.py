# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2

import torch
import argparse

from pathlib import Path
from memory_bank_exp2 import SimpleFeatureExtractor, SimpleMemoryBank

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def load_processed_files(name):
    if os.path.exists(f"processed_files_{name}.txt"):
        with open(f"processed_files_{name}.txt", "r") as f:
            return list(line.strip() for line in f)
    return list()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DROID data for training the VLA."
    )
    
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", default="data/debug_jsonl", type=Path)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    processed_files = load_processed_files(args.scene.stem)

    with open("data/aggregated-annotations-030724.json", "r") as f:
        aggregated_annotations = json.load(f)
    valid_uuid = set(aggregated_annotations.keys())

    json_data = []
    total_frames = 0
    action_list = []

    window_size = 20
    bank_size = 5
    history_len = bank_size
    # history_len = 5

    memory_bank = SimpleMemoryBank(window_size=window_size, bank_size=bank_size)
    proprio_hist = deque(maxlen=history_len)


    for processed_file in tqdm(processed_files, desc=f"Processing {args.scene}"):
        scene = processed_file.split("/")[2]
        data = processed_file.split("/")[-2]
        timestep = processed_file.split("/")[-1]
        file_folder = Path("data/droid_processed_data") / scene / data / timestep

        # Load action
        action_path = file_folder / "action.json"
        if not os.path.exists(action_path):
            print(f"Action path {action_path} does not exist")
            continue
        actions = []
        with open(action_path, "r") as f:
            for line in f:
                action_data = json.loads(line)
                actions.append(action_data)
        

        # Load metadata, language, extrinsics, intrinsics
        metadata_path = file_folder / "metadata.json"
        if not os.path.exists(metadata_path):
            print(f"Metadata path {metadata_path} does not exist")
            continue
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        uuid = metadata["uuid"]
        if uuid not in valid_uuid:
            continue
        
        # camera_idx = 1
        for camera_idx in range(1, 3):

            languages = aggregated_annotations[uuid]
            random_keys = random.choice(list(languages.keys()))
            language = languages[random_keys]

            camera_param = metadata[f"camera_ext{camera_idx}"]
            extrinsics = camera_param["left"]["extrinsics"]
            translation = np.array(extrinsics["translation"])
            rotation = np.array(extrinsics["rotation_matrix"])
            left_extrinsic = np.eye(4)
            left_extrinsic[:3, :3] = rotation
            left_extrinsic[:3, 3] = translation
            left_intrinsic = np.array(camera_param["left"]["intrinsic_matrix"])

            trajectory_length = metadata['trajectory_length']

            # assert trajectory_length == len(actions)
            if trajectory_length != len(actions):
                print(f"Trajectory length {trajectory_length} != len(actions) {len(actions)}")
                continue

            image_size = metadata['image_size']

            # Load proprio
            proprio_path = file_folder / "robot_dtata.json"
            with open(proprio_path, "r") as f:
                proprios = json.load(f)

            steps_list = list(proprios.keys()) 

            
            # if len(action_list) != 0:
            #     a=np.stack(action_list)
            #     print(a[:,0:3].max(), a[:,0:3].min())
            memory_bank.clear()
            img_list = []
            for t in range(trajectory_length):
                image_path_ = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/left/left_image_{steps_list[t]}.jpg"
                pil_img = Image.open(f"/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/droid_processed_data/{image_path_}")
                pil_img = np.array(pil_img)[None, :]
                img_list.append(pil_img)
            img_list = np.concatenate(img_list, axis=0)
            img_feat_ep = memory_bank.feature_extractor.extract_image_feature(img_list)

            for step in range(trajectory_length):
                action = np.array(actions[step]['relative_action'])  # 7,
                action[:3] = action[0:3] * 15
                action[3:6] = action[3:6] * 5
                # action[6:7] = (action[6:7] >= 0.5).astype(np.float32)
                # action_list.append(action)
                proprio = np.array(proprios[steps_list[step]]['proprio'])  # 6,

                if len(proprio_hist) == 0:
                    for _ in range(history_len):
                        proprio_hist.append(np.zeros_like(proprio))
                    proprio_hist.append(proprio)
                else:
                    proprio_hist.append(proprio)

                # Load image
                image_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/left/left_image_{steps_list[step]}.jpg"

                # Load depth
                depth_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/depth/depth_image_{steps_list[step]}.png"

                if memory_bank.get_length() == 0:
                    for _ in range(history_len):
                        memory_bank.add(img_feat_ep[step], int(steps_list[step].split("_")[-1]), from_img=False)
                    memory_bank.add(img_feat_ep[step], int(steps_list[step].split("_")[-1]), from_img=False)
                else:
                    memory_bank.add(img_feat_ep[step], int(steps_list[step].split("_")[-1]), from_img=False)


                entry1 = {
                    "id": total_frames,
                    "frame_id": [],
                    "image": [],
                    "image_depth": [],
                    "width_list": [],
                    "height_list": [],
                    "width_depth_list": [],
                    "height_depth_list": [],
                    # "image": [image_path],
                    # "image_depth": [depth_path],
                    # "width_list": [image_size[0]],
                    # "height_list": [image_size[1]],
                    # "width_depth_list": [image_size[0]],
                    # "height_depth_list": [image_size[1]],
                    "conversations": [],
                    # "proprio": [proprio.tolist()],
                    "action": action.tolist(), 
                    "proprio": [],
                    "camera_info": {
                        "camera_extrinsic": left_extrinsic.tolist(),
                        "camera_intrinsic": left_intrinsic.tolist()
                    }
                }

                fid_list = memory_bank.get_fid()[:-1]
                for f in fid_list:
                    # img_path = f"ep{ep_idx}-fid{f}.jpg"
                    entry1['frame_id'].append(f)
                    image_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/left/left_image_{f}.jpg"
                    entry1['image'].append(image_path)
                    entry1['width_list'].append(image_size[0])
                    entry1['height_list'].append(image_size[1])
                    
                    depth_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/depth/depth_image_{f}.png"
                    entry1['image_depth'].append(depth_path)
                    entry1['width_depth_list'].append(image_size[0])
                    entry1['height_depth_list'].append(image_size[1])
                
                entry1['frame_id'].append(int(steps_list[step].split("_")[-1]))
                image_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/left/left_image_{steps_list[step]}.jpg"
                entry1['image'].append(image_path)
                entry1['width_list'].append(image_size[0])
                entry1['height_list'].append(image_size[1])
                
                depth_path = f"{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/depth/depth_image_{steps_list[step]}.png"
                entry1['image_depth'].append(depth_path)
                entry1['width_depth_list'].append(image_size[0])
                entry1['height_depth_list'].append(image_size[1])

                q = {
                    "from": "human",
                    "value": "Current:\n<image>\n " + f"What action should the robot take to {language}"
                }
                a = {
                    "from": "gpt",
                    "value": f"<state_pred>"
                    # "value": f"<action>[{', '.join(formatted_action)}]</action>"
                }
                entry1["conversations"].append(q)
                entry1["conversations"].append(a)

                for i in range(history_len):
                    entry1[f"proprio"].append(proprio_hist[i].tolist())


                json_data.append(entry1)
                total_frames += 1

                if len(entry1['frame_id']) == 1:
                    raise
    
    len_data = len(json_data) // 1000
    jsonl_file_name = f"{args.scene.stem}_{len_data}k.jsonl"

    jsonl_file_path = os.path.join(args.output, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 
        

    print(f"Finished processing {total_frames} batches. Saved to {jsonl_file_path}")

        





        

