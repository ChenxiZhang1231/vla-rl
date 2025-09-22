# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse
import re
from pathlib import Path

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def project_direction_vector(start_point_3d, delta_xyz, left_extrinsic, left_intrinsic):
    direction_end = start_point_3d + np.array(delta_xyz) * 3

    start_point_homogeneous = np.hstack((start_point_3d, [1]))
    direction_end_homogeneous = np.hstack((direction_end, [1]))
    world_to_camera = np.linalg.inv(left_extrinsic)


    start_point_world = world_to_camera @ start_point_homogeneous
    direction_end_world = world_to_camera @ direction_end_homogeneous

    projected_start = left_intrinsic @ start_point_world[:3]
    projected_start /= projected_start[2]  # 齐次坐标归一化

    projected_end = left_intrinsic @ direction_end_world[:3]
    projected_end /= projected_end[2]  # 齐次坐标归一化

    return (int(projected_start[0]), int(projected_start[1])), (int(projected_end[0]), int(projected_end[1]))

def plot_action_curves(actions, index, total_steps):
    time = np.arange(total_steps)
    
    plt.figure(figsize=(8, 6))
    
    # Delta XYZ
    plt.subplot(3, 1, 1)
    plt.plot(time[:index + 1], actions[:index + 1, 0], label='Delta X')
    plt.plot(time[:index + 1], actions[:index + 1, 1], label='Delta Y')
    plt.plot(time[:index + 1], actions[:index + 1, 2], label='Delta Z')
    plt.title("Delta XYZ over Time")
    plt.xlabel("Time")
    plt.ylabel("Delta XYZ")
    plt.legend()
    
    # Delta Rotation
    plt.subplot(3, 1, 2)
    plt.plot(time[:index + 1], actions[:index + 1, 3], label='Delta Roll')
    plt.plot(time[:index + 1], actions[:index + 1, 4], label='Delta Pitch')
    plt.plot(time[:index + 1], actions[:index + 1, 5], label='Delta Yaw')
    plt.title("Delta Rotation over Time")
    plt.xlabel("Time")
    plt.ylabel("Delta Rotation")
    plt.legend()
    
    # Grasp
    plt.subplot(3, 1, 3)
    plt.plot(time[:index + 1], actions[:index + 1, 6], label='Grasp')
    plt.title("Grasp over Time")
    plt.xlabel("Time")
    plt.ylabel("Grasp")
    plt.legend()
    
    plt.tight_layout()
    plt.draw()
    
    # Convert plot to image
    plot_image_path = '/tmp/plot.png'
    plt.savefig(plot_image_path)
    plt.close()
    
    # Read the plot image
    plot_img = cv2.imread(plot_image_path)
    plot_img = cv2.resize(plot_img, (width, 400))  # Resize to match width
    
    return plot_img
        
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
    skip = 2
    for i in range(skip):
        processed_files.pop(0)
    for processed_file in processed_files:


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

        relative_action = []
        for a in actions:
            relative_action.append(a['relative_action'])
        relative_action = np.stack(relative_action)
        

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
        for camera_idx in range(1, 2):

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

            image_folder = f"data/droid_processed_data/{scene}/{data}/{timestep}/images/camera_ext{camera_idx}/left"

            video_filename = f'output_combined_video_{total_frames}.mp4'
            total_frames += 1
            fps = 10  # 每秒帧数，可调整

            # 获取图像列表
            image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")], key=natural_sort_key)
            relative_action = np.array(relative_action)  # 确保 relative_action 是 numpy 数组

            # 获取图像尺寸（假设所有图像尺寸一致）
            sample_image_path = os.path.join(image_folder, image_files[0])
            sample_image = cv2.imread(sample_image_path)
            height, width, _ = sample_image.shape

            # 定义视频编码器为 MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height + 400))

            # 创建一个用于绘制曲线的函数


            # 遍历图像并写入视频
            for i, img_file in enumerate(tqdm(image_files, desc="Processing frames")):
                # 读取图像
                img_path = os.path.join(image_folder, img_file)
                img = cv2.imread(img_path)

                delta_xyz = relative_action[i, :3]
                step_key = steps_list[i]
                start_point_3d = np.array(proprios[step_key]['proprio'])[:3]
                projected_start, projected_end = project_direction_vector(start_point_3d, delta_xyz, left_extrinsic, left_intrinsic)

                # cv2.circle(img, projected_start, radius=5, color=(0, 255, 0), thickness=-1) 
                cv2.arrowedLine(img, projected_start, projected_end, (0, 255, 0), 2, tipLength=0.2)
                output_image_path = os.path.join("debug_image", f"frame_{i:03d}.jpg")
                # cv2.imwrite(output_image_path, img)
                
                
                # 绘制 action 曲线
                plot_img = plot_action_curves(relative_action, i, len(image_files))
                
                # 将图像和曲线图合并
                combined_frame = np.vstack((img, plot_img))
                
                # 写入视频帧
                video.write(combined_frame)

            # 释放视频写入对象
            video.release()
            print(f"视频已保存为 {video_filename}")


            
            