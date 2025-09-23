#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import rerun as rr
import cv2
from scipy.spatial.transform import Rotation
import glob
import h5py
import json
from common import h5_tree, CAMERA_NAMES, log_angle_rot, blueprint_row_images, extract_extrinsics, log_cartesian_velocity, POS_DIM_NAMES, link_to_world_transform
from rerun_loader_urdf import URDFLogger
import argparse
import os
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import torch
import time
# import logging
# logging.getLogger().setLevel(logging.WARNING) 
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import gc

def replace_nan_with_nearest(depth_image):
    depth_image = np.where(np.isinf(depth_image), np.nan, depth_image)
    nan_mask = np.isnan(depth_image)

    non_nan_indices = np.array(np.nonzero(~nan_mask)).T
    non_nan_values = depth_image[~nan_mask]

    nan_indices = np.array(np.nonzero(nan_mask)).T

    if len(non_nan_values) > 0:
        interpolator = interpolate.NearestNDInterpolator(non_nan_indices, non_nan_values)
        depth_image[nan_mask] = interpolator(nan_indices)

    return depth_image



def is_noop(action, prev_action=None, xyz_threshold=5e-3, rot_threshold=5e-3):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """

    no_move = (action[:3] < xyz_threshold).all() and (action[3:6] < rot_threshold).all()
    if prev_action is None:
        return no_move
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    no_gripper = ((gripper_action == 0) and (prev_gripper_action == 0)) or (gripper_action >= 0.5 and prev_gripper_action >= 0.5)

    
    return (no_move and no_gripper)

def convert_to_serializable(data):
    """
    递归地将 NumPy 类型（如 int64、float64）转换为 Python 原生类型
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # 转换 NumPy 数组为列表
    elif isinstance(data, np.integer):
        return int(data)  # 转换 NumPy 整数为 Python int
    elif isinstance(data, np.floating):
        return float(data)  # 转换 NumPy 浮点数为 Python float
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}  # 递归处理字典
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]  # 递归处理列表
    else:
        return data  # 保持其他类型不变
    
class StereoCamera:
    left_images: list[np.ndarray]
    right_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    width: float
    height: float
    left_dist_coeffs: np.ndarray
    left_intrinsic_mat: np.ndarray

    right_dist_coeffs: np.ndarray
    right_intrinsic_mat: np.ndarray

    def __init__(self, recordings: Path, serial: int, device_id: int, new_image_size: tuple[int, int] = (1280, 720)):
        
        try:
            import pyzed.sl as sl
            init_params = sl.InitParameters()
            init_params.sdk_verbose = 0
            # print(f"device_id: {device_id}")
            init_params.sdk_gpu_id = device_id
            svo_path = recordings / "SVO" / f"{serial}.svo"
            init_params.set_from_svo_file(str(svo_path))
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = 0.2

            zed = sl.Camera()
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise Exception(f"Error reading camera data: {err}")

            params = (
                zed.get_camera_information().camera_configuration.calibration_parameters
            )
            
            self.left_intrinsic_mat = np.array(
                [
                    [params.left_cam.fx, 0, params.left_cam.cx],
                    [0, params.left_cam.fy, params.left_cam.cy],
                    [0, 0, 1],
                ]
            )
            self.right_intrinsic_mat = np.array(
                [
                    [params.right_cam.fx, 0, params.right_cam.cx],
                    [0, params.right_cam.fy, params.right_cam.cy],
                    [0, 0, 1],
                ]
            )
            self.zed = zed

            if new_image_size != (1280, 720):
                scale_x = new_image_size[0] / 1280
                scale_y = new_image_size[1] / 720 
                self.left_intrinsic_mat = np.array(
                    [
                        [params.left_cam.fx * scale_x, 0, params.left_cam.cx * scale_x],
                        [0, params.left_cam.fy * scale_y, params.left_cam.cy * scale_y],
                        [0, 0, 1],
                    ]
                )
                self.right_intrinsic_mat = np.array(
                    [
                        [params.right_cam.fx * scale_x, 0, params.right_cam.cx * scale_x],
                        [0, params.right_cam.fy * scale_y, params.right_cam.cy * scale_y],
                        [0, 0, 1],
                    ]
                )

        except ModuleNotFoundError:
            # pyzed isn't installed we can't find its intrinsic parameters
            # so we will have to make a guess.
            self.left_intrinsic_mat = np.array([
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ])
            self.right_intrinsic_mat = self.left_intrinsic_mat
            
            mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
            if (recordings / "MP4" / f'{serial}-stereo.mp4').exists():
                mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
            elif (recordings / "MP4" / f'{serial}.mp4').exists():
                # Sometimes they don't have the '-stereo' suffix
                mp4_path = recordings / "MP4" / f'{serial}.mp4'
            else:
                raise Exception(f"unable to video file for camera {serial}")

            self.cap = cv2.VideoCapture(str(mp4_path))
            print(f"opening {mp4_path}")


    def get_next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Gets the the next from both cameras and maybe computes the depth."""

        if hasattr(self, "zed"):
            # We have the ZED SDK installed.
            import pyzed.sl as sl
            left_image = sl.Mat()
            # right_image = sl.Mat()
            depth_image = sl.Mat()

            rt_param = sl.RuntimeParameters()
            err = self.zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                left_image = np.array(left_image.numpy())

                # self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                # right_image = np.array(right_image.numpy())
                right_image = None

                self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                depth_image = np.array(depth_image.numpy())

                return (left_image, right_image, depth_image)
            else:
                return None
        else:
            # We don't have the ZED sdk installed
            ret, frame = self.cap.read()
            if ret:
                left_image = frame[:,:1280,:]
                right_image = frame[:,1280:,:]
                return (left_image, right_image, None)
            else:
                print("empty!")
                return None

class RawScene:
    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, StereoCamera]
    device_id: int

    def __init__(self, dir_path: Path, save_path: Path, device_id: int):
        self.dir_path = dir_path
        self.image_save_path = save_path / "images/"
        self.action_save_path = save_path / "action.json"
        self.mate_save_path = save_path / "metadata.json"
        self.state_save_path = save_path / "robot_dtata.json"
        self.new_image_size = (32*14, 18*14)


        json_file_paths = glob.glob(str(self.dir_path) + "/*.json")
        if len(json_file_paths) < 1:
            raise Exception(f"Unable to find metadata file at '{self.dir_path}'")

        with open(json_file_paths[0], "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.trajectory = h5py.File(str(self.dir_path / "trajectory.h5"), "r")
        self.action = self.trajectory['action']
        proprio = self.trajectory['observation']['robot_state']['cartesian_position'][:]
        abs_action = self.trajectory['action']['cartesian_position'][:]
        rel_action_xyz = abs_action[:, :3] - proprio[:, :3]

        abs_action_angle = abs_action[:, 3:]
        proprio_angle = proprio[:, 3:]

        abs_action_rotation = Rotation.from_euler('xyz', abs_action_angle, degrees=False).as_matrix()
        proprio_rotation = Rotation.from_euler('xyz', proprio_angle, degrees=False).as_matrix()

        relative_rotation = abs_action_rotation @ proprio_rotation.transpose(0, 2, 1)
        delta_euler_angle = Rotation.from_matrix(relative_rotation).as_euler('xyz', degrees=False)

        rel_action_rot = delta_euler_angle

        self.rel_action = np.concatenate([rel_action_xyz, rel_action_rot], axis=1)

        
        self.rel_action = np.concatenate([self.rel_action, self.action['gripper_position'][:][:, None]], axis=1)

        # We ignore the robot_state under action/, don't know why where is two different robot_states.
        self.robot_state = self.trajectory['observation']['robot_state']
        # h5_tree(self.trajectory)

        self.trajectory_length = self.metadata["trajectory_length"]

        # Mapping from camera name to it's serial number.
        self.serial = {
            camera_name: self.metadata[f"{camera_name}_cam_serial"]
            for camera_name in CAMERA_NAMES
        }

        self.cameras = {}
        for camera_name in CAMERA_NAMES:
            self.cameras[camera_name] = StereoCamera(
                self.dir_path / "recordings",
                self.serial[camera_name],
                device_id,
                self.new_image_size,
                
            )

    def log_cameras_next(self, i: int, base_folder: str, all_camera_data: dict, log_another=False) -> None:
        """
        Log data from cameras at step `i`.
        All data will be saved into a single JSON file for each camera.
        """

        for camera_name, camera in self.cameras.items():
            # 创建保存当前相机数据的文件夹
            camera_folder = os.path.join(base_folder, f"camera_{camera_name}")
            os.makedirs(camera_folder, exist_ok=True)

            if camera_name not in all_camera_data:
                all_camera_data[camera_name] = {}

            # 记录时间戳
            time_stamp_camera = self.trajectory["observation"]["timestamp"][
                "cameras"
            ][f"{self.serial[camera_name]}_estimated_capture"][i]
            
            # step_data = {
            #     'real_time': time_stamp_camera
            # }
            step_data = {}

            # 记录左相机的外参和内参
            extrinsics_left = self.trajectory["observation"]["camera_extrinsics"][
                f"{self.serial[camera_name]}_left"
            ][i]
            rotation_left = Rotation.from_euler(
                "xyz", np.array(extrinsics_left[3:])
            ).as_matrix()
            
            step_data['left'] = {
                'intrinsic_matrix': convert_to_serializable(camera.left_intrinsic_mat),
                'extrinsics': {
                    'translation': convert_to_serializable(extrinsics_left[:3]),
                    'rotation_matrix': convert_to_serializable(rotation_left)
                },
                'pose': convert_to_serializable(extrinsics_left)
            }

            # 记录右相机的外参和内参
            # extrinsics_right = self.trajectory["observation"]["camera_extrinsics"][
            #     f"{self.serial[camera_name]}_right"
            # ][i]
            # rotation_right = Rotation.from_euler(
            #     "xyz", np.array(extrinsics_right[3:])
            # ).as_matrix()
            
            # step_data['right'] = {
            #     'intrinsic_matrix': convert_to_serializable(camera.right_intrinsic_mat),
            #     'extrinsics': {
            #         'translation': convert_to_serializable(extrinsics_right[:3]),
            #         'rotation_matrix': convert_to_serializable(rotation_right)
            #     }
            # }

            # 记录深度相机
            depth_translation = np.array(extrinsics_left[:3])
            step_data['depth'] = {
                'intrinsic_matrix': convert_to_serializable(camera.left_intrinsic_mat),
                'extrinsics': {
                    'translation': convert_to_serializable(depth_translation),
                    'rotation_matrix': convert_to_serializable(rotation_left)
                }
            }

            # 将该步骤的数据保存到相机数据中
            all_camera_data[camera_name][f'step_{i}'] = step_data

            # 获取下一帧的图像
            frames = camera.get_next_frame()
            if frames:
                left_image, right_image, depth_image = frames

                # 保存左相机图像
                left_image = cv2.resize(left_image, self.new_image_size)
                camera_folder_left = os.path.join(camera_folder, "left")
                os.makedirs(camera_folder_left, exist_ok=True)
                left_image_path = os.path.join(camera_folder_left, f"left_image_step_{i}.jpg")
                cv2.imwrite(left_image_path, left_image)

                if log_another:
                    another_path = os.path.join("/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/400steps", f"left_image_step_{i}.png")
                    cv2.imwrite(another_path, left_image)

                # # 保存右相机图像
                # right_image = cv2.resize(right_image, self.new_image_size)
                # camera_folder_right = os.path.join(camera_folder, "right")
                # os.makedirs(camera_folder_right, exist_ok=True)
                # right_image_path = os.path.join(camera_folder_right, f"right_image_step_{i}.jpg")
                # cv2.imwrite(right_image_path, right_image)

                # 保存深度图像
                if depth_image is not None:
                    # 将远处的点置为 0
                    depth_image[depth_image > 1.8] = 0

                    # depth_image = np.nan_to_num(depth_image, nan=0.0)
                    depth_image = cv2.resize(depth_image, self.new_image_size, interpolation=cv2.INTER_NEAREST)
                    # depth_image = replace_nan_with_nearest(depth_image)
                    depth_image = np.where(np.isinf(depth_image), np.nan, depth_image)
                    depth_image = np.nan_to_num(depth_image, nan=0.0)
                    depth_image = depth_image.astype(np.float32)
                    depth_image = (depth_image * 256).astype(np.uint16)
                    camera_folder_depth = os.path.join(camera_folder, "depth")
                    os.makedirs(camera_folder_depth, exist_ok=True)
                    # depth_image_path = os.path.join(camera_folder_depth, f"depth_image_step_{i}.npy")
                    # np.save(depth_image_path, depth_image)
                    depth_image_path = os.path.join(camera_folder_depth, f"depth_image_step_{i}.png")
                    cv2.imwrite(depth_image_path, depth_image)

        return all_camera_data
    
    def save_meta_data_to_json(self, all_camera_data: dict) -> None:
        """
        Save all metadata to a single JSON file in the base folder.
        """
        key_list = list(all_camera_data['ext1'].keys())
        meta_data = {
            'uuid': self.metadata['uuid'],
            'trajectory_length': self.real_len,
            'camera_ext1': all_camera_data['ext1'][key_list[0]],
            'camera_ext2': all_camera_data['ext2'][key_list[0]],
            # 'camera_wrist': all_camera_data['wrist']['step_0'],
            'current_task': self.metadata['current_task'],
            'image_size': self.new_image_size
        }
        # for i in key_list:
        #     camera_wrist = all_camera_data['wrist'][i]
        #     meta_data[f'camera_wrist_step_{i}'] = camera_wrist

        # 保存所有元数据到一个 JSON 文件中
        with open(self.mate_save_path, 'w') as json_file:
            json.dump(meta_data, json_file, indent=4)


    def save_camera_data_to_json(self, file_path: str, all_camera_data: dict) -> None:
        """
        Save all camera data to a single JSON file in the base folder.
        """
        for camera_name, camera_data in all_camera_data.items():
            camera_folder = os.path.join(file_path, f"camera_{camera_name}")
            json_path = os.path.join(camera_folder, "camera_data.json")
            
            # 将所有步骤的数据写入到单个 JSON 文件中
            with open(json_path, 'w') as json_file:
                json.dump(camera_data, json_file, indent=4)

    def log_action(self, i: int, file_path: str) -> None:
        log_data = {}

        # 记录 Cartesian Position
        pose = self.trajectory['action']['cartesian_position'][i]
        trans, mat = extract_extrinsics(pose)
        
        # 将 ndarray 转换为 list
        log_data['cartesian_position'] = {
            'transform': {
                'translation': trans.tolist() if isinstance(trans, np.ndarray) else trans,
                'mat3x3': mat.tolist() if isinstance(mat, np.ndarray) else mat
            },
            'origin': trans.tolist() if isinstance(trans, np.ndarray) else trans,
            'pose': pose.tolist() if isinstance(pose, np.ndarray) else pose
        }
        log_data['relative_action'] = self.rel_action[i].tolist() \
            if isinstance(self.rel_action[i], np.ndarray) else self.rel_action[i]

        # 记录 Cartesian Velocity
        log_data['cartesian_velocity'] = self.action['cartesian_velocity'][i].tolist() \
            if isinstance(self.action['cartesian_velocity'][i], np.ndarray) else self.action['cartesian_velocity'][i]

        # 记录 Gripper Position 和 Velocity
        log_data['gripper_position'] = self.action['gripper_position'][i].tolist() \
            if isinstance(self.action['gripper_position'][i], np.ndarray) else self.action['gripper_position'][i]

        log_data['gripper_velocity'] = self.action['gripper_velocity'][i].tolist() \
            if isinstance(self.action['gripper_velocity'][i], np.ndarray) else self.action['gripper_velocity'][i]

        # # 记录 Joint Velocities
        # log_data['joint_velocity'] = {}
        # for j, vel in enumerate(self.trajectory['action']['cartesian_position'][i]):
        #     log_data['joint_velocity'][f'joint_{j}'] = vel.tolist() if isinstance(vel, np.ndarray) else vel

        # # 记录 Target Cartesian Position
        # pose = self.trajectory['action']['target_cartesian_position'][i]
        # trans, mat = extract_extrinsics(pose)
        # log_data['target_cartesian_position'] = {
        #     'transform': {
        #         'translation': trans.tolist() if isinstance(trans, np.ndarray) else trans,
        #         'mat3x3': mat.tolist() if isinstance(mat, np.ndarray) else mat
        #     },
        #     'origin': trans.tolist() if isinstance(trans, np.ndarray) else trans
        # }

        # # 记录 Target Gripper Position
        # log_data['target_gripper_position'] = self.action['target_gripper_position'][i].tolist() \
        #     if isinstance(self.action['target_gripper_position'][i], np.ndarray) else self.action['target_gripper_position'][i]

        # 将数据追加到文件中
        with open(file_path, 'a') as f:
            json.dump(log_data, f)
            f.write('\n')  # 确保每条记录都换行存储
        
    def log_robot_state(self, i: int, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]], all_robot_data: dict) -> None:
        robot_data = {}

        # 记录关节角度
        joint_angles = self.robot_state['joint_positions'][i]
        robot_data['joint_positions'] = joint_angles.tolist()

        # 记录轨迹 (如果 i > 1)
        # if i > 1:
        #     lines = []
        #     for j in range(i-1, i+1):
        #         joint_angles = self.robot_state['joint_positions'][j]
        #         joint_origins = []
        #         for joint_idx in range(len(joint_angles)+1):
        #             transform = link_to_world_transform(entity_to_transform, joint_angles, joint_idx+1)
        #             joint_org = (transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        #             joint_origins.append(list(joint_org))
        #         lines.append(joint_origins)
            
        #     # 保存轨迹线条
        #     robot_data['trajectory'] = lines

        # 记录夹爪位置
        robot_data['gripper_position'] = self.robot_state['gripper_position'][i]

        # # 记录关节速度
        # joint_velocities = self.robot_state['joint_velocities'][i]
        # robot_data['joint_velocities'] = joint_velocities.tolist()

        # # 记录计算得到的关节力矩
        # joint_torques_computed = self.robot_state['joint_torques_computed'][i]
        # robot_data['joint_torques_computed'] = joint_torques_computed.tolist()

        # # 记录测量的电机力矩
        # motor_torques_measured = self.robot_state['motor_torques_measured'][i]
        # robot_data['motor_torques_measured'] = motor_torques_measured.tolist()

        proprio = self.robot_state['cartesian_position'][i]
        robot_data['proprio'] = proprio.tolist()

        # 将该步骤的数据存储在 all_robot_data 中
        all_robot_data[f'step_{i}'] = robot_data

        return all_robot_data

    def save_robot_data_to_json(self, base_folder: str, all_robot_data: dict) -> None:
        """
        Save all robot data to a single JSON file.
        """
        # robot_folder = os.path.join(base_folder, "robot_state")
        # os.makedirs(robot_folder, exist_ok=True)

        # json_path = os.path.join(robot_folder, "robot_data.json")
        
        with open(base_folder, 'w') as json_file:
            json.dump(all_robot_data, json_file, indent=4)

    def log(self, urdf_logger) -> None:
        all_camera_data = {}
        all_robot_data = {}

        downsample_rate = 1
        log_another = False
        if self.trajectory_length > 100:
            downsample_rate = self.trajectory_length // 100
        else:
            downsample_rate = 1

        self.real_len = 0
        for i in range(self.trajectory_length-3):
            if i == 0:
                # We want to log the robot model here so that it appears in the right timeline
                urdf_logger.log()
            
            prev_action = None if i == 0 else self.rel_action[i-1]
            if is_noop(self.rel_action[i], prev_action):
                for camera_name, camera in self.cameras.items():
                    frames = camera.get_next_frame()
                continue

            if i % downsample_rate != 0:
                for camera_name, camera in self.cameras.items():
                    frames = camera.get_next_frame()
                continue

            self.real_len += 1

            self.log_action(i, self.action_save_path)
            all_camera_data = self.log_cameras_next(i, self.image_save_path, all_camera_data, log_another)
            all_robot_data = self.log_robot_state(i, urdf_logger.entity_to_transform, all_robot_data)


        # self.save_camera_data_to_json(self.image_save_path, all_camera_data)
        self.save_meta_data_to_json(all_camera_data)
        self.save_robot_data_to_json(self.state_save_path, all_robot_data)

def process_timestamp_folder(date_folder, timestamp_folder, raw_folder, save_path, urdf_logger, device_id, processed_files, valid_uuid):
    task_id = f"{raw_folder}/{date_folder}/{timestamp_folder}"
    if task_id in processed_files:
        print(f"Skipping {task_id}, already processed.")
        return
    
    date_path = raw_folder / date_folder
    timestamp_path = date_path / timestamp_folder
    save_path_ = save_path / date_folder / timestamp_folder
    os.makedirs(save_path_, exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    if not os.path.exists(str(timestamp_path / "trajectory.h5")):
        print(f"Skipping {task_id}, no trajectory file.")
        return

    json_file_paths = glob.glob(str(timestamp_path) + "/*.json")

    try:
        with open(json_file_paths[0], "r") as metadata_file:
            metadata = json.load(metadata_file)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to process {timestamp_path}")
        return

    if metadata["uuid"] not in valid_uuid:
        print(f"Skipping {task_id}, invalid uuid.")
        return

    try:
        raw_scene = RawScene(timestamp_path, save_path_, device_id=device_id)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to process {timestamp_path}")
        return
        
    st = time.time()
    # raw_scene.log(urdf_logger)
    try:
        raw_scene.log(urdf_logger)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to process {timestamp_path}")
        return
    ed = time.time()
    for camera_name, camera in raw_scene.cameras.items():
        camera.zed.close()
    gc.collect() 
    print(f"Processed {date_folder}/{timestamp_folder} in {ed-st:.2f}s")
    # print(f"Processed {date_folder}/{timestamp_folder}")
    # for step in tqdm(range(1), desc=f"Processing {timestamp_folder} on GPU {device_id}", leave=False):
    #     raw_scene.log(urdf_logger)
    with open(f"processed_files_{raw_folder.parts[-2]}.txt", "a") as f:
        f.write(f"{task_id}\n")
    print(f"Processed {task_id}")
    

def load_processed_files(name):
    if os.path.exists(f"processed_files_{name}.txt"):
        with open(f"processed_files_{name}.txt", "r") as f:
            return set(line.strip() for line in f)
    return set()

def main():
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )
    
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--save_path", default='data/debug/', type=Path)
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads to use for processing.")
    args = parser.parse_args()

    urdf_logger = URDFLogger(args.urdf)

    save_path = args.save_path / args.scene.stem
    
    processed_files = load_processed_files(args.scene.stem)
    
    with open("data/aggregated-annotations-030724.json", "r") as f:
        aggregated_annotations = json.load(f)
    valid_uuid = set(aggregated_annotations.keys())
    del aggregated_annotations
    
    tasks = []
    raw_folder = args.scene / "success"
    date_folders = os.listdir(raw_folder)
    for date_folder in date_folders:
        timestamp_folders = os.listdir(raw_folder / date_folder)
        for idx, timestamp_folder in enumerate(timestamp_folders):
            device_id = idx % 8
            # print(f"device_id: {device_id}")    
            tasks.append((date_folder, timestamp_folder, raw_folder, save_path, urdf_logger, device_id, processed_files, valid_uuid))
            
    raw_folder = args.scene / "failure"
    date_folders = os.listdir(raw_folder)
    for date_folder in date_folders:
        timestamp_folders = os.listdir(raw_folder / date_folder)
        for idx, timestamp_folder in enumerate(timestamp_folders):
            device_id = idx % 8
            # print(f"device_id: {device_id}")    
            tasks.append((date_folder, timestamp_folder, raw_folder, save_path, urdf_logger, device_id, processed_files, valid_uuid))

    total_tasks = len(tasks) - len(processed_files)
    with Pool(processes=args.num_threads) as pool:
        with tqdm(total=total_tasks, desc="Total Progress") as pbar:
            for _ in pool.starmap(process_timestamp_folder, tasks):
                pbar.update(1)


if __name__ == "__main__":
    main()