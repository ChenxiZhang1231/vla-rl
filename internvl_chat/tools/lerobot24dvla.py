# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &
# 
import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2
import glob
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import packaging.version
import datasets
from datasets import concatenate_datasets, load_dataset
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internvl.model.smolvla import SmolVLAConfig
from lerobot.datasets.video_utils import get_safe_default_codec
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import (
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    get_delta_indices,
    check_version_compatibility,
    create_empty_dataset_info,
    get_hf_features_from_features,
    check_timestamps_sync,
    get_episode_data_index,
    hf_transform_to_torch,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,

)
from lerobot.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)

# from internvl.vla.datasets import RLDSDataset, EpisodicRLDSDataset
from internvl.model.internvl_chat.memory_bank_exp2 import SimpleFeatureExtractor, SimpleMemoryBank

def quat_to_matrix(q):
    """将四元数转为 3x3 旋转矩阵"""
    return R.from_quat(q).as_matrix()

def delta_rot_quat(prev_q, curr_q):
    R_prev = R.from_quat(prev_q).as_matrix()
    R_curr = R.from_quat(curr_q).as_matrix()
    
    R_delta = R_prev.T @ R_curr
    delta_euler = R.from_matrix(R_delta).as_euler('xyz', degrees=False)
    return delta_euler.tolist()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def resolve_delta_timestamps(
    policy_cfg, ds_meta,
) -> dict[str, list] | None:
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action" and policy_cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in policy_cfg.action_delta_indices]
        if key.startswith("observation.") and policy_cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in policy_cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


CODEBASE_VERSION = "v2.1"
class LeRobotDatasetMetadata:
    def __init__(
        self,
        meta,
    ):
        self.revision = CODEBASE_VERSION
        self.root = Path(meta['root'])
        self.load_metadata()


    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
    
    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size
    
    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]
    
    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]
        
    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

class LeRobotDataset(Dataset):
    def __init__(
        self,
        meta,
        episodes: list[int] | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        Copy form lerobot/src/lerobot/datasets/lerobot_dataset.py
        """
        super().__init__()
        root = meta['root']
        self.root = Path(root)
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            meta,
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset
    
    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths


    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int, flip=True) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            if flip:
                flipped_frames = torch.flip(frames, dims=[-1])
            else:
                flipped_frames = frames
            item[vid_key] = flipped_frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                item = self.hf_dataset[idx]
                ep_idx = item["episode_index"].item()

                query_indices = None
                if self.delta_indices is not None:
                    query_indices, padding = self._get_query_indices(idx, ep_idx)
                    query_result = self._query_hf_dataset(query_indices)
                    item = {**item, **padding}
                    for key, val in query_result.items():
                        item[key] = val
                item['action'] = item['action'][:, 0]
                if len(self.meta.video_keys) > 0:
                    current_ts = item["timestamp"].item()
                    query_timestamps = self._get_query_timestamps(current_ts, query_indices)
                    video_frames = self._query_videos(query_timestamps, ep_idx, flip=False)
                    item = {**video_frames, **item}

                # Add task as a string
                task_idx = item["task_index"].item()
                item["task"] = self.meta.tasks[task_idx]
                
                for key, values in item.items():
                    if key != "task":
                        item[key] = item[key].to(torch.bfloat16)
                break
            except Exception as e:
                 try_cnt += 1
                 print(e, flush=True)
                 idx = random.randint(0, self.num_frames - 1)

        # return item
        return {
            'batch': item
        }

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


if __name__ == "__main__":
    # data_root_dir = "/SSD_DISK/users/zhangjiahui/openvla/data/franka/replay_traj_sampled_task3"
    data_root_dir = '/SSD_DISK/users/zhangjiahui/data/pick_black_eraser_kb'
    saved_path = '/SSD_DISK/users/zhangjiahui/data/pick_black_eraser_kb_4dvla'

    policy_cfg = SmolVLAConfig()
    ds_meta = LeRobotDatasetMetadata(
        {'root': data_root_dir}
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        {'root': data_root_dir},
        delta_timestamps=delta_timestamps,
        video_backend=get_safe_default_codec(),
    )
    output_features = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(1, 7),
    )
    normalize_targets = Normalize(
            {'action': output_features}, policy_cfg.normalization_mapping, ds_meta.stats, device_name='cpu'
    )

    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False


    window_size = 20
    bank_size = 5
    history_len = bank_size
    sample_interval = 1

    memory_bank = SimpleMemoryBank(window_size=window_size, bank_size=bank_size)
    memory_bank.feature_extractor.resnet50.to("cuda")
    len_dataset = dataset.__len__()
    cur_ep_idx = None
    json_data1 = []
    last_data = None
    ep_data_list = []
    for idx in tqdm(range(len_dataset)):
        data_curr = dataset.__getitem__(idx)['batch']
        ep_idx = int(data_curr['episode_index'].item())
        frame_idx = int(data_curr['frame_index'].item())
        if cur_ep_idx is None:
            cur_ep_idx = ep_idx
        if cur_ep_idx == ep_idx:
            ep_data_list.append(data_curr)
            continue
        cur_ep_idx = ep_idx
        last_data = data_curr
        prev_pos, prev_quat = None, None

        is_zero_saved = False
        total_frames = 0
        
        image_path_hist = deque(maxlen=history_len)
        depth_path_hist = deque(maxlen=history_len)
        proprio_hist = deque(maxlen=history_len)
        memory_bank.clear()
        memory_bank.queue = deque(maxlen=history_len)
        
        for fid, item in enumerate(tqdm(ep_data_list)):
            data = ep_data_list[fid]
            total_frames += 1
            image_save_path = os.path.join(saved_path, 'images', f'ep_{ep_idx:03d}-{frame_idx:06d}.png')
            image_path = os.path.join('images', f'ep_{ep_idx:03d}-{frame_idx:06d}.png')
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            image = data['observation.images.front'].to(torch.float32)
            image = (image * 255).to(torch.uint8)
            pil_img = TF.to_pil_image(image)
            
            if not os.path.exists(image_save_path):
                pil_img.save(image_save_path)
                
                if not is_zero_saved:
                    Image.fromarray(np.zeros_like(np.array(pil_img))).save(os.path.join(saved_path, 'images', "zero.jpg"))
                    is_zero_saved = True
                
            
            if memory_bank.get_length() == 0:
                for _ in range(history_len):
                    memory_bank.add(np.array(pil_img)[None], fid, real=True)
                    memory_bank.queue.append(fid)
                memory_bank.add(np.array(pil_img)[None], fid, real=True)
                memory_bank.queue.append(fid)
                image_path_hist.append(1)
            else:
                memory_bank.add(np.array(pil_img)[None], fid, real=True)
                memory_bank.queue.append(fid)

            # curr_pos = np.array(item["executed_ee_position"])
            # curr_quat = np.array(item["executed_ee_quaternion"])
            # curr_quat = [curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]]  # 转为 [x, y, z, w]
            
            # tar_pos = np.array(data[fid+1]["executed_ee_position"])
            # tar_quat = np.array(data[fid+1]["executed_ee_quaternion"])
            # tar_quat = [tar_quat[1], tar_quat[2], tar_quat[3], tar_quat[0]]  # 转为 [x, y, z, w]

            # delta_pos = (tar_pos - curr_pos).tolist()
            # delta_rot = delta_rot_quat(curr_quat, tar_quat)

            # if data[fid+1]["gripper_width"] > 0.07:
            #     gripper_width = 1
            # else:
            #     gripper_width = 0
                
            # action = delta_pos + delta_rot + [gripper_width]
            
            data = normalize_targets(data)
            action = data['action'][0]
            action = action.to(torch.float32).numpy().tolist()
            # action = [round(a, 5) for a in action]
            # intrinsic = [606.359375, 389.8868408203125, 320.4429626464844, 240.7040252685547, 640, 480]
            # fx, fy, cx, cy, w, h = intrinsic

            # raw_size = (w, h)
            # target_size = (448, 448)
            # scale_x = target_size[0] / raw_size[0]
            # scale_y = target_size[1] / raw_size[1]

            # agentview_intrinsic = [
            #     fx * scale_x, 0,         cx * scale_x,
            #     0,           fy * scale_y, cy * scale_y,
            #     0,           0,         1
            # ]
            # agentview_intrinsic = np.array(agentview_intrinsic).reshape(3, 3)
            # agentview_intrinsic = np.round(agentview_intrinsic, 4)
            
            # quat = [-0.6579468466651127, -0.6194541929018136, 0.28531243677131374, 0.31934192221412583]  # x, y, z, w
            # trans = [1.187292437489465, -0.09629429279576884, 0.964384707990467]

            # rot_mat = R.from_quat(quat).as_matrix()  # shape: [3, 3]

            # agentview_extrinsic = np.eye(4)
            # agentview_extrinsic[:3, :3] = rot_mat
            # agentview_extrinsic[:3, 3] = trans
            # agentview_extrinsic = np.round(agentview_extrinsic, 4)
            
            # agentview_extrinsic_list = agentview_extrinsic.tolist()
            # agentview_intrinsic_list = agentview_intrinsic.tolist()
            # lang = "move the yellow block to the plate"  # TODO: replace with actual language
            # lang = "put all the green cubes on the table into the plate"
            lang = data['task']
            entry = {
                "id": total_frames,
                "image": [],
                # "image_depth": [],
                "width_list": [],
                "height_list": [],
                # "width_depth_list": [],
                # "height_depth_list": [],
                "action": action,
                # "camera_info": {
                #     "agentview_extrinsic": agentview_extrinsic_list,
                #     "agentview_intrinsic": agentview_intrinsic_list,
                # },
                
                
                "conversations": [
                    {
                        "from": "human",
                        "value": "History:\n" + "<image>\n" * (history_len - 1) + f"Current:\n<image>\n What action should the robot take to {lang}?"
                    },
                    {
                        "from": "gpt",
                        "value": "<state_pred>"
                    }
                ]
            }

            fid_list = memory_bank.get_fid()[:-1]
            # fid_list = list(memory_bank.queue)[:-1]
            for f in fid_list:
                entry["image"].append(os.path.join('images', f'ep_{ep_idx:03d}-{f:06d}.png'))
                # entry["image_depth"].append(f"{data_folder}/images_processed/fid{f}_depth.npy")
                entry["width_list"].append(448)
                entry["height_list"].append(448)
                # entry["width_depth_list"].append(256)
                # entry["height_depth_list"].append(256)

            entry["image"].append(os.path.join('images', f'ep_{ep_idx:03d}-{fid:06d}.png'))
            # entry["image_depth"].append(os.path.join(data_folder, "images_processed", depth_name))
            entry["width_list"].append(448)
            entry["height_list"].append(448)
            # entry["width_depth_list"].append(256)
            # entry["height_depth_list"].append(256)

            json_data1.append(entry)
            
        # output_jsonl = os.path.join(saved_path, "real_robot_data_ws{}_bs{}_rgbd_mb.jsonl".format(window_size, bank_size))
        # with open(output_jsonl, "w") as f:
        #     for entry in json_data1:
        #         json.dump(entry, f)
        #         f.write("\n")
        # print(f"Saved {len(json_data1)} samples to {output_jsonl}")
        ep_data_list = [last_data]
    output_jsonl = os.path.join(saved_path, "real_robot_data_ws{}_bs{}_rgbd_mb.jsonl".format(window_size, bank_size))
    with open(output_jsonl, "w") as f:
        for entry in json_data1:
            json.dump(entry, f)
            f.write("\n")
    print(f"Final jsonl saved to {output_jsonl}")