

import os
import random
from pathlib import Path
import packaging.version
import datasets
from datasets import concatenate_datasets, load_dataset
from PIL import Image
import numpy as np

import json

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

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
from lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from tqdm import tqdm
from torch.utils.data import DataLoader


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
                item['action'] = item['action'] # [:, 0]
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



# root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Spatial'
# saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Spatial-jsonl'

# root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Long'
# saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Long-jsonl'

# root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Goal'
# saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Goal-jsonl'

# root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Object'
# saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-Split/LIBERO-Object-jsonl'

# root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-4tasks/libero_4tasks'
# saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot-4tasks/libero_4tasks_jsonl'

root_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot/libero_full_lerobot'
saved_path = '/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot/libero_full_lerobot_jsonl'

policy_cfg = SmolVLAConfig()
ds_meta = LeRobotDatasetMetadata(
    {'root': root_path}
)
delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
dataset = LeRobotDataset(
    {'root': root_path},
    delta_timestamps=delta_timestamps,
    video_backend=get_safe_default_codec(),
)

data_list = []
len_dataset = dataset.__len__()

def save_data_item(data):
    """
    处理从DataLoader获取的单个数据项：保存图片并返回结构化数据。
    """
    # try:
    data = data['batch']
    ep_idx = int(data['episode_index'].item())
    frame_idx = int(data['frame_index'].item())

    image_filename = f'ep_{ep_idx:03d}-{frame_idx:06d}.png'
    image_save_path = os.path.join(saved_path, 'images', image_filename)
    image_relative_path = os.path.join('images', image_filename)
    
    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)

    image = data['observation.images.image'].to(torch.float32)
    image = (image * 255).to(torch.uint8)
    pil_img = TF.to_pil_image(image)
    pil_img.save(image_save_path)
    
    data_item = {
        "observation.images.image": [image_relative_path],
        "observation.images.image_is_pad": data['observation.images.image_is_pad'].to(torch.float32).numpy().tolist(),
        "observation.state": data['observation.state'].to(torch.float32).numpy().tolist(),
        "observation.state_is_pad": data['observation.state_is_pad'].to(torch.float32).numpy().tolist(),
        "action": data['action'].to(torch.float32).numpy().tolist(),
        "action_is_pad": data['action_is_pad'].to(torch.float32).numpy().tolist(),
        "task": data['task'],
    }
    return data_item
    # except Exception as e:
    #     print(f"处理数据时发生错误: {e}")
    #     return None

# 2. 配置并创建 DataLoader
# - num_workers: 开启的子进程数量，用于并行加载数据。通常设置为CPU核心数的 0.5 到 2 倍。
# - prefetch_factor: 每个worker预先加载的样本数。
# - pin_memory: 如果使用GPU，可以设置为True以加速数据到GPU的传输（这里不必要）。
data_loader = DataLoader(
    dataset,
    batch_size=None,  # 设置为 None, DataLoader 会返回单个样本而不是一个批次
    shuffle=False,
    num_workers=32,    # 根据你的CPU核心数调整
    prefetch_factor=4 # 建议设置
)


futures = []
data_list = []

# tqdm现在包裹的是DataLoader，显示数据加载的进度
for data_item in tqdm(data_loader, total=len(dataset)):
    item = save_data_item(data_item)
    data_list.append(item)


print(f"\n处理完成！共成功处理 {len(data_list)} 个项目。")


jsonl_path = os.path.join(saved_path, 'data.jsonl')
with open(jsonl_path, "w", encoding="utf-8") as f:
    for data_item in data_list:
        line = json.dumps(data_item, ensure_ascii=False)
        f.write(line + "\n")
print('Done! Saved to', saved_path)