import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from typing import Dict, List
from einops import rearrange
import mediapy as mp
from collections import deque

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from megatron.core import parallel_state

import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/world_model/ActionWorldModel")
from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedBatchPipeline
from imaginaire.constants import (
    CosmosPredict2ActionConditionedModelSize,
    get_cosmos_predict2_action_conditioned_checkpoint,
)
from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)
from imaginaire.utils import distributed, log, misc


class CosMosWorldModel(nn.Module):
    def __init__(self, model_config):
        """
        """
        super().__init__()
        
        print(f"--- [CosMosWorldModel] Initialized ---")
        print(f"--- PID: {os.getpid()} ---")
        print(f"--- Received model_config: {model_config} ---")
        
        super().__init__()
        self.config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = self.config.get("chunk_size", "20")
        
        print(f"--- [CosMosWorldModel] Initializing on PID {os.getpid()} ---")

        args = argparse.Namespace(
            model_size=self.config.get("model_size", "2B"),
            dit_path=self.config.get("dit_path", ""),
            load_ema=self.config.get("load_ema", False),
            seed=self.config.get("seed", 0),
            guidance=self.config.get("guidance", 7.0),
            num_gpus=1,
            disable_guardrail=True,
            disable_prompt_refiner=True,
        )

        resolution = '480'
        fps = 10
        pipeline_config = get_cosmos_predict2_action_conditioned_pipeline(model_size=args.model_size, resolution=resolution, fps=fps)
        vae_folder = self.config.get("vae_folder", "")
        pipeline_config.tokenizer.vae_pth = os.path.join(vae_folder, pipeline_config.tokenizer.vae_pth)
        pipeline_config.state_t = self.config.get("state_t", 6)
        pipeline_config.net.action_dim = self.config.get("action_dim", 14) * (self.config.get("pred_len", 21) - 1)
        pipeline_config.net.use_black = self.config.get("use_black", False)
        pipeline_config.net.use_history = self.config.get("use_history", False)
        self.history_video_length = self.config.get("history_video_length", 60)
        self.history_frams = deque(maxlen=self.history_video_length)
        
        if hasattr(args, "dit_path") and args.dit_path:
            dit_path = args.dit_path
        else:
            dit_path = get_cosmos_predict2_action_conditioned_checkpoint(
                model_size=args.model_size, resolution=resolution, fps=fps
            )
        
        misc.set_random_seed(seed=args.seed, by_rank=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"--- [CosMosWorldModel] Initializing Video2WorldPipeline... ---")
        self.pipe = Video2WorldActionConditionedBatchPipeline.from_config(
            config=pipeline_config,
            dit_path=dit_path,
            use_text_encoder=False,
            device=self.device,
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=args.load_ema,
            load_prompt_refiner=False,
        )
        self.pipe.eval() # 确保模型在推理模式
        
        print(f"--- [CosMosWorldModel] Initialization complete on device {self.device} ---")


    def load_state_dict(self, state_dict, strict=True):
        """
        """
        print(f"--- [CosMosWorldModel] load_state_dict called. Skipping actual loading. ---")
        pass

    @torch.no_grad()
    def step(self, current_obs_batch: np.ndarray, action_batch: np.ndarray, step) -> np.ndarray:
        """        
        Args:
            current_obs_batch (np.ndarray): 形状为 [B, H, W, 3]，uint8，范围 [0, 255]
            action_batch (np.ndarray): 形状为 [B, action_dim]

        Returns:
            np.ndarray: 下一步的批量观测，形状与输入相同 [B, H, W, 3]，uint8
        """
        # breakpoint()
        # self.history_frams.append()
        if self.use_history:
            history_obs_batch = np.array(self.history_frams)
            
        B = int(current_obs_batch.shape[0])

        # Standardize actions to [B, T, A]
        if action_batch.ndim == 2:
            actions = action_batch[:, None, :]          # [B, 1, A]
        elif action_batch.ndim == 3:
            actions = action_batch                      # [B, T, A]
        else:
            raise ValueError(f"action_batch must be [B, A] or [B, T, A], got {action_batch.shape}")

        T, A = int(actions.shape[1]), int(actions.shape[2])
        if T == 0:
            raise ValueError("Empty action sequence (T=0).")

        # --------- 1) Preprocess actions (match your current logic) ---------
        actions = actions.copy()
        # Normalize the last action dimension to [-1, +1], then sign and flip
        actions[..., -1] = 2.0 * (actions[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
        actions[..., -1] = np.sign(actions[..., -1]) * -1.0

        # Dual concat (keep your current behavior: concat zeros on the last dimension)
        dummy = np.zeros_like(actions)
        actions_dual = np.concatenate([actions, dummy], axis=-1)   # [B, T, 2A]

        # --------- 2) Chunking plan ---------
        max_len = int(self.config.get("wm_max_steps", 20))  # WM一次支持的最大帧数（默认20）
        if max_len <= 0:
            raise ValueError(f"wm_max_steps must be positive, got {max_len}")

        # First frames for the first chunk
        cur_first_frames: List[np.ndarray] = [frame for frame in current_obs_batch]  # B x (H,W,3), uint8
        if self.use_history:
            history_obs_list = [frame for frame in history_obs_batch]
        out_chunks: List[np.ndarray] = []

        # --------- 3) Loop over chunks ---------
        for s in range(0, T, max_len):
            e = min(s + max_len, T)                         # [s, e) chunk interval
            chunk_len = e - s
            chunk_actions_list: List[np.ndarray] = [actions_dual[b, s:e] for b in range(B)]  # per-batch seq

            chunk_actions_list: List[np.ndarray] = []
            for b in range(B):
                chunk = actions_dual[b, s:e]     # [chunk_len, 2A]
                if chunk_len < max_len:
                    pad = np.zeros(
                        (max_len - chunk_len, actions_dual.shape[-1]),
                        dtype=actions_dual.dtype
                    )
                    chunk = np.concatenate([chunk, pad], axis=0)  # [max_len, 2A]
                chunk_actions_list.append(chunk)
                
            # Run the WM pipeline for this chunk
            breakpoint()
            predicted_videos = self.pipe(
                first_frames=cur_first_frames,              # list of B images, H W 3
                actions_list=chunk_actions_list,            # list of B arrays, (chunk_len, 2A)
                blacks_list=None,
                num_conditional_frames=1,
                guidance=self.config.get("guidance", 7.0),
                seed=self.config.get("seed", 0),
                num_sampling_step=self.config.get("num_sampling_step", 10),
                use_cuda_graphs=self.config.get("use_cuda_graphs", False),
                history_list=history_obs_list if self.use_history else None,
            )

            # Convert the pipeline outputs to uint8 frames & drop the conditional frame at index 0
            # expected: each item in predicted_videos is like (video[C,F,H,W], ...); we take [0]
            # 
            video_tensors = [vid[0] for vid in predicted_videos]           # list of [C, F, H, W]
            batch = torch.stack(video_tensors, dim=0)                      # [B, C, F, H, W]
            batch = (batch / 2 + 0.5).clamp(0, 1)
            batch_hwc = rearrange(batch, 'b c f h w -> b f h w c').cpu().numpy()
            frames_uint8_full = (batch_hwc * 255).astype(np.uint8)[:, 1:]       # [B, chunk_len, H, W, 3]
            frames_uint8 = frames_uint8_full[:, :chunk_len]
            out_chunks.append(frames_uint8)

            # Next chunk starts from the last predicted frame of this chunk
            cur_first_frames = [frames_uint8[b, -1] for b in range(B)]
        # breakpoint()
        # --------- 4) Concatenate along time ---------
        out = np.concatenate(out_chunks, axis=1)   # [B, T, H, W, 3]
        # breakpoint()
        # self.save_video_grid(videos_batch_uint8, f'debug{step}.mp4')
        # self.save_trajectory_grid_image(videos_batch_uint8, 'debug.png')
        breakpoint()
        if self.use_history:
            for t in out.shape[1]:
                self.history_frams.append(out[:, t])
        return out  # B, chunk_size, H, W, C
    
    def reset(self, current_obs_batch_np):
        self.history_frames = deque(maxlen=self.history_video_length)
        for i in range(self.history_video_length):
            self.history_frames.append(current_obs_batch_np)
    
    # def init_env(self, init_state):
    #     breakpoint()
    #     if self.use_history:
    #         while len(self.history_frames) < self.history_video_length:
    #             self.history_frames.append(init_state)
    
    def save_video_grid(self, video_batch: np.ndarray, save_path: str, fps: int = 10):
        """
        将一个批量的视频数据 (B, F, H, W, C) 拼接成一个网格视频并保存。
        最终的视频将会有 F 帧，每一帧都是一个 B 行 1 列的图像网格。

        Args:
            video_batch (np.ndarray): 视频数据，形状为 (B, F, H, W, C)。
            save_path (str): 保存视频的路径。
            fps (int): 视频的帧率。
        """
        B, F, H, W, C = video_batch.shape

        # 1. 交换 Batch 和 Frame 维度，方便我们遍历每一帧
        # (B, F, H, W, C) -> (F, B, H, W, C)
        video_frames_first = video_batch.transpose(1, 0, 2, 3, 4)

        # 2. 为视频的每一帧创建一个网格图像
        grid_frames = []
        for i in range(F):
            # 获取当前时间步的所有 B 个样本帧
            frame_batch_at_t = video_frames_first[i]  # 形状: (B, H, W, C)
            
            # 使用 np.vstack 将这 B 张图片垂直堆叠起来
            # 这会创建一个 B 行 1 列的网格
            grid_frame = np.vstack(list(frame_batch_at_t)) # 形状: (B*H, W, C)
            grid_frames.append(grid_frame)

        # 3. 使用 mediapy 将这些网格帧保存为视频
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"正在将 {len(grid_frames)} 帧视频保存到: {save_path}")
        mp.write_video(save_path, grid_frames, fps=fps)
        print(f"网格视频成功保存。")

    def save_trajectory_grid_image(self, video_batch: np.ndarray, save_path: str):
        """
        将一个批量的视频数据 (B, F, H, W, C) 拼接成一个静态的网格大图并保存。
        最终的图片将会是 B 行 F 列的图像网格。

        Args:
            video_batch (np.ndarray): 视频数据，形状为 (B, F, H, W, C)。
            save_path (str): 保存图片的路径 (例如: "output/grid.png")。
        """
        B, F, H, W, C = video_batch.shape

        # 1. 将一个 batch 样本的所有 F 帧在水平方向拼接成一个长条
        #    `[np.hstack(video_batch[b]) for b in range(B)]` 会生成 B 个 (H, F*W, C) 的数组
        rows = [np.hstack(video_batch[b]) for b in range(B)]

        # 2. 将所有 B 个长条在垂直方向拼接成一个巨大的网格图
        grid_image = np.vstack(rows) # 形状: (B*H, F*W, C)
        
        # 3. 使用 mediapy 保存这张图片
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"正在将形状为 {grid_image.shape} 的网格图片保存到: {save_path}")
        mp.write_image(save_path, grid_image)
        print(f"网格图片成功保存。")
        
    def process_action(self, action_sequences):
        action_processed_list = []
        for old_action in action_sequences:
            old_action = old_action.reshape(-1, 2, 7)
            end_position = old_action[..., 0:3]
            end_rotation = old_action[..., 3:6]
            effector_position = old_action[..., -1]
            frame_num = end_position.shape[0]
            action = np.zeros((frame_num - 1, 14))
            for i in range(end_position.shape[1]):
                for k in range(1, frame_num):
                    prev_xyz = end_position[k - 1, i, 0:3]
                    prev_rpy = end_rotation[k - 1, i, 0:3]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = end_position[k, i, 0:3]
                    curr_rpy = end_rotation[k, i, 0:3]
                    curr_gripper = effector_position[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper * 0.03
            # action = np.concatenate([end_position[:, 0], end_rotation[:, 0], effector_position[..., None][:, 0], end_position[:, 1], end_rotation[:, 1], effector_position[..., None][:, 1]], -1)
            # import pdb; pdb.set_trace()
            action = action * np.array([20, 20, 20, 20, 20, 20, 1e-5, 20, 20, 20, 20, 20, 20, 1e-5])
            action = np.concatenate([action, action[-2:-1]], axis=0)
            action_processed_list.append(action)
        return action_processed_list
            