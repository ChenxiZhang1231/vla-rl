# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import statistics
from functools import partial
import numpy as np
import cv2
import base64
import random
from pathlib import Path
from PIL import Image
from verl_vla.utils.rm_utils.evo_vlac import GAC_model_client
from swift.plugin import InferStats
from swift.llm.infer.protocol import RequestConfig
import copy

from typing import List, Tuple, Optional, Dict
import re
from collections import Counter
from dataclasses import dataclass
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from swift.llm import InferClient, InferRequest


import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/LIBERO")
from verl_vla import DataProto
import torch
# from verl_vla.utils.reward_score import gsm8k, math, countdown, multiply, logic
import math

from verl_vla.utils.prompt_utils.prompt import build_system_prompt
from verl_vla.trainer.ppo.ray_trainer import RayTrainer

class RobRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config
        if config.actor_rollout_ref.model.vla == 'smolvla':
            self.data_key = 'action_tensor'
        else:
            self.data_key = 'responses'

    def verify(self, data, global_steps=-1):
        completes = data.batch['complete'].tolist()
        batch_size = data.batch[self.data_key].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        return score, reward_metrics, format_metrics, reward_format_metrics

    def __call__raw(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32)
        # reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        # verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))
        reward_tensor = verifier_reward.reshape(reward_tensor.shape[0], -1, reward_tensor.shape[-1])
        verifier_reward = verifier_reward.reshape(verifier_reward.shape[0], -1, verifier_reward.shape[-1])
        
        # valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len
        B, S, K, CH, D = data.batch['x_t'].shape
        # s_fin  = (data.batch['finish_step'] // CH).view(B)
        # c_fin  = (data.batch['finish_step'] %  CH).view(B)
        valid_response_length = data.batch['finish_step']
       
        if 'acc' in data.batch:
            # the separated rewards have been logged; now we add format correctness back for reward shaping
            #verifier_score = data.batch['acc'].cpu().numpy().tolist() + (0.0 * data.batch['format_correctness'].cpu().numpy()).tolist()
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i, valid_response_length[i]-1, :] += verifier_score[i]
            
        reward_tensor_dict['gt_scores'] = verifier_reward  # .reshape(B, S, CH, 7)

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     raise  ValueError
        #     reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
        #     reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
        #     if self.config.reward_model.rm_coef!=0:
        #         reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0:
            # reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=(1, 2)).mean().item()
            reward_tensor = self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        # reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        # reward_tensor_dict['all'] = reward_tensor.reshape(B, S, CH, 7)
        
        # reward_tensor_dict['gt_scores'] = reward_tensor_dict['gt_scores'].reshape(B, S, CH, 7)
        reward_metrics['reward_all'] = reward_tensor.sum(dim=(-1, -2)).mean(dim=0).item()
        
        return reward_tensor_dict, reward_metrics
    
    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))
        
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len 

        
        # valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len
        # s_fin  = (data.batch['finish_step'] // CH).view(B)
        # c_fin  = (data.batch['finish_step'] %  CH).view(B)
        # valid_response_length = data.batch['finish_step']
       
        if 'acc' in data.batch:
            # the separated rewards have been logged; now we add format correctness back for reward shaping
            #verifier_score = data.batch['acc'].cpu().numpy().tolist() + (0.0 * data.batch['format_correctness'].cpu().numpy()).tolist()
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i,valid_response_length[i]-1] += verifier_score[i]
            
        reward_tensor_dict['gt_scores'] = verifier_reward

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     raise  ValueError
        #     reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
        #     reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
        #     if self.config.reward_model.rm_coef!=0:
        #         reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0:
            # reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=(1)).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=(-1)).mean(dim=0).item()
        
        return reward_tensor_dict, reward_metrics
    
from verl_vla.utils.ray_gate import RMGateway
def ensure_gateway():
    endpoints = [( "0.0.0.0", p) for p in range(8000, 8008)]
    gw = RMGateway.options(
        name="rm_gateway",        # 全局唯一名字
        lifetime="detached",      # Driver 退出也不被杀（方便复用）
        get_if_exists=True,       # 已存在则直接复用
    ).remote(
        endpoints=endpoints,
        max_batch_size=512,
        max_wait_ms=15,
        concurrency=8, 
        max_inflight=8,
        default_cfg={"temperature": 0.0, "max_tokens": 64, "stream": False},
        round_robin=True,
    )
    print("RMGateway started:", gw)


REF_DICT = {
    "libero_spatial": {
        0: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_0_trial_0--success=True--ran=9350.mp4",
        1: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_1_trial_5--success=True--ran=3448.mp4",
        2: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_2_trial_0--success=True--ran=4112.mp4",
        3: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_3_trial_0--success=True--ran=7681.mp4",
        4: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_4_trial_2--success=True--ran=6921.mp4",
        5: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_5_trial_0--success=True--ran=2259.mp4",
        6: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_6_trial_3--success=True--ran=6237.mp4",
        7: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_7_trial_16--success=True--ran=6198.mp4",
        8: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_8_trial_46--success=True--ran=6438.mp4",
        9: "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/rollouts/smolvla-bs32-n8-mb256-lr5e6-kl004-trainset/step=0--task=libero_spatial_task_9_trial_3--success=True--ran=5920.mp4",
    }
}
SUCCESS_VALUE_THRESH = 95.0      # 认为成功的 value 阈值
NO_PROGRESS_M = 500                # 连续 m 步 Δvalue <= 0 判定为无进展
BETA = 0.05                      # r_t = BETA * Δvalue 的缩放
CLIP_CRITIC_MIN = -90.0          # critic 数值裁剪下界，避免 -100 邻域不稳定
CLIP_CRITIC_MAX = 100.0          # critic 数值裁剪上界
CLIP_VALUE = True     

class RobVLACRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config
        self.vlm_input_num_frames = self.config.reward_model.vlm_input_num_frames
        self.return_env_score = self.config.reward_model.return_env_score
        self.use_world_model = config.actor_rollout_ref.world_model.dit_path != ""
        if config.actor_rollout_ref.model.vla == 'smolvla':
            self.data_key = 'action_tensor'
        else:
            self.data_key = 'responses'
        
        self.reward_model = GAC_model_client(tag='critic')
        self.reward_model.init_model(model_path=self.config.reward_model.model.path, model_type='internvl2', use_server=True, device_map=f'cuda:0')
        self.reward_model.temperature=0.5
        self.reward_model.top_k=1
        self.reward_model.set_config()
        self.reward_model.set_system_prompt()

    def verify_env(self, data):
        completes = data.batch['complete_raw'].tolist()
        batch_size = data.batch[self.data_key].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        return score, reward_metrics, format_metrics, reward_format_metrics

    def save_video_grid(
        self,
        video_tensor: torch.Tensor, 
        output_path: str, 
        fps: int = 10, 
        grid_size: tuple = None,
        padding: int = 5,
        pad_value: int = 0,
        scores=None,                 # 新增：模型/判别器分数（长度=B）
        scores_env=None,             # 新增：环境返回的分数（长度=B）
        score_text_fn=None,          # 可选：主分数文本格式化函数
        score_env_text_fn=None,      # 可选：环境分数文本格式化函数
        label_scores: str = "S",     # 可选：主分数标签
        label_scores_env: str = "E", # 可选：环境分数标签
        font_scale: float = None,    # 可选：字体大小（不传则按H自适应）
        thickness: int = None        # 可选：线宽（不传则按H自适应）
    ):
        """
        将一个批次的视频张量拼接成一个网格，并保存为视频文件，
        同时在每个小视频左上角叠加 score（支持 scores 与 scores_env 两行）。
        video_tensor: (B, T, H, W, C), uint8, RGB
        """
        # --- 1. 参数校验和准备 ---
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(video_tensor)}")
        if video_tensor.dim() != 5:
            raise ValueError(f"Input tensor must have 5 dimensions (B, T, H, W, C), but got {video_tensor.dim()}")
        if video_tensor.dtype != torch.uint8:
            print(f"Warning: dtype is {video_tensor.dtype}, converting to uint8. Values should be in [0,255].")
            video_tensor = video_tensor.byte()

        B, T, H, W, C = video_tensor.shape
        if C != 3:
            raise ValueError(f"Input tensor must have 3 channels (RGB), but got {C}")

        # 处理 scores
        def _to_list(x):
            if x is None: return None
            if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):   return x.tolist()
            return list(x)

        scores     = _to_list(scores)
        scores_env = _to_list(scores_env)

        if scores is not None and len(scores) != B:
            raise ValueError(f"len(scores)={len(scores)} != batch size B={B}")
        if scores_env is not None and len(scores_env) != B:
            raise ValueError(f"len(scores_env)={len(scores_env)} != batch size B={B}")

        # 自适应字体和线宽
        if font_scale is None:
            font_scale = max(0.4, H / 256.0)
        if thickness is None:
            thickness = max(1, int(round(H / 160.0)))

        # 文本函数
        if score_text_fn is None:
            score_text_fn = lambda s: f"{int(s)}" if isinstance(s, (int, bool)) or str(s) in ("0","1","True","False") else str(s)
        if score_env_text_fn is None:
            score_env_text_fn = lambda s: f"{int(s)}" if isinstance(s, (int, bool)) or str(s) in ("0","1","True","False") else str(s)

        # 颜色映射
        def score_to_color(val):
            if str(val) in ("1","True") or val == 1 or val is True:
                return (0, 200, 0)     # 绿（BGR）
            if str(val) in ("0","False") or val == 0 or val is False:
                return (0, 0, 220)     # 红
            return (220, 160, 0)       # 橙蓝

        # 张量转 numpy
        video_np = video_tensor.cpu().numpy()

        # --- 2. 自动计算网格尺寸 ---
        if grid_size is None:
            rows = int(math.sqrt(B)) or 1
            cols = (B + rows - 1) // rows
            grid_size = (rows, cols)
        rows, cols = grid_size
        if rows * cols < B:
            raise ValueError(f"Grid size {grid_size} is too small for batch size {B}")

        # --- 3. 计算最终视频帧尺寸 ---
        grid_h = rows * H + (rows - 1) * padding
        grid_w = cols * W + (cols - 1) * padding

        # --- 4. 初始化视频写入器 ---
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))
        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for path {output_path}")

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- 5. 逐帧处理并写入 ---
        for t in range(T):
            grid_frame = np.full((grid_h, grid_w, C), pad_value, dtype=np.uint8)
            frame_batch = video_np[:, t, :, :, :]  # (B, H, W, C), RGB

            for i in range(B):
                row_idx = i // cols
                col_idx = i % cols

                start_y = row_idx * (H + padding)
                start_x = col_idx * (W + padding)

                # 转 BGR 给 OpenCV
                frame_rgb = frame_batch[i]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                grid_frame[start_y:start_y+H, start_x:start_x+W] = frame_bgr

                # 叠加多行 score：第一行 S，第二行 E（如果提供）
                lines = []
                if scores is not None:
                    lines.append((f"{label_scores}:{score_text_fn(scores[i])}", score_to_color(scores[i])))
                if scores_env is not None:
                    lines.append((f"{label_scores_env}:{score_env_text_fn(scores_env[i])}", score_to_color(scores_env[i])))

                if lines:
                    # 计算整体文本框尺寸
                    line_gap = max(2, int(round(H / 200.0)))
                    pad = 4
                    text_sizes = [cv2.getTextSize(txt, font, font_scale, thickness)[0] for (txt, _) in lines]
                    tw = max(w for (w, h) in text_sizes)
                    th_total = sum(h for (w, h) in text_sizes) + (len(lines)-1)*line_gap

                    rect_x1 = start_x + 2
                    rect_y1 = start_y + 2
                    rect_x2 = min(start_x + 2 + tw + 2*pad, start_x + W - 1)
                    rect_y2 = min(start_y + 2 + th_total + 2*pad, start_y + H - 1)

                    # 半透明背景（统一深灰）
                    alpha = 0.5
                    roi = grid_frame[rect_y1:rect_y2, rect_x1:rect_x2].astype(np.float32)
                    bg_patch = np.full_like(roi, (30,30,30), dtype=np.float32)
                    blended = (alpha * bg_patch + (1 - alpha) * roi).astype(np.uint8)
                    grid_frame[rect_y1:rect_y2, rect_x1:rect_x2] = blended

                    # 逐行绘制文本与小圆点
                    cursor_y = rect_y1 + pad
                    dot_r = max(2, int(H / 64))
                    for (txt, color), (tw_i, th_i) in zip(lines, text_sizes):
                        text_x = rect_x1 + pad + 2*dot_r + 4   # 预留左侧圆点位置
                        text_y = cursor_y + th_i
                        # 左侧状态圆点
                        cy = cursor_y + th_i//2
                        cx = rect_x1 + pad + dot_r
                        cv2.circle(grid_frame, (cx, cy), dot_r, color, -1, lineType=cv2.LINE_AA)
                        # 文本白色
                        cv2.putText(grid_frame, txt, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                        cursor_y = text_y + line_gap

            video_writer.write(grid_frame)

        # --- 6. 释放资源 ---
        video_writer.release()
        print(f"Video saved successfully to {output_path}")
           
    def compute_confusion(self, y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    def compute_basic_metrics(self, cm: Dict[str, int]) -> Dict[str, float]:
        TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
        n = TP + FP + TN + FN
        acc = (TP + TN) / n if n > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        bal_acc = (recall + tnr) / 2.0
        import math
        denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0
        return {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tnr_specificity": tnr, "fnr": fnr,
            "balanced_accuracy": bal_acc, "mcc": mcc,
        }

    def compute_all_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
        cm = self.compute_confusion(y_true, y_pred)
        basic = self.compute_basic_metrics(cm)
        return {"confusion": cm, "metrics": basic}

    def process_video_to_PIL_frames_with_indices(self, video_path: Path, num_frames: int = 10) -> Tuple[List[str], List[int]]:
        """
        读取视频，均匀下采样到指定帧数，返回 (base64-URL 列表, 采样到的原始帧编号列表)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [], []

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        image_list = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                resized_frame = cv2.resize(frame, (128, 128))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                image_list.append(pil_image)
        cap.release()
        # 若读帧失败导致数量不一致，截齐
        L = min(len(image_list), len(frame_indices))
        return image_list[:L], frame_indices[:L]
    
    def _build_pairs_full_trajectory(self, task_records, task_descriptions, step_interval, ref_frames_list= None, ref_num=1):
        """
        从每个 env 的整段轨迹构造『抽样相邻帧对』用于一次性 RM 推理。
        约定：
        - rec['step_images'] 里：第0张是初始帧，之后每步末帧各1张；
            因此总帧数 F = T + 1，步数 T = F - 1
        - 以 step_interval 为步长做分段：
            段 j: [s, e)，其中 s = j*step_interval，e = min(s+step_interval, T)
            送入 RM 的帧对是 (frame[s], frame[e])
        返回：
        pairs: [{"img_t": .., "img_tp1": ..}, ...]
        pair_task_texts: 与 pairs 对齐的任务文本
        pair_map: [(env_idx, start_step, seg_len), ...]
        """
        pairs = []
        pair_task_texts = []
        pair_map = []  # (env_idx, start_step, seg_len)

        for env_idx, rec in enumerate(task_records):
            imgs = rec
            if len(imgs) < 2:
                continue

            T = len(imgs) - 1  # 步数
            if T <= 0:
                continue

            task_text = task_descriptions[env_idx]
            
            def _even_subsample(lst, k: int):
                if not lst or k <= 0:
                    return []
                if len(lst) <= k:
                    return list(lst)
                # 从 [0, len-1] 等距取 k 个索引
                step = (len(lst) - 1) / (k - 1)
                return [lst[int(round(i * step))] for i in range(k)]
            
            s = 0
            while s < T:
                e = min(s + step_interval, T)
                # 帧对：起点帧=imgs[s]，终点帧=imgs[e]
                ref_seq = ref_frames_list[env_idx]
                refs = _even_subsample(ref_seq, ref_num)
                pairs.append({"img_t": imgs[s], "img_tp1": imgs[e], "img_0": imgs[0], "img_ref": refs})
                pair_task_texts.append(task_text)
                pair_map.append((env_idx, s, e - s))  # 段长 seg_len = e-s
                s = e

        return pairs, pair_task_texts, pair_map
    
    def _distribute_and_finalize_rewards(
        self,
        step_images,              # np.ndarray: (B, T+1, H, W, C)
        pair_map,                 # List[Tuple[int, int, int]]: (env_idx, start, seg_len)
        critics,                  # np.ndarray: (B, S_i)  每个env的段级critic序列
        *,
        beta: float = 0.05,
        success_window: int = 2,
        success_thresh: float = 95.0,
        max_steps: int | None = None,
    ):
        """
        返回：
        - total_critic:  List[List[float]]   每env的 per-step critic，长度=T
        - total_delta:   List[List[float]]   每env Δvalue，长度=T
        - total_value:   List[List[float]]   每env value，长度=T+1
        - total_reward:  List[List[float]]   每env reward=beta*Δvalue，长度=T
        - total_active:  List[bool]          统一 False（事后评估完成）
        - total_complete:List[bool]          是否命中 success_window
        - total_finish_step: List[int]       成功则首次命中步；否则 min(T, max_steps)
        """
        import numpy as np

        assert isinstance(step_images, np.ndarray) and step_images.ndim >= 3, "step_images 形状应为 (B, T+1, ...)"
        B = step_images.shape[0]
        T = max(step_images.shape[1] - 1, 0)
        if max_steps is None:
            max_steps = T

        # 1) 初始化 per-env 容器 & 段计数器
        per_env_step_critic = [[0.0] * T for _ in range(B)]
        seg_ptr = [0] * B  # 指向 critics[env_idx] 的当前段号

        # 2) 段级 critic 均匀摊回每步
        #    对于每个 (env_idx, start, seg_len)，使用 critics[env_idx, seg_ptr[env_idx]]，然后 seg_ptr+1
        for (env_idx, start, seg_len) in pair_map:
            if not (0 <= env_idx < B):
                continue
            if seg_len is None or seg_len <= 0:
                # 无效段，跳过并推进指针（若你希望严格对齐，可不推进）
                if seg_ptr[env_idx] < critics.shape[1]:
                    seg_ptr[env_idx] += 1
                continue

            if seg_ptr[env_idx] >= critics.shape[1]:
                # 当前 env 的段级分数不足；跳过
                continue
            c = float(critics[env_idx, seg_ptr[env_idx]])
            seg_ptr[env_idx] += 1

            contrib = c / float(seg_len)
            step_list = per_env_step_critic[env_idx]
            # 把贡献均匀加到 start..start+seg_len-1
            for t in range(start, start + seg_len):
                if 0 <= t < T:
                    step_list[t] += contrib

        # 3) 累计每个 env 的 value/Δvalue，并计算 reward/完成度
        total_critic, total_delta, total_value = [], [], []
        total_reward, total_active, total_complete, total_finish_step = [], [], [], []

        for env_idx in range(B):
            step_critic = per_env_step_critic[env_idx]
            if T == 0:
                # 空轨迹
                total_critic.append([])
                total_delta.append([])
                total_value.append([0.0])
                total_reward.append([])
                total_active.append(False)
                total_complete.append(False)
                total_finish_step.append(0)
                continue

            # 用你的累计函数：value 长度 T+1，deltas 长度 T
            value, deltas = self._accumulate_value_delta(step_critic, v0=0.0)
            rewards = [beta * d for d in deltas]

            # 成功判定（连续 success_window 个 value >= success_thresh）
            finish_step = None
            consec = 0
            for i in range(1, T + 1):  # value[i] 对应“完成第 i 步后的进度”
                if value[i] >= success_thresh:
                    consec += 1
                else:
                    consec = 0
                if consec >= success_window:
                    finish_step = i  # 首次满足窗口的步
                    break

            if finish_step is not None:
                complete = True
            else:
                complete = False
                finish_step = min(T, max_steps)

            # 统一置 inactive（事后评估）
            active = False

            total_critic.append(step_critic)
            total_delta.append(deltas)
            total_value.append(value)
            total_reward.append(rewards)
            total_active.append(active)
            total_complete.append(complete)
            total_finish_step.append(int(finish_step))

        return (total_critic,
                total_delta,
                total_value,
                total_reward,
                total_active,
                total_complete,
                total_finish_step)
            
    def _accumulate_value_delta(self, critic_seq, v0=0.0, eps: float = 1e-3):
        """
        累计 critic → value（0..100）并给出 Δvalue。
        - 正向：v_next = v + (100 - v) * (c/100)
        - 负向：v_next = 100 - (100 - v) * 100/(100 + c)   # c<0
        （移除 d 的 1.0 下限，或改为极小 eps，避免临门一脚“巨幅下坠”）
        """
        value = [float(v0)]
        deltas = []
        for c in critic_seq:
            c = float(max(min(c, CLIP_CRITIC_MAX), CLIP_CRITIC_MIN))  # 例如 [-90, 100]
            v = value[-1]
            if c >= 0:
                v_next = v + (100.0 - v) * (c / 100.0)
            else:
                gap = max(100.0 - v, eps)   # 以前是 max(..., 1.0) —— 改为极小 eps
                v_next = 100.0 - gap * (100.0 / (100.0 + c))
            if CLIP_VALUE:
                v_next = min(max(v_next, 0.0), 100.0)
            value.append(v_next)
            deltas.append(v_next - v)
        return value, deltas

    def _accumulate_value_delta_symmetric(self, critic_seq, v0=0.0):
        value = [float(v0)]
        deltas = []
        
        for c in critic_seq:
            c = float(max(min(c, CLIP_CRITIC_MAX), CLIP_CRITIC_MIN))
            v = value[-1]
            if c >= 0:
                v_next = v + (100.0 - v) * (c / 100.0)
            else:
                v_next = v + v * (c / 100.0)   # 朝 0 退，幅度与 v 成正比
            if CLIP_VALUE:
                v_next = min(max(v_next, 0.0), 100.0)
            value.append(v_next)
            deltas.append(v_next - v)
        return value, deltas
    
    def _round_robin_shards(self, n_items: int, n_shards: int):
        shards = [[] for _ in range(n_shards)]
        for i in range(n_items):
            shards[i % n_shards].append(i)
        return shards
    
    def verify(self, data, global_steps=-1):
        # breakpoint()
        if self.return_env_score and ('complete' in data.batch):
            score_env, reward_metrics_env, format_metrics_env, reward_format_metrics_env = self.verify_env(data)
        else:
            scores_env = None
        step_images = data.batch["step_images"].cpu().numpy()
        step_images_mask = data.batch["step_images_mask"]
        task_lang = data.non_tensor_batch['task_lang']
        task_suite_name = data.non_tensor_batch['task_suite_name']
        task_id = list(data.batch['task_id'])
        B, N, H, W, C = step_images.shape
        
        step_interval = self.config.reward_model.rm_step_interval
        ref_video_path_list = [Path(REF_DICT[task_suite_name[i]][task_id[i].item()]) for i in range(len(task_id))]
        ref_frames_list = []
        for ref_video_path in ref_video_path_list:
            ref_frames, ref_frame_indices = self.process_video_to_PIL_frames_with_indices(ref_video_path, num_frames=10)
            ref_frames_list.append(ref_frames)
        pairs, pair_task_texts, pair_map = self._build_pairs_full_trajectory(
            step_images, task_lang, step_interval=step_interval, ref_frames_list=ref_frames_list, ref_num=10
        )
        
        endpoints = [( "0.0.0.0", p) for p in range(8000, 8008)]
        num_eps = len(endpoints)

        shard_indices = self._round_robin_shards(len(pairs), num_eps)
        def _run_shard(ep_id: int, idxs: list[int]):
            """在线程内：克隆 RM、指定 engine、跑 reward_step，返回 (idxs, critics)"""
            if not idxs:
                return [], []
            rm_local = copy.copy(self.reward_model)     # 浅拷贝：共享只读配置，替换 engine
            host, port = endpoints[ep_id]
            rm_local.engine = InferClient(host=host, port=port)
            rm_local.infer_stats = InferStats()
            # 如果你在 RM 里有 request_config，可复用；否则给个默认
            if not hasattr(rm_local, "request_config") or rm_local.request_config is None:
                rm_local.request_config = RequestConfig(max_tokens=256, temperature=0.0, stream=False)

            sub_pairs = [pairs[i] for i in idxs]
            sub_tasks = [pair_task_texts[i] for i in idxs]

            critics = rm_local.reward_step(
                sub_pairs, sub_tasks,
                use_ref=True,           # 你现在用 ref
                batch_num=256,          # 每端点子批；可按显存调整 128~512
                addition_scale=1.0,
                divide_skip=1,
                related_critic=False,
                return_value=False,
                rich=False,
            )
            return idxs, [float(c) for c in critics]

        critic_chunks = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=num_eps) as ex:
            futs = [ex.submit(_run_shard, ep, idxs) for ep, idxs in enumerate(shard_indices) if idxs]
            for fut in as_completed(futs):
                idxs, critics = fut.result()
                for j, i in enumerate(idxs):
                    critic_chunks[i] = critics[j]
        
        
        # self.reward_model.engine = InferClient(host='0.0.0.0', port=8000)
        # critic_chunks = self.reward_model.reward_step(
        #     pairs, pair_task_texts,
        #     use_ref=True,
        #     batch_num=512,
        #     addition_scale=1.0,
        #     divide_skip=1,
        #     related_critic=False,
        #     return_value=False,
        #     rich=False,
        # )
        # breakpoint()
        critics = np.array(critic_chunks).reshape(B, -1)

        (total_critic, 
         total_delta, 
         total_value, 
         total_reward, 
         total_active, 
         total_complete, 
         total_finish_step
        ) = self._distribute_and_finalize_rewards(
            step_images, pair_map, critics, beta=0.05, success_window=2, max_steps=N
        )
        # breakpoint()
        
        pred_success = list(map(int, total_complete))
        scores = pred_success
        finish_step = torch.tensor(total_finish_step, device=data.batch[self.data_key].device)
        complete = total_complete
        
        if self.return_env_score:
            cls_stat = self.compute_all_metrics(score_env, pred_success)
        if self.use_world_model and (global_steps != -1):
            # breakpoint()
            ran_id = random.randint(1, 10000)
            save_path = f"work_dirs/{self.config.actor_rollout_ref.rollout.experiment_name}_train_rollouts/{global_steps}_rand{ran_id}.mp4"
            self.save_video_grid(
                step_images, save_path, fps=10,
                scores=scores,
                scores_env=scores_env,
                # score_text_fn=lambda s: f"{int(s)}",
                # score_env_text_fn=lambda s: f"{int(s)}",
                label_scores="S",
                label_scores_env="E",)
    
        format = [1.0 for _ in range(len(scores))]

        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['dense_reward'] = torch.tensor(total_reward, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['complete'] = complete
        data.batch['finish_step'] = finish_step
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        if self.return_env_score:
            reward_metrics['rm'] = cls_stat
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()
        return scores, reward_metrics, format_metrics, reward_format_metrics
        
    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors
        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32) # batch * 64 * 56
        reward_tensor_dense = torch.zeros_like(reward_tensor, dtype=torch.float32)
        verifier_reward = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        reward_tensor_dense = reward_tensor_dense.reshape((reward_tensor_dense.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))
        verifier_dense_reward = torch.zeros_like(verifier_reward, dtype=torch.float32)
        # breakpoint()
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len 
       
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
            verifier_dense_score = data.batch['dense_reward'].cpu().numpy()
            verifier_dense_score = np.repeat(verifier_dense_score[..., None], self.config.actor_rollout_ref.model.action_token_len, axis=-1)
            verifier_dense_score = verifier_dense_score.reshape(len(verifier_score), -1)
            verifier_dense_score = torch.from_numpy(verifier_dense_score)
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i,valid_response_length[i]-1] += verifier_score[i]
            verifier_dense_reward[i, :valid_response_length[i]-1] = verifier_dense_score[i, :valid_response_length[i]-1]
        reward_tensor_dict['gt_scores'] = verifier_reward
        reward_tensor_dict['gt_dense_scores'] = verifier_dense_reward

        if self.config.verifier.reward_coef!=0:
            # reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=(1)).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']
            reward_tensor_dense += self.config.verifier.reward_coef * reward_tensor_dict['gt_dense_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_tensor_dict['all_dense'] = reward_tensor_dense
        reward_metrics['reward_all'] = reward_tensor.sum(dim=(-1)).mean(dim=0).item()
        
        return reward_tensor_dict, reward_metrics

@dataclass
class RewardTask:
    frames: List[str]
    frame_indices: List[int]
    description: str

PROMPT = """You are a task-conditioned video rollout success judge.

You are given an ordered sequence of frames from a policy rollout video.
Your job is to decide (1) whether the task is successfully completed,
and (2) at which step index (from the provided step_id list) the success is FIRST
visibly satisfied.

Principles
- Use only the provided frames. Do not assume off-camera facts.
- Success requires visible, decisive evidence in-frame.
- Do NOT infer “about to succeed” (hovering ≠ ON/IN).
- If a required condition cannot be verified from the frames, choose Failure.
- The reported finish_step must be one of the provided step_ids; if Failure, use -1.

Required Output (JSON only; no extra text):
{"success": 0 or 1, "finish_step": <int>}
"""

class RobVLMRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config
        self.vlm_input_num_frames = self.config.reward_model.vlm_input_num_frames
        self.return_env_score = self.config.reward_model.return_env_score
        self.use_world_model = config.actor_rollout_ref.world_model.dit_path != ""
        if config.actor_rollout_ref.model.vla == 'smolvla':
            self.data_key = 'action_tensor'
        else:
            self.data_key = 'responses'
        self.client = openai.OpenAI(
            base_url="http://localhost:18901/v1",
            api_key="not-needed"
        )
        self.vote_n = self.config.reward_model.vote_n
        self.vote_m = self.config.reward_model.vote_m
        self.temperature = self.config.reward_model.temperature
        self.top_p = self.config.reward_model.top_p

    def verify_env(self, data):
        completes = data.batch['complete_raw'].tolist()
        batch_size = data.batch[self.data_key].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        return score, reward_metrics, format_metrics, reward_format_metrics

    def get_rewards_from_judge_batch_sync(
        self,
        client,
        tasks: List[RewardTask],
        max_workers: int = 10,
        temperature: float = 0.0, 
        top_p: float = 1.0,
        seeds: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[str]]:
        """
        返回:
        pred_success_list: 与 tasks 对应的 0/1 列表
        pred_finish_steps: 与 tasks 对应的 int（或 -1）
        raw_texts:         与 tasks 对应的模型原始输出
        """
        pred_success = [0] * len(tasks)
        pred_finish = [-1] * len(tasks)
        raw_texts = [""] * len(tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2idx = {
                ex.submit(
                    self.fetch_one_reward_sync, client, task, i,
                    temperature, top_p,
                    None if seeds is None else seeds[i],): i
                for i, task in enumerate(tasks)
            }
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                try:
                    _, s, f, t = fut.result()
                    pred_success[idx] = s
                    pred_finish[idx] = f
                    raw_texts[idx] = t
                except Exception as e:
                    print(f"[ERR] Task {idx} future error: {e}")

        return pred_success, pred_finish, raw_texts

    def fetch_one_reward_sync(self, client, task: RewardTask, task_index: int, temperature, top_p, seed) -> Tuple[int, int, int, str]:
        """
        对单个样本调用 judge。
        返回: (task_index, pred_success (0/1), pred_finish_step (int or -1), raw_text)
        """
        if not task.frames:
            return task_index, 0, -1, "Empty"

        step_ids = task.frame_indices[:len(task.frames)]
        question = self.build_question(task.description, step_ids=step_ids)
        user_content = [{"type": "text", "text": question}]
        for frame_url in task.frames:
            user_content.append({"type": "image_url", "image_url": {"url": frame_url}})

        try:
            completion = client.chat.completions.create(
                model="judge",
                messages=[
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                temperature=temperature,
                top_p=top_p,
            )
            resp = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[ERR] Task {task_index} API error: {e}")
            return task_index, 0, -1, f"APIError: {e}"

        data = self.try_parse_json_response(resp)
        if data is None:
            return task_index, 0, -1, resp

        succ = 1 if int(data.get("success", 0)) == 1 else 0
        fs_raw = int(data.get("finish_step", -1))
        fs_mapped = self.map_finish_step_to_sampled_idx(fs_raw, step_ids) if succ == 1 else -1
        return task_index, succ, fs_mapped, resp

    def try_parse_json_response(self, text: str) -> Optional[Dict]:
        """
        从模型输出中尽可能提取 JSON（容错）
        """
        if not text:
            return None
        # 直接尝试整体解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 截取第一个 '{' 到最后一个 '}' 之间
        try:
            l = text.find("{")
            r = text.rfind("}")
            if 0 <= l < r:
                return json.loads(text[l:r+1])
        except Exception:
            return None
        return None

    def map_finish_step_to_sampled_idx(self, finish_step_raw: int, sampled_indices: List[int]) -> int:
        """
        将原视频的完成步编号映射到采样后的 step_id（就地 frame index）。
        若 sampled_indices 为空，或 finish_step_raw 无效，返回 -1。
        若模型/GT 给了不在集合内的值，映射到最近的 step_id。
        """
        if finish_step_raw is None or finish_step_raw < 0 or len(sampled_indices) == 0:
            return -1
        # 最近邻
        arr = np.asarray(sampled_indices)
        j = int(np.argmin(np.abs(arr - finish_step_raw)))
        return int(arr[j])

    def get_reward_tasks(self, step_images, step_images_mask, task_lang):
        B, L, H, W, C = step_images.shape
        tasks_to_process = []
        for i in range(B):
            boolean_mask = step_images_mask[i].bool() 
            valid_images = step_images[i][boolean_mask].cpu().numpy()
            total_frames, H, W, C = valid_images.shape
            frame_indices = np.linspace(0, total_frames - 1, self.vlm_input_num_frames, dtype=int)
            base64_frames = []
            for idx in frame_indices:
                frame = valid_images[idx]
                resized_frame = cv2.resize(frame, (128, 128))
                _, buffer = cv2.imencode('.jpg', resized_frame)
                base64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(f"data:image/jpeg;base64,{base64_str}")
            reward_task = RewardTask(frames=base64_frames, frame_indices=frame_indices, description=task_lang[i])
            tasks_to_process.append(reward_task)
        return tasks_to_process
     
    def build_question(self, task_lang: str, step_ids: list[int]) -> str:
        """
        生成与你 SFT 一致的 user 文本：先 PROMPT，然后 Task，再逐行
        'frame_step{step_id}-<image>'，不使用 system prompt。
        注意：这里的 step_ids 必须与后续附加的图片顺序一一对应。
        """
        # 如果视频解码有丢帧，务必用 len(frames) 截齐 step_ids，以保证一一对应
        frame_str = "".join([f"frame_step{sid}-<image>\n" for sid in step_ids])
        return f"{PROMPT}\nTask: {task_lang}\n{frame_str}"

    def try_parse_json_response(self, text: str) -> Optional[Dict]:
        """
        从模型输出中尽可能提取 JSON（容错）
        """
        if not text:
            return None
        # 直接尝试整体解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 截取第一个 '{' 到最后一个 '}' 之间
        try:
            l = text.find("{")
            r = text.rfind("}")
            if 0 <= l < r:
                return json.loads(text[l:r+1])
        except Exception:
            return None
        return None
       
    def save_video_grid(
        self,
        video_tensor: torch.Tensor, 
        output_path: str, 
        fps: int = 10, 
        grid_size: tuple = None,
        padding: int = 5,
        pad_value: int = 0,
        scores=None,                 # 新增：模型/判别器分数（长度=B）
        scores_env=None,             # 新增：环境返回的分数（长度=B）
        score_text_fn=None,          # 可选：主分数文本格式化函数
        score_env_text_fn=None,      # 可选：环境分数文本格式化函数
        label_scores: str = "S",     # 可选：主分数标签
        label_scores_env: str = "E", # 可选：环境分数标签
        font_scale: float = None,    # 可选：字体大小（不传则按H自适应）
        thickness: int = None        # 可选：线宽（不传则按H自适应）
    ):
        """
        将一个批次的视频张量拼接成一个网格，并保存为视频文件，
        同时在每个小视频左上角叠加 score（支持 scores 与 scores_env 两行）。
        video_tensor: (B, T, H, W, C), uint8, RGB
        """
        # --- 1. 参数校验和准备 ---
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(video_tensor)}")
        if video_tensor.dim() != 5:
            raise ValueError(f"Input tensor must have 5 dimensions (B, T, H, W, C), but got {video_tensor.dim()}")
        if video_tensor.dtype != torch.uint8:
            print(f"Warning: dtype is {video_tensor.dtype}, converting to uint8. Values should be in [0,255].")
            video_tensor = video_tensor.byte()

        B, T, H, W, C = video_tensor.shape
        if C != 3:
            raise ValueError(f"Input tensor must have 3 channels (RGB), but got {C}")

        # 处理 scores
        def _to_list(x):
            if x is None: return None
            if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):   return x.tolist()
            return list(x)

        scores     = _to_list(scores)
        scores_env = _to_list(scores_env)

        if scores is not None and len(scores) != B:
            raise ValueError(f"len(scores)={len(scores)} != batch size B={B}")
        if scores_env is not None and len(scores_env) != B:
            raise ValueError(f"len(scores_env)={len(scores_env)} != batch size B={B}")

        # 自适应字体和线宽
        if font_scale is None:
            font_scale = max(0.4, H / 256.0)
        if thickness is None:
            thickness = max(1, int(round(H / 160.0)))

        # 文本函数
        if score_text_fn is None:
            score_text_fn = lambda s: f"{int(s)}" if isinstance(s, (int, bool)) or str(s) in ("0","1","True","False") else str(s)
        if score_env_text_fn is None:
            score_env_text_fn = lambda s: f"{int(s)}" if isinstance(s, (int, bool)) or str(s) in ("0","1","True","False") else str(s)

        # 颜色映射
        def score_to_color(val):
            if str(val) in ("1","True") or val == 1 or val is True:
                return (0, 200, 0)     # 绿（BGR）
            if str(val) in ("0","False") or val == 0 or val is False:
                return (0, 0, 220)     # 红
            return (220, 160, 0)       # 橙蓝

        # 张量转 numpy
        video_np = video_tensor.cpu().numpy()

        # --- 2. 自动计算网格尺寸 ---
        if grid_size is None:
            rows = int(math.sqrt(B)) or 1
            cols = (B + rows - 1) // rows
            grid_size = (rows, cols)
        rows, cols = grid_size
        if rows * cols < B:
            raise ValueError(f"Grid size {grid_size} is too small for batch size {B}")

        # --- 3. 计算最终视频帧尺寸 ---
        grid_h = rows * H + (rows - 1) * padding
        grid_w = cols * W + (cols - 1) * padding

        # --- 4. 初始化视频写入器 ---
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))
        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for path {output_path}")

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- 5. 逐帧处理并写入 ---
        for t in range(T):
            grid_frame = np.full((grid_h, grid_w, C), pad_value, dtype=np.uint8)
            frame_batch = video_np[:, t, :, :, :]  # (B, H, W, C), RGB

            for i in range(B):
                row_idx = i // cols
                col_idx = i % cols

                start_y = row_idx * (H + padding)
                start_x = col_idx * (W + padding)

                # 转 BGR 给 OpenCV
                frame_rgb = frame_batch[i]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                grid_frame[start_y:start_y+H, start_x:start_x+W] = frame_bgr

                # 叠加多行 score：第一行 S，第二行 E（如果提供）
                lines = []
                if scores is not None:
                    lines.append((f"{label_scores}:{score_text_fn(scores[i])}", score_to_color(scores[i])))
                if scores_env is not None:
                    lines.append((f"{label_scores_env}:{score_env_text_fn(scores_env[i])}", score_to_color(scores_env[i])))

                if lines:
                    # 计算整体文本框尺寸
                    line_gap = max(2, int(round(H / 200.0)))
                    pad = 4
                    text_sizes = [cv2.getTextSize(txt, font, font_scale, thickness)[0] for (txt, _) in lines]
                    tw = max(w for (w, h) in text_sizes)
                    th_total = sum(h for (w, h) in text_sizes) + (len(lines)-1)*line_gap

                    rect_x1 = start_x + 2
                    rect_y1 = start_y + 2
                    rect_x2 = min(start_x + 2 + tw + 2*pad, start_x + W - 1)
                    rect_y2 = min(start_y + 2 + th_total + 2*pad, start_y + H - 1)

                    # 半透明背景（统一深灰）
                    alpha = 0.5
                    roi = grid_frame[rect_y1:rect_y2, rect_x1:rect_x2].astype(np.float32)
                    bg_patch = np.full_like(roi, (30,30,30), dtype=np.float32)
                    blended = (alpha * bg_patch + (1 - alpha) * roi).astype(np.uint8)
                    grid_frame[rect_y1:rect_y2, rect_x1:rect_x2] = blended

                    # 逐行绘制文本与小圆点
                    cursor_y = rect_y1 + pad
                    dot_r = max(2, int(H / 64))
                    for (txt, color), (tw_i, th_i) in zip(lines, text_sizes):
                        text_x = rect_x1 + pad + 2*dot_r + 4   # 预留左侧圆点位置
                        text_y = cursor_y + th_i
                        # 左侧状态圆点
                        cy = cursor_y + th_i//2
                        cx = rect_x1 + pad + dot_r
                        cv2.circle(grid_frame, (cx, cy), dot_r, color, -1, lineType=cv2.LINE_AA)
                        # 文本白色
                        cv2.putText(grid_frame, txt, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                        cursor_y = text_y + line_gap

            video_writer.write(grid_frame)

        # --- 6. 释放资源 ---
        video_writer.release()
        print(f"Video saved successfully to {output_path}")
        
    def aggregate_success_and_finish(self, pred_success_list: List[int], pred_finish_list: List[int], m: int, strategy: str="min") -> Tuple[int,int]:
        """
        m-of-n 聚合：
        - 若 >= m 票成功 → 最终成功=1，否则 0
        - finish_step 仅在成功时聚合成功票的 step
        strategy: "min" | "median" | "mode"
        """
        success_votes = [i for i, s in enumerate(pred_success_list) if s == 1]
        if len(success_votes) >= m:
            finishes = [pred_finish_list[i] for i in success_votes if pred_finish_list[i] is not None and pred_finish_list[i] >= 0]
            if not finishes:
                return 1, -1
            if strategy == "min":
                return 1, int(min(finishes))          # 最保守：取最早完成
            elif strategy == "median":
                return 1, int(np.median(finishes))
            elif strategy == "mode":
                c = Counter(finishes).most_common(1)[0][0]
                return 1, int(c)
            else:
                return 1, int(min(finishes))
        else:
            return 0, -1
        
    def compute_confusion(self, y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    def compute_basic_metrics(self, cm: Dict[str, int]) -> Dict[str, float]:
        TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
        n = TP + FP + TN + FN
        acc = (TP + TN) / n if n > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        bal_acc = (recall + tnr) / 2.0
        import math
        denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0
        return {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tnr_specificity": tnr, "fnr": fnr,
            "balanced_accuracy": bal_acc, "mcc": mcc,
        }

    def compute_all_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
        cm = self.compute_confusion(y_true, y_pred)
        basic = self.compute_basic_metrics(cm)
        return {"confusion": cm, "metrics": basic}

    def verify(self, data, global_steps=-1):
        if self.return_env_score and ('complete' in data.batch):
            score_env, reward_metrics_env, format_metrics_env, reward_format_metrics_env = self.verify_env(data)
        else:
            scores_env = None
        step_images = data.batch["step_images"]
        step_images_mask = data.batch["step_images_mask"]
        task_lang = data.non_tensor_batch['task_lang']
        B, N, H, W, C = step_images.shape
        reward_tasks = self.get_reward_tasks(step_images, step_images_mask, task_lang)
        
        if self.vote_n <= 1:
            pred_success, pred_finish, raw_texts = self.get_rewards_from_judge_batch_sync(
                self.client, reward_tasks, max_workers=64
            )
        else:
            flat_tasks = [t for t in reward_tasks for _ in range(self.vote_n)]
            seeds = None
            v_succ, v_finish, v_text = self.get_rewards_from_judge_batch_sync(
                self.client, flat_tasks, max_workers=64,
                temperature=self.temperature, top_p=self.top_p, seeds=seeds
            )

            ptr = 0
            pred_success, pred_finish = [], []
            for i, t in enumerate(reward_tasks):
                seg_succ  = v_succ[ptr:ptr + self.vote_n]
                seg_finish = v_finish[ptr:ptr + self.vote_n]
                seg_text   = v_text[ptr:ptr + self.vote_n]
                ptr += self.vote_n

                agg_s, agg_f = self.aggregate_success_and_finish(seg_succ, seg_finish, m=self.vote_m, strategy="min")
                pred_success.append(int(agg_s))
                pred_finish.append(int(agg_f))
        
        scores = pred_success
        finish_step = torch.tensor([p if p != -1 else N - 1 for p in pred_finish], device=data.batch[self.data_key].device)
        # scores = score_env
        # finish_step = data.batch["finish_step_raw"]
        complete = (torch.tensor(scores) == 1).to(device=data.batch[self.data_key].device)
        if self.return_env_score:
            cls_stat = self.compute_all_metrics(score_env, pred_success)
        if self.use_world_model and (global_steps != -1):
            # breakpoint()
            ran_id = random.randint(1, 10000)
            save_path = f"work_dirs/{self.config.actor_rollout_ref.rollout.experiment_name}_train_rollouts/{global_steps}_rand{ran_id}.mp4"
            self.save_video_grid(
                step_images, save_path, fps=10,
                scores=scores,
                scores_env=scores_env,
                # score_text_fn=lambda s: f"{int(s)}",
                # score_env_text_fn=lambda s: f"{int(s)}",
                label_scores="S",
                label_scores_env="E",)
    
        format = [1.0 for _ in range(len(scores))]

        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['complete'] = complete
        data.batch['finish_step'] = finish_step
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        if self.return_env_score:
            reward_metrics['rm'] = cls_stat
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()
        return scores, reward_metrics, format_metrics, reward_format_metrics
        
    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors
        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))
        
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len 
       
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i,valid_response_length[i]-1] += verifier_score[i]
            
        reward_tensor_dict['gt_scores'] = verifier_reward

        if self.config.verifier.reward_coef!=0:
            # reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=(1)).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=(-1)).mean(dim=0).item()
        
        return reward_tensor_dict, reward_metrics
    
import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env,
                     num_cpus=config.ray_init.num_cpus)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl_vla.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl_vla.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl_vla.workers.fsdp_workers import ActorRolloutRefWorker, RobCriticWorker, RobActorRolloutRefWorker
        from verl_vla.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl_vla.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl_vla.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl_vla.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        Role.Critic: ray.remote(RobCriticWorker),
        Role.RefPolicy: ray.remote(RobActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef!=0.:
        if config.reward_model.rm_type == 'normal':
            if config.reward_model.strategy == 'fsdp':
                from verl_vla.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl_vla.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == 'prime':
            from verl_vla.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id
    
    if config.reward_model.type == "rule":
        reward_fn = RobRewardManager(num_examine=0, config=config) # note: verifier is called both inside reward_fn and outside.
    elif config.reward_model.type == "vlm_serve":
        reward_fn = RobVLMRewardManager(num_examine=0, config=config)
    elif config.reward_model.type == 'vlac':
        reward_fn = RobVLACRewardManager(num_examine=0, config=config)
    else:
        raise NotImplementedError
        
    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager(num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
