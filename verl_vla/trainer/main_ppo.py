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
from typing import List, Tuple, Optional, Dict
import re
from collections import Counter
from dataclasses import dataclass
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed


import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/LIBERO")
from verl_vla import DataProto
import torch
from verl_vla.utils.reward_score import gsm8k, math, countdown, multiply, logic

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

    def verify(self, data):
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
        if config.actor_rollout_ref.model.vla == 'smolvla':
            self.data_key = 'action_tensor'
        else:
            self.data_key = 'responses'
        self.client = openai.OpenAI(
            base_url="http://localhost:18901/v1", # 请确保这是您的服务地址
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
        pad_value: int = 0
    ):
        """
        将一个批次的视频张量拼接成一个网格，并保存为视频文件。

        Args:
            video_tensor (torch.Tensor): 输入的视频张量。
                形状应为 (B, T, H, W, C)，数据类型为 torch.uint8。
                B: 批次大小 (视频数量)
                T: 帧数
                H: 高度
                W: 宽度
                C: 通道数 (应为3，即RGB)
            output_path (str): 输出视频文件的路径 (例如 "output.mp4")。
            fps (int): 输出视频的帧率 (Frames Per Second)。
            grid_size (tuple, optional): 网格的 (行数, 列数)。
                如果为 None，函数将自动计算一个尽可能接近方形的网格。
            padding (int): 网格中图像之间的间距（像素）。
            pad_value (int): 间距的颜色 (0=黑色, 255=白色)。
        """
        # --- 1. 参数校验和准备 ---
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(video_tensor)}")
        
        if video_tensor.dim() != 5:
            raise ValueError(f"Input tensor must have 5 dimensions (B, T, H, W, C), but got {video_tensor.dim()}")

        if video_tensor.dtype != torch.uint8:
            print(f"Warning: Input tensor dtype is {video_tensor.dtype}, converting to uint8. "
                "Values should be in the range [0, 255].")
            video_tensor = video_tensor.byte()

        B, T, H, W, C = video_tensor.shape
        
        if C != 3:
            raise ValueError(f"Input tensor must have 3 channels (RGB), but got {C}")

        # 将张量移动到 CPU 并转换为 NumPy 数组，以便 OpenCV 处理
        # .permute(0, 1, 2, 3, 4) 是为了确保顺序，虽然在这里不是必须的
        # .contiguous() 确保内存是连续的，对于某些操作是必需的
        video_np = video_tensor.cpu().numpy()

        # --- 2. 自动计算网格尺寸 ---
        if grid_size is None:
            # 尝试找到最接近方形的布局
            rows = int(math.sqrt(B))
            cols = (B + rows - 1) // rows # 向上取整
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        if rows * cols < B:
            raise ValueError(f"Grid size {grid_size} is too small for batch size {B}")

        # --- 3. 计算最终视频帧的尺寸 ---
        grid_h = rows * H + (rows - 1) * padding
        grid_w = cols * W + (cols - 1) * padding

        # --- 4. 初始化视频写入器 ---
        # 定义视频编码器，'mp4v' 是一种常见的选择，兼容性好
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # 或者使用 'avc1' for H.264
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for path {output_path}")

        # --- 5. 逐帧处理并写入 ---
        for t in range(T):
            # 创建一个用于拼接的空白大画布
            grid_frame = np.full((grid_h, grid_w, C), pad_value, dtype=np.uint8)
            
            # 从批次中获取当前时间步的所有帧
            frame_batch = video_np[:, t, :, :, :] # Shape: (B, H, W, C)
            
            # 将帧填充到网格中
            for i in range(B):
                row_idx = i // cols
                col_idx = i % cols
                
                # 计算当前帧在网格中的起始坐标
                start_y = row_idx * (H + padding)
                start_x = col_idx * (W + padding)
                
                # 从 BGR (OpenCV 默认) 转换为 RGB
                # PyTorch/NumPy 通常是 RGB, OpenCV 是 BGR
                # 如果你的输入已经是 BGR，可以注释掉这行
                frame_rgb = frame_batch[i]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # 将帧复制到画布上
                grid_frame[start_y : start_y + H, start_x : start_x + W] = frame_bgr
            
            # 将拼接好的帧写入视频
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

    def verify(self, data):
        if self.return_env_score and ('complete' in data.batch):
            score_env, reward_metrics_env, format_metrics_env, reward_format_metrics_env = self.verify_env(data)
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
        cls_stat = self.compute_all_metrics(score_env, pred_success)
        # breakpoint()
        # self.save_video_grid(
        #     step_images, 
        #     "output_4x4_padded_white.mp4", 
        #     fps=10, 
        #     grid_size=(4, 4), 
        #     padding=10,
        #     pad_value=255
        # )
    
        format = [1.0 for _ in range(len(scores))]

        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['complete'] = complete
        data.batch['finish_step'] = finish_step
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
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
