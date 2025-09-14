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
from typing import List
import re
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
    description: str

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

    def verify_env(self, data):
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

    def get_rewards_from_judge_batch_sync(
        self,
        tasks: List[RewardTask], 
        max_workers: int = 10
    ) -> List[float]:
        """
        Args:
            tasks: 一个RewardTask对象的列表。
            max_workers: 同时执行任务的最大线程数。

        Returns:
            一个浮点数列表，包含了与输入tasks顺序对应的奖励分数。
        """
        results = [0.0] * len(tasks) # 初始化一个与tasks等长的结果列表
        results_text = ['a'] * len(tasks) 
        # 使用线程池来管理并发请求
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            # future_to_index 映射了每个future对象到它的原始索引
            future_to_index = {
                executor.submit(self.fetch_one_reward_sync, task, i): i
                for i, task in enumerate(tasks)
            }

            # 当每个任务完成时，处理它的结果
            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    # 获取任务的结果 (index, reward)
                    _, reward, response_text = future.result()
                    # 将奖励值放入结果列表中正确的位置
                    results[original_index] = reward
                    results_text[original_index] = response_text
                except Exception as e:
                    print(f"Task {original_index} generated an exception: {e}")
                    # 在结果列表中保留默认值0.0

        return results, results_text

    def parse_reward_from_box(self, response: str) -> float:
        """
        """
        match = re.search(r"\\box\{(.*?)\}", response.lower())
        if match:
            answer = match.group(1).strip()
            if answer == "success":
                return 1.0
        return 0.0

#     def fetch_one_reward_sync(self, task: RewardTask, task_index: int) -> tuple[int, float]:
#         """
#         """
#         if not task.frames:
#             print(f"Task {task_index}: Input list is empty")
#             return task_index, 0.0

#         prompt_text = f"""Please analyze the following image sequence to determine task completion.\n
# Task Description: {task.description}\n
# Input: A sequence of {len(task.frames)} temporally ordered image frames.\n
# Instruction: Based on the visual evidence in the sequence, judge if the task objective has been met.\n
# Required Output Format: Output your thought process first, then output the final answer. Your final answer must be strictly one of the following two words: 'Success' or 'Failure', and it must be enclosed in \\box{{}}.\n
# Example: \\box{{Success}}"""

# #         prompt_text = f"""请分析以下图像序列以判断任务是否完成。
# # 任务描述：{task.description}
# # 输入：一个由 {len(task.frames)} 帧按时间顺序排列的图像序列。
# # 指令要求：请根据序列中的视觉证据，判断任务目标是否已经达成。
# # 要求输出格式：先输出你的思考过程，然后输出最终答案。你的最终答案必须也只能是'成功'或'失败'这两个词中的一个，并且必须用 \\box{{}} 包裹。
# # 例如：\\box{{成功}}"""

#         content = [{"type": "text", "text": prompt_text}]
#         for frame_url in task.frames:
#             content.append({"type": "image_url", "image_url": {"url": frame_url}})

#         try:
#             print(f"Sending request for task {task_index}...")
#             completion = self.client.chat.completions.create(
#                 model="judge",
#                 messages=[{"role": "user", "content": content}],
#                 max_tokens=500,
#                 temperature=0.0
#             )
#             response_text = completion.choices[0].message.content
#             print(f"Task {task_index} response: '{response_text}'")
#             reward = self.parse_reward_from_box(response_text)
#             return task_index, reward, response_text

#         except Exception as e:
#             print(f"Task {task_index} - Callback API Error: {e}")
#             return task_index, 0.0, "Empty"
        
    def fetch_one_reward_sync(self, task: RewardTask, task_index: int) -> tuple[int, float, str]:
        """
        Run a strict VLM judge on a sequence of frames to decide Success/Failure.
        Returns: (task_index, reward_float, raw_response_text)
        reward_float is parsed from the model's \\box{{Success}} / \\box{{Failure}} output.
        """
        # Guard: empty input
        if not getattr(task, "frames", None):
            print(f"Task {task_index}: Input frame list is empty")
            return task_index, 0.0, "Empty"

        n_frames = len(task.frames)
        system_prompt = build_system_prompt(mode="v1") 

        # === User文本头：任务与输入说明 ===
        user_header = (
            f"BEGIN INPUT\n"
            f"Task: {task.description}\n\n"
            f"Frames: {n_frames} frames in chronological order (Frame 1 = earliest … Frame {n_frames} = latest).\n"
            f"Judge using only the provided frames.\n"
            f"END INPUT\n"
        )

        # === 组装多模态内容：文本 + 每帧的编号与图片 ===
        # 建议把“Frame k:”这行文本放在对应图片前，帮助模型建立时序对齐。
        user_content = [{"type": "text", "text": user_header}]
        for i, frame_url in enumerate(task.frames, start=1):
            user_content.append({"type": "text", "text": f"Frame {i}:"})
            user_content.append({"type": "image_url", "image_url": {"url": frame_url}})

        try:
            print(f"Sending request for task {task_index} with {n_frames} frames...")
            completion = self.client.chat.completions.create(
                model="judge",  # 你的72B VLM别名
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            response_text = completion.choices[0].message.content or ""
            print(f"Task {task_index} response: '{response_text}'")

            # 解析 \box{Success} / \box{Failure}
            reward = self.parse_reward_from_box(response_text)  # 你已有的解析函数
            return task_index, reward, response_text

        except Exception as e:
            print(f"Task {task_index} - Callback API Error: {e}")
            return task_index, 0.0, "Empty"

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
                _, buffer = cv2.imencode('.jpg', frame)
                base64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(f"data:image/jpeg;base64,{base64_str}")
            reward_task = RewardTask(frames=base64_frames, description=task_lang[i])
            tasks_to_process.append(reward_task)
        return tasks_to_process
            
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
        
    def verify(self, data):
        if self.return_env_score and ('complete' in data.batch):
            score_env, reward_metrics_env, format_metrics_env, reward_format_metrics_env = self.verify_env(data)
        step_images = data.batch["step_images"]
        step_images_mask = data.batch["step_images_mask"]
        task_lang = data.non_tensor_batch['task_lang']
        
        reward_tasks = self.get_reward_tasks(step_images, step_images_mask, task_lang)
        scores, results_text = self.get_rewards_from_judge_batch_sync(reward_tasks, max_workers=64)
        breakpoint()
        self.save_video_grid(
            step_images, 
            "output_4x4_padded_white.mp4", 
            fps=10, 
            grid_size=(4, 4), 
            padding=10,
            pad_value=255
        )
    
        format = [1.0 for _ in range(len(scores))]

        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=data.batch[self.data_key].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch[self.data_key].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
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
