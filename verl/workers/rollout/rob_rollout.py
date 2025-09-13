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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import itertools
import contextlib
import math
import os
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
# import torch.multiprocessing as mp
import torchvision.transforms.functional as F
from typing import List, Tuple

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
import numpy as np
from PIL import Image
import tensorflow as tf
from verl import DataProto
from libero.libero import benchmark
from codetiming import Timer
from collections import deque
import random

import multiprocessing
import gc
from multiprocessing import Process, Queue
from collections import defaultdict
from multiprocessing.connection import Connection
import time
import multiprocessing as mp
from multiprocessing import connection as mp_connection
from collections import defaultdict

__all__ = ['RobHFRollout']

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image


def _to_cpu_list(x):
    import numpy as np, torch
    if isinstance(x, torch.Tensor): return x.detach().cpu().float().tolist()
    if isinstance(x, np.ndarray):   return x.astype(np.float32).tolist()
    return x

def _close_workers_with_pipe(processes, parents, grace_s=20):
    # 通知退出
    for conn in parents:
        try: conn.send(None)
        except Exception: pass
    # 等待退出
    for p in processes: p.join(timeout=grace_s)
    # 强杀残留
    for p in processes:
        if p.is_alive(): p.terminate()
    # 关闭连接
    for conn in parents:
        try: conn.close()
        except Exception: pass
        
# def center_crop_image(image: Image.Image) -> Image.Image:
  
#     crop_scale = 0.9
#     final_size = (224, 224)

#     image_np = np.array(image.convert("RGB"))
#     orig_height, orig_width, _ = image_np.shape

#     tensor_img = torch.from_numpy(image_np).permute(2, 0, 1)

#     side_scale = math.sqrt(crop_scale)
#     crop_height = int(orig_height * side_scale)
#     crop_width = int(orig_width * side_scale)

#     cropped_tensor = F.center_crop(tensor_img, (crop_height, crop_width))


#     resized_tensor = F.resize(
#         cropped_tensor.to(torch.float32), 
#         size=list(final_size), 
#         interpolation=F.InterpolationMode.BILINEAR, 
#         antialias=True
#     )

#     final_tensor = resized_tensor.round().clamp(0, 255).to(torch.uint8)

#     final_numpy = final_tensor.permute(1, 2, 0).numpy()
#     pil_image = Image.fromarray(final_numpy)

#     return pil_image.convert("RGB")

def env_worker(task_name, task_id, trial_id, config, input_queue, output_queue, is_valid, global_steps, max_steps):
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[trial_id]
    
    
    env = None
    while True:
        try:
            env, task_description = get_libero_env(task, config.model_family, resolution=256)
            break  
        except:
            print(f"*** env initialization failed ***")
            if env is not None:
                try:
                    env.close()  
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
    
    env.reset()
    obs = env.set_init_state(initial_state)
    
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
        t += 1
        
    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description":task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    
    active = True
    complete = False
    finish_step = 0
    
    while True:
        
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(inverted_action.tolist())
            
            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)
            
            
            finish_step += 1
            #if done or finish_step >= config.max_steps[config.task_suite_name]:
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images.copy() if is_valid else []
        }
        output_queue.put(output_data)
        
    

def env_worker_smolvla(task_name, task_id, trial_id, init_state, config, input_queue, output_queue, is_valid, global_steps, max_steps):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)

    env = None
    while True:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0" 
        # env, task_description = get_libero_env(task, config.model_family, resolution=256)
        try:
            env, task_description = get_libero_env(task, config.model_family, resolution=256)
            break  
        except:
            print(f"*** env initialization failed ***")
            if env is not None:
                try:
                    env.close()  
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
    
    env.reset()
    obs = env.set_init_state(init_state)
    
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, done, _ = env.step(get_libero_dummy_action(config.model_family))
        t += 1
        
    if is_valid:
        # img = obs["agentview_image"][::-1, ::-1]
        img = obs["agentview_image"][::-1, :]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description":task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    obs_global = obs
    done_global = done
    active = True
    complete = False
    finish_step = 0
    while True:
        
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            # obs, reward, done, info = env.step(inverted_action.tolist())
            try:
                obs, reward, done, info = env.step(inverted_action.tolist())
                # obs = obs_global
                # done = done_global
            except Exception as e:
                print(f"!!!!!! [Worker {os.getpid()}] CRASHED IN ENV.STEP !!!!!!")
                print(f"Action that caused crash: {inverted_action.tolist()}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            # if is_valid:
            #     # img = obs["agentview_image"][::-1, ::-1]
            #     img = obs["agentview_image"][::-1, :]  # flip up down,
            #     step_images.append(img)
            img = obs["agentview_image"][::-1, :]  # flip up down,
            step_images.append(img)
            
            
            finish_step += 1
            #if done or finish_step >= config.max_steps[config.task_suite_name]:
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            # 'valid_images': step_images.copy() if is_valid else []
            'valid_images': step_images.copy()
        }
        output_queue.put(output_data)
        
def env_worker_smolvla_vlm_reward(task_name, task_id, trial_id, init_state, config, input_queue, output_queue, is_valid, global_steps, max_steps):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)

    env = None
    while True:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0" 
        # env, task_description = get_libero_env(task, config.model_family, resolution=256)
        try:
            env, task_description = get_libero_env(task, config.model_family, resolution=256)
            break  
        except:
            print(f"*** env initialization failed ***")
            if env is not None:
                try:
                    env.close()  
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
    
    env.reset()
    obs = env.set_init_state(init_state)
    
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, done, _ = env.step(get_libero_dummy_action(config.model_family))
        t += 1
        
    if is_valid:
        # img = obs["agentview_image"][::-1, ::-1]
        img = obs["agentview_image"][::-1, :]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description":task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    obs_global = obs
    done_global = done
    active = True
    complete = False
    finish_step = 0
    while True:
        
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            # obs, reward, done, info = env.step(inverted_action.tolist())
            try:
                obs, reward, done, info = env.step(inverted_action.tolist())
                # obs = obs_global
                # done = done_global
            except Exception as e:
                print(f"!!!!!! [Worker {os.getpid()}] CRASHED IN ENV.STEP !!!!!!")
                print(f"Action that caused crash: {inverted_action.tolist()}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            # if is_valid:
            #     # img = obs["agentview_image"][::-1, ::-1]
            #     img = obs["agentview_image"][::-1, :]  # flip up down,
            #     step_images.append(img)
            img = obs["agentview_image"][::-1, :]  # flip up down,
            step_images.append(img)
            
            
            finish_step += 1
            #if done or finish_step >= config.max_steps[config.task_suite_name]:
            if done or finish_step >= max_steps:
                active = False
                complete = done
                if finish_step >= max_steps:
                    break
        
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            # 'valid_images': step_images.copy() if is_valid else []
            'valid_images': step_images.copy()
        }
        output_queue.put(output_data)
        
def env_worker_pipe(task_name, task_id, trial_id, config, conn: Connection, is_valid, global_steps, max_steps):
    """
    Pipe 版 env_worker：父进程通过 conn.send(action_list/None) 发消息；
    子进程通过 conn.send({'type': ...}) 回消息。
    """
    import gc
    import torch
    import traceback

    env = None
    try:
        # 1) 任务与初始状态
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        initial_state = initial_states[trial_id]

        # 2) 初始化 env（带重试）
        max_retries = getattr(getattr(config, "env", None), "init_retries", None)
        if max_retries is None:
            max_retries = getattr(config, "env_init_retries", 5)
        attempt, last_exc = 0, None
        while True:
            try:
                env, task_description = get_libero_env(task, config.model_family, resolution=256)
                break
            except Exception as e:
                last_exc = e
                print(f"*** env initialization failed (attempt {attempt+1}) ***: {e}")
                if env is not None:
                    try: env.close()
                    except Exception as ce: print(f"error when close the env: {ce}")
                    env = None
                torch.cuda.empty_cache(); gc.collect()
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(f"Env init failed after {max_retries} attempts") from last_exc

        # 3) reset + set init + 等待若干步
        env.reset()
        obs = env.set_init_state(initial_state)
        t = 0
        valid_images = []
        while t < config.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
            t += 1

        if is_valid:
            img = obs["agentview_image"][::-1, :]  # flip up-down
            valid_images.append(img)

        # 4) 发送 init
        conn.send({
            'type': 'init',
            'obs': obs,
            "task_description": task_description,
            'valid_images': valid_images.copy(),
            'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
            'active': True,
            'complete': False,
            'finish_step': 0
        })

        # 5) 主循环
        active = True
        complete = False
        finish_step = 0

        while True:
            msg = conn.recv()          # 父进程发来：list[actions] 或 None
            if msg is None:
                try: env.close()
                finally:
                    conn.send({'type': 'terminate'})
                break

            # 一段 action-chunk
            step_images = []
            if active:
                actions = msg
                for i in range(len(actions)):
                    a = actions[i]
                    if isinstance(a, list):
                        a = np.array(a)
                    normalized_action = normalize_gripper_action(a, binarize=True)
                    inverted_action   = invert_gripper_action(normalized_action)
                    obs, reward, done, info = env.step(inverted_action.tolist())

                    if is_valid:
                        img = obs["agentview_image"][::-1, :]
                        step_images.append(img)

                    finish_step += 1
                    if done or finish_step >= max_steps:
                        active = False
                        complete = done
                        break
            # 若已 inactive，则忽略动作但仍回消息，避免父进程阻塞

            conn.send({
                'type': 'step',
                'obs': obs,
                'active': active,
                'complete': complete,
                'finish_step': finish_step,
                'valid_images': step_images.copy() if is_valid else []
            })

    except Exception:
        tb = traceback.format_exc()
        try: conn.send({'type': 'error', 'traceback': tb})
        except Exception: pass
        try:
            if env is not None: env.close()
        except Exception: pass
        torch.cuda.empty_cache(); gc.collect()

  
LANG_TOKENS = "lang_tokens"
LANG_MASKS  = "lang_masks"

def pad_dataprotos_lang(dp_list, pad_id: int, pad_to: int | None = None):

    lengths = [dp.batch[LANG_TOKENS].shape[-1] for dp in dp_list]
    max_L = max(lengths) if pad_to is None else int(pad_to)
    
    out = []
    for dp in dp_list:
        bt = dp.batch.clone()  # tensordict 支持 clone；或者用 deepcopy(dp.batch) 也行
        tok = bt[LANG_TOKENS]  # [B, L_i]
        msk = bt[LANG_MASKS]   # [B, L_i]
        B, N, L = tok.shape

        if L < max_L:
            pad_tok = tok.new_full((B, N, max_L - L), pad_id, dtype=tok.dtype)
            pad_msk = msk.new_zeros((B, N, max_L - L), dtype=msk.dtype)
            tok = torch.cat([tok, pad_tok], dim=-1)
            msk = torch.cat([msk, pad_msk], dim=-1)

        bt[LANG_TOKENS] = tok
        bt[LANG_MASKS]  = msk

        new_dp = type(dp)(batch=bt, non_tensor_batch=dp.non_tensor_batch,)
        out.append(new_dp)
    return out

def pad_dataprotos_step_images(
    videos: List[np.ndarray], 
    padding_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将一个批次的、长度不一的视频序列填充到相同的长度。

    Args:
        videos (List[np.ndarray]): 视频列表。
            列表中的每个元素都是一个视频，其形状为 (T, H, W, C)，
            其中 T 是帧数 (可变)，H, W, C 是高、宽、通道数 (固定)。
        padding_value (int): 用于填充的数值，通常为0。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - padded_videos (torch.Tensor): 填充后的视频张量。
          形状为 (B, T_max, H, W, C)，其中 B 是批次大小，
          T_max 是这个批次中最长的视频的帧数。
        - attention_mask (torch.Tensor): 注意力遮罩张量。
          形状为 (B, T_max)，其中真实帧的位置为1，填充帧的位置为0。
    """
    if not videos:
        # 处理空输入的情况
        return torch.empty(0), torch.empty(0)

    # --- 1. 获取每个视频的形状信息 ---
    # 获取每个视频的帧数
    lengths = [video.shape[0] for video in videos]
    # 获取批次大小
    batch_size = len(videos)
    # 获取这个批次中的最大帧数
    max_len = max(lengths)
    
    # 假设所有视频的 H, W, C 维度是相同的，我们从第一个视频中获取
    h, w, c = videos[0].shape[1:]
    dtype = videos[0].dtype

    # --- 2. 创建用于填充的目标张量和遮罩张量 ---
    # 创建一个全零（或指定padding_value）的目标张量
    padded_videos = torch.full(
        (batch_size, max_len, h, w, c), 
        fill_value=padding_value, 
        dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype # 保持与输入 NumPy 数组相同的数据类型
    )
    
    # 创建一个全零的遮罩张量，稍后我们会将真实帧的位置设为1
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    # --- 3. 循环遍历，将原始数据复制到目标张量中 ---
    for i, video in enumerate(videos):
        # 获取当前视频的实际长度
        current_len = lengths[i]
        
        # 将视频数据从 NumPy 数组转换为 PyTorch 张量
        video_tensor = torch.from_numpy(video)
        
        # 将原始视频数据复制到 padded_videos 张量的相应位置
        # 例如，对于第一个视频，我们填充 padded_videos[0, :125, :, :, :]
        padded_videos[i, :current_len, ...] = video_tensor
        
        # 在 attention_mask 中，将真实帧的位置标记为1
        # 例如，对于第一个视频，我们将 attention_mask[0, :125] 设为1
        attention_mask[i, :current_len] = 1
        
    return padded_videos, attention_mask

class RobHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.max_steps = {   "libero_spatial": 256,   # max step length 193
                                    "libero_object": 512,    # max step length 254
                                    "libero_goal": 512,      # max step length 270
                                    "libero_10": 512,        # max step length 505
                                    "libero_90": 512         # max step length 373 org 400 now change to 512
                                }
        if self.config.vla in ["smolvla"]:
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        self.vla_preprocess()
        #oft add
        # unnorm_key=config.unnorm_key
        # if  unnorm_key not in self.module.norm_stats and f"{unnorm_key}_no_noops" in self.module.norm_stats:
        #     unnorm_key = f"{unnorm_key}_no_noops"
        # assert unnorm_key in self.module.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
        # self.config.unnorm_key = unnorm_key
        #add end
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     for gpu in gpus:  
        #         tf.config.experimental.set_memory_growth(gpu, True)
    
    def vla_preprocess(self):
        if self.config.vla in ["openvla","openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:  
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if  self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"


    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]
        is_train = prompts.meta_info.get('is_train', False)
        if not is_train:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        if self.config.vla == "smolvla":
            if self.use_world_model and is_train:
                output = [self._generate_minibatch_smolvla_wm(p) for p in batch_prompts]
            elif self.config.reward_type == 'vlm' and is_train:
                output = [self._generate_minibatch_smolvla_vlm_reward(p) for p in batch_prompts]
            else:
                # output = [self._generate_minibatch_smolvla(p) for p in batch_prompts]
                output = [self._generate_minibatch_smolvla_vlm_reward(p) for p in batch_prompts]
            output = pad_dataprotos_lang(output, pad_id=self.module.language_tokenizer.pad_token_id, pad_to=None)
        else:
            output = [self._generate_minibatch(p) for p in batch_prompts]
        
        output = DataProto.concat(output)
        return output
    
    
    def process_input(self,inputs:list, task_descriptions:list):
        
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature  = self.processor(prompt, image)
            
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_batch_feature = self.processor(prompt, wrist_image)
                primary_pixel_values = batch_feature["pixel_values"]
                batch_feature["pixel_values"] = torch.cat([primary_pixel_values] + [wrist_batch_feature["pixel_values"]], dim=1)
                
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert  torch.all(padding_mask==batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int() 
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"] , dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata
    
    def process_input_smolvla(self, inputs:list, task_descriptions:list):
        
        batchdata = {"observation.images.image":[], 
                     "observation.images.image_is_pad": [],
                     "task": []}  
        if "state" in inputs[0].keys():
            batchdata["observation.state"] = []
            batchdata["observation.state_is_pad"] = []
        if "wrist_image" in inputs[0].keys():
            batchdata["observation.images.wrist_image"] = []
            batchdata["observation.images.wrist_image_is_pad"] = []
            
        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
            batchdata["task"].append(task_description)
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            image_tensor = transform(image).to(torch.bfloat16)
            batchdata["observation.images.image"].append(image_tensor)
            batchdata["observation.images.image_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
            
                
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_image_tensor = transform(wrist_image).to(torch.bfloat16)
                batchdata["observation.images.wrist_image"].append(wrist_image_tensor) 
                batchdata["observation.images.wrist_image_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
            
            if "state" in input.keys():
                state_tensor = torch.tensor(input['state']).unsqueeze(0).to(torch.bfloat16)
                batchdata["observation.state"].append(state_tensor)
                batchdata["observation.state_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
        
        device = torch.device('cuda') 
        
        batchdata["observation.images.image"] = torch.stack([x for x in batchdata["observation.images.image"]]).to(device)
        batchdata["observation.images.image_is_pad"] = torch.stack([x for x in batchdata["observation.images.image_is_pad"]]).to(device)
        if "state" in input.keys():
            batchdata["observation.state"] = torch.stack([x for x in batchdata["observation.state"]]).to(device)
            batchdata["observation.state_is_pad"] = torch.stack([x for x in batchdata["observation.state_is_pad"]]).to(device)
        if "wrist_image" in input.keys():
            batchdata["observation.images.wrist_image"] = torch.stack([x for x in batchdata["observation.images.wrist_image"]]).to(device)
            batchdata["observation.images.wrist_image_is_pad"] = torch.stack([x for x in batchdata["observation.images.wrist_image_is_pad"]]).to(device)
        
        return batchdata
   
    
        
    def _generate_minibatch(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        processes = []
        input_queues = []
        output_queues = []
        
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker,
                args=(task_name, t_id, tr_id, self.config, input_q, output_q, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs']))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name']
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
           
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                    "responses": vla_output["responses"],
                    "input_ids": vla_output["input_ids"],
                    "attention_mask": vla_output["attention_mask"],
                    "pixel_values": vla_output["pixel_values"],
                    "action": actions,
                    "step": step
                }
            vla_history.append(step_data)
            
            for  idx in active_indices:
                input_queues[idx].put(actions[idx])
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'])
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        
        torch.cuda.empty_cache()
        
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': []
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = []
        batch["finish_step"] = []
        
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_smolvla(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
        processes = []
        input_queues = []
        output_queues = []
        # mp.set_start_method('spawn')
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            in_state = init_state[idx].cpu().numpy()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker_smolvla,
                args=(task_name, t_id, tr_id, in_state, self.config, input_q, output_q, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs']))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name']
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
           
            
            vla_input = self.process_input_smolvla(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step_smolvla(vla_input, use_sde=is_train)
            actions = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            step_data["lang_tokens"] = vla_output["lang_tokens"]
            step_data["lang_masks"] = vla_output["lang_masks"]

            vla_history.append(step_data)
            
            for idx in active_indices:
                input_queues[idx].put(actions[idx])
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'])
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        torch.cuda.empty_cache()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        self.module.train()
        batch = {"observation.images.image":[], 
                "observation.images.image_is_pad": [],
                # "observation.images.wrist_image":[], 
                # "observation.images.wrist_image_is_pad": [],
                "observation.state":[], 
                "observation.state_is_pad": [],
                "action_tensor": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "lang_tokens": [],
                "lang_masks": []
                }  
        # for k in ["observation.images.image", "observation.images.wrist_image", "observation.images.image_is_pad", "observation.images.wrist_image_is_pad",
        #           "observation.state", "observation.state_is_pad", "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
        for k in ["observation.images.image", "observation.images.image_is_pad",
                  "observation.state", "observation.state_is_pad", "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        batch["complete"] = []
        batch["finish_step"] = []
        
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_smolvla_vlm_reward(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
        processes = []
        input_queues = []
        output_queues = []
        # mp.set_start_method('spawn')
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            in_state = init_state[idx].cpu().numpy()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker_smolvla_vlm_reward,
                args=(task_name, t_id, tr_id, in_state, self.config, input_q, output_q, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs']))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "step_images": [init_data['obs']['agentview_image'][::-1]]
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        step = 0
        vla_history = []
        while step < max_steps:
            # active_indices = [i for i, r in enumerate(task_records) if r['active']]
            active_indices = [i for i, r in enumerate(task_records)]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
           
            
            vla_input = self.process_input_smolvla(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step_smolvla(vla_input, use_sde=is_train)
            actions = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            step_data["lang_tokens"] = vla_output["lang_tokens"]
            step_data["lang_masks"] = vla_output["lang_masks"]

            vla_history.append(step_data)
            
            for idx in active_indices:
                input_queues[idx].put(actions[idx])
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'])
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        torch.cuda.empty_cache()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        self.module.train()
        batch = {"observation.images.image":[], 
                "observation.images.image_is_pad": [],
                # "observation.images.wrist_image":[], 
                # "observation.images.wrist_image_is_pad": [],
                "observation.state":[], 
                "observation.state_is_pad": [],
                "action_tensor": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "lang_tokens": [],
                "lang_masks": [],
                }  
        # for k in ["observation.images.image", "observation.images.wrist_image", "observation.images.image_is_pad", "observation.images.wrist_image_is_pad",
        #           "observation.state", "observation.state_is_pad", "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
        for k in ["observation.images.image", "observation.images.image_is_pad",
                  "observation.state", "observation.state_is_pad", "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        batch["complete"] = []
        batch["finish_step"] = []
        batch["step_images"] = []
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            batch["step_images"].append(np.stack(k["step_images"]))
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        batch["step_images"] = padded_step_images.to(device=batch['observation.images.image'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['observation.images.image'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_smolvla_pipe(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id   = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id  = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)

        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)

        init_timeout = getattr(self.config, "env_init_timeout", 120)
        step_total_timeout = getattr(self.config, "env_step_total_timeout", 120)
        join_grace = getattr(self.config, "env_join_grace", 20)

        # === 启动子进程（spawn）并建立 Pipe ===
        ctx = mp.get_context("spawn")
        processes, parents = [], []
        conn_to_idx = {}
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=env_worker,
                args=(task_name, t_id, tr_id, self.config, child_conn, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            parents.append(parent_conn)
            conn_to_idx[parent_conn] = idx

        # === 聚合 init：用 connection.wait ===
        inputs, task_descriptions, task_records = [], [], []
        valid_video = defaultdict(list)

        pending = set(range(batch_size))
        deadline = time.time() + init_timeout
        while pending:
            timeout = max(0.0, deadline - time.time())
            ready = mp_connection.wait([parents[i] for i in pending], timeout)
            if not ready:
                stuck = [(i, processes[i].is_alive(), processes[i].exitcode) for i in pending]
                _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                raise RuntimeError(f"Init wait timeout; stuck workers: {stuck}")
            for conn in ready:
                idx = conn_to_idx[conn]
                msg = conn.recv()
                typ = msg.get('type')
                if typ == 'error':
                    _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                    raise RuntimeError(f"env_worker[{idx}] crashed during init:\n{msg['traceback']}")
                if typ != 'init':
                    _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                    raise RuntimeError(f"worker {idx} bad init type: {typ}")
                task_descriptions.append(msg["task_description"])
                inputs.append(self._obs_to_input(msg['obs']))
                task_records.append({
                    "active": msg['active'],
                    "complete": msg['complete'],
                    "finish_step": msg['finish_step'],
                    "task_file_name": msg['task_file_name']
                })
                if is_valid:
                    valid_video[msg['task_file_name']].extend(msg.get('valid_images', []))
                pending.remove(idx)

        step = 0
        vla_history = []

        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            if not active_indices: break

            # 整 batch 推一次模型（保持你的逻辑）
            vla_input = self.process_input_smolvla(inputs, task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step_smolvla(vla_input, use_sde=is_train)
            actions = vla_output["action"]

            # 记录历史
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            step_data["lang_tokens"] = vla_output["lang_tokens"]
            step_data["lang_masks"] = vla_output["lang_masks"]
            vla_history.append(step_data)

            # 发送动作（仅 active）
            for i in active_indices:
                parents[i].send(_to_cpu_list(actions[i]))

            # 聚合本轮 step：wait 直到所有 active 都返回
            pending = set(active_indices)
            deadline = time.time() + step_total_timeout
            new_inputs = list(inputs)
            while pending:
                timeout = max(0.0, deadline - time.time())
                ready = mp_connection.wait([parents[i] for i in pending], timeout)
                if not ready:
                    stuck = [(i, processes[i].is_alive(), processes[i].exitcode) for i in pending]
                    _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                    raise RuntimeError(f"Step wait timeout; stuck workers: {stuck}")
                for conn in ready:
                    idx = conn_to_idx[conn]
                    msg = conn.recv()
                    typ = msg.get('type')
                    if typ == 'error':
                        _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                        raise RuntimeError(f"env_worker[{idx}] crashed:\n{msg['traceback']}")
                    if typ not in ('step', 'done', 'terminate'):
                        _close_workers_with_pipe(processes, parents, grace_s=join_grace)
                        raise RuntimeError(f"worker {idx} unexpected msg type: {typ}")

                    if typ in ('step', 'done'):
                        new_inputs[idx] = self._obs_to_input(msg['obs'])
                        task_records[idx]['active'] = msg['active']
                        task_records[idx]['complete'] = msg['complete']
                        task_records[idx]['finish_step'] = msg['finish_step']
                        if is_valid:
                            valid_video[task_records[idx]['task_file_name']].extend(msg.get('valid_images', []))

                    pending.remove(idx)

            inputs = new_inputs
            step += self.config.action_chunks_len

        # 关闭 workers
        _close_workers_with_pipe(processes, parents, grace_s=join_grace)
        torch.cuda.empty_cache()

        # 保存 valid 视频
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(images, self.config.experiment_name, task_file, global_steps, complete)

        self.module.train()

        # 组装训练 batch（保持你的键与形状）
        batch = {
            "observation.images.image": [],
            "observation.images.image_is_pad": [],
            "observation.state": [],
            "observation.state_is_pad": [],
            "action_tensor": [],
            "x_t": [],
            "t": [],
            "x_next": [],
            "lang_tokens": [],
            "lang_masks": []
        }
        for k in list(batch.keys()):
            for h in vla_history:
                batch[k].append(h[k])
            batch[k] = torch.stack(batch[k], dim=1)

        batch["complete"] = []
        batch["finish_step"] = []
        for rec in task_records:
            batch["complete"].append(rec["complete"])
            batch["finish_step"].append(rec["finish_step"])

        dev = batch['observation.images.image'].device
        batch["complete"]    = torch.tensor(batch["complete"], dtype=torch.bool,  device=dev)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=dev)

        output_batch = TensorDict(batch, batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def save_obs_png(self, current_obs_batch_np, save_path = "stitched_observations_grid.png"):
        row1 = np.concatenate((current_obs_batch_np[0], current_obs_batch_np[1]), axis=1)
        row2 = np.concatenate((current_obs_batch_np[2], current_obs_batch_np[3]), axis=1)
        grid_image_np = np.concatenate((row1, row2), axis=0)
        grid_image_pil = Image.fromarray(grid_image_np)
        grid_image_pil.save(save_path)

    def _generate_minibatch_smolvla_wm(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        current_obs_batch_np = init_state.cpu().numpy()
        

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,  # 在World Model中，'complete'通常只在达到max_steps时为True
                "finish_step": 0,
                "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}"
            })
            
        valid_video = defaultdict(list)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        if is_valid:
            # current_obs_batch_np 的形状是 [B, H, W, 3]
            # 我们需要把它拆开，存到每个任务对应的视频列表中
            for idx in range(batch_size):
                task_file = task_records[idx]['task_file_name']
                # 从批次中取出第 idx 帧图像，并进行上下翻转（如果需要的话，这模仿了原函数的行为）
                img = current_obs_batch_np[idx]  # [::-1, :, :] 
                valid_video[task_file].append(img)
        
        step = 0
        vla_history = []
        trajectory_video_batch = [current_obs_batch_np]
        vla_timings = []
        wm_timings = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            if not active_indices:
                break
            
            inputs = [self._obs_to_input(obs) for obs in current_obs_batch_np]
            
            vla_input = self.process_input_smolvla(inputs, task_descriptions)
            vla_input.update(meta_info)
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_smolvla(vla_input, use_sde=is_train)
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions_batch
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            step_data["lang_tokens"] = vla_output["lang_tokens"]
            step_data["lang_masks"] = vla_output["lang_masks"]

            vla_history.append(step_data)
            if self.world_model is None:
                raise ValueError("World Model Worker Group has not been set!")
            
            with Timer(name="World_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            trajectory_video_batch.append(next_obs_batch_np)

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :]  #[::-1, :, :]
                        valid_video[task_file].append(img)
            
            for r in task_records:
                if r['active']:
                    r['finish_step'] = step
                    if r['finish_step'] >= max_steps:
                        r['active'] = False
                        r['complete'] = True 
                                
        torch.cuda.empty_cache()
        if is_valid:
            for task_file, images in valid_video.items():
                complete_flags = [r['complete'] for r in task_records if r['task_file_name'] == task_file]
                complete = complete_flags[0] if complete_flags else False
                
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        initial_frame_expanded = np.expand_dims(trajectory_video_batch[0], axis=1) # -> (B, 1, H, W, C)
        video_chunks = [initial_frame_expanded] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"output/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        self.world_model.save_video_grid(
            full_trajectory_video, 
            f"output/{self.config.experiment_name}/trajectory_grid_{global_steps}.mp4"
        )
            
            
        print("\n" + "="*50)
        print(" Performance Measurement Report")
        print("="*50)
        
        if vla_timings:
            print("\n--- VLA Inference (`_generate_one_step_smolvla`) ---")
            print(f"  Total steps measured: {len(vla_timings)}")
            print(f"  Total time spent:     {np.sum(vla_timings):.4f} seconds")
            print(f"  Average time per step:  {np.mean(vla_timings):.4f} seconds")
            print(f"  Standard deviation:     {np.std(vla_timings):.4f} seconds")
            print(f"  Fastest step:         {np.min(vla_timings):.4f} seconds")
            print(f"  Slowest step:         {np.max(vla_timings):.4f} seconds")

        if wm_timings:
            print("\n--- World Model Step (`world_model.step`) ---")
            print(f"  Total steps measured: {len(wm_timings)}")
            print(f"  Total time spent:     {np.sum(wm_timings):.4f} seconds")
            print(f"  Average time per step:  {np.mean(wm_timings):.4f} seconds")
            print(f"  Standard deviation:     {np.std(wm_timings):.4f} seconds")
            print(f"  Fastest step:         {np.min(wm_timings):.4f} seconds")
            print(f"  Slowest step:         {np.max(wm_timings):.4f} seconds (首次调用可能因CUDA Graph录制而较慢)")

        if vla_timings and wm_timings:
            total_mean_per_step = np.mean(vla_timings) + np.mean(wm_timings)
            print("\n--- Combined ---")
            print(f"  Average total processing time per step: {total_mean_per_step:.4f} seconds")

        print("="*50 + "\n")

            
        self.module.train()
        batch = {"observation.images.image":[], 
                "observation.images.image_is_pad": [],
                # "observation.images.wrist_image":[], 
                # "observation.images.wrist_image_is_pad": [],
                # "observation.state":[], 
                # "observation.state_is_pad": [],
                "action_tensor": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "lang_tokens": [],
                "lang_masks": []
                }  
        # for k in ["observation.images.image", "observation.images.wrist_image", "observation.images.image_is_pad", "observation.images.wrist_image_is_pad",
        #           "observation.state", "observation.state_is_pad", "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
        for k in ["observation.images.image", "observation.images.image_is_pad",
                #   "observation.state", "observation.state_is_pad", 
                  "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        if self.config.vla == "openvla-oft":
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
        
        
            param_ctx = contextlib.nullcontext()

            # make sampling args can be overriden by inputs
            do_sample = prompts.get('do_sample', self.config.do_sample)
        

            temperature = prompts.get('temperature', self.config.temperature)

            #generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions, response = self.module.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        padding_idx = self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.unnorm_key,
                        temperature=temperature, )
            
            # print('Waiting for debugger 5679'); import os,debugpy; debugpy.listen(('localhost', 5679 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()

            assert self.processor.tokenizer.pad_token_id is not None

            assert idx.ndim == 2
            idx = verl_F.pad_sequence_to_length(idx,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch ={
                    'responses': response,
                    'input_ids': idx,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                }

            return batch
        
        elif self.config.vla == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            
            # used to construct attention_mask
            eos_token_id = prompts['eos_token_id']
            pad_token_id = prompts['pad_token_id']

            batch_size = idx.size(0)
            prompt_length = idx.size(1)
            #self.module.eval()
            param_ctx = contextlib.nullcontext()

            do_sample = prompts.get('do_sample', self.config.do_sample)
            response_length =  self.module.get_action_dim(self.config.unnorm_key)
            top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
            top_k = prompts.get('top_k', self.config.get('top_k', 0))
            if top_k is None:
                top_k = 0
            top_k = max(0, top_k)  # to be compatible with vllm

            temperature = prompts.get('temperature', self.config.temperature)
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    output = self.module.generate(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    
           
            seq = output.sequences
            sequence_length = prompt_length + response_length
            delta_length = sequence_length - seq.shape[1]
            
            assert delta_length == 0
            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            response_length = response.size(1)
            #delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            #delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            #response_position_ids = position_ids[:, -1:] + delta_position_id
            #position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = response.detach().cpu().numpy()
            discretized_actions = self.module.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1)
            normalized_actions = self.module.bin_centers[discretized_actions]

            # Unnormalize actions
            action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            
            actions = np.expand_dims(actions, axis=1)
            
            assert self.processor.tokenizer.pad_token_id is not None
            assert prompt.ndim == 2
            prompt = verl_F.pad_sequence_to_length(prompt,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert seq.ndim == 2
            seq = verl_F.pad_sequence_to_length(seq,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            batch ={
                    'prompts': prompt,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    #'position_ids': position_ids
                }
            
            return batch
        
    @torch.no_grad()
    def _generate_one_step_smolvla(self, prompts: dict, use_sde: bool = False):
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # actions = self.module.select_action(prompts)
                actions, lang_tokens, lang_masks, return_dict = self.module.predict_action_chunk(prompts, use_sde=use_sde)
        
        batch = prompts.copy()
        batch["action_tensor"] = actions
        batch["action"] = actions.to(torch.float32).cpu().numpy()
        batch["return_dict"] = return_dict
        batch["lang_tokens"] = lang_tokens
        batch["lang_masks"] = lang_masks
        
        return batch

    def _obs_to_input(self, obs):
        if self.use_world_model:
            return {
                "full_image": obs,
            }
        else:
            if self.config.num_images_in_input > 1:
                return {
                    "full_image": get_libero_image(obs, 224),
                    "wrist_image": get_libero_wrist_image(obs, 224),
                    "state": np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]
                    ])
                }
            else:
                return {
                    "full_image": get_libero_image(obs, 224),
                    "state": np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]
                    ])
                }