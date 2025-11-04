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
from pathlib import Path
import cv2 

from verl_vla import DataProto
from verl_vla.utils.torch_functional import get_eos_mask
import verl_vla.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

from verl_vla.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
import numpy as np
from PIL import Image
import tensorflow as tf
import os, json, h5py, numpy as np
from collections import defaultdict
import robosuite.utils.transform_utils as T
import torch
import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/siiRL")
from siirl.workers.environment.vla import LIBEROAdapter

# 获取所有物理上的GPU设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 遍历所有GPU，并为它们设置内存动态增长
    # 这会使得TensorFlow按需分配显存
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Successfully set memory growth to True for {len(gpus)} GPU(s)")
  except RuntimeError as e:
    # 内存动态增长的设置必须在GPU被初始化之前完成
    print(e)
    
from verl_vla import DataProto
from libero.libero import benchmark
from codetiming import Timer
from collections import deque
import random

import multiprocessing
import gc
# from multiprocessing import Process, Queue
import multiprocessing as mp
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

def rotate_180(img):
    # (H,W,3) or (T,H,W,3) -> 同型返回
    if img.ndim == 4:
        return img[:, ::-1, ::-1, :]
    return img[::-1, ::-1, :]

def as_uint8(arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.max() <= 1.0 + 1e-6:
            arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def h5_create_dset(grp, name, data, img_like=False):
    data = np.ascontiguousarray(data)
    kwargs = dict(compression="gzip", compression_opts=4, shuffle=True)
    if img_like:
        assert data.ndim == 4 and data.shape[-1] in (1, 3, 4)
        chunks = (1, data.shape[1], data.shape[2], data.shape[3])
        return grp.create_dataset(name, data=data, chunks=chunks, **kwargs)
    else:
        if data.ndim >= 2:
            chunks = (min(1024, data.shape[0]),) + tuple(data.shape[1:])
        elif data.ndim == 1:
            chunks = (min(4096, data.shape[0]),)
        else:
            chunks = None
        return grp.create_dataset(name, data=data, chunks=chunks, **kwargs)

def open_or_create_h5(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, "a")
    if "data" not in f:
        f.create_group("data")
    return f


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
    curr_step = 0
    finish_step = 0
    update_finish = True
    final_done = False
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
            
            
            curr_step += 1
            #if done or finish_step >= config.max_steps[config.task_suite_name]:
            if not done and update_finish:
                finish_step += 1
            
            if done:
                update_finish = False
                final_done = True
                
            if curr_step >= max_steps:
                active = False
                complete = final_done
                break
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete_raw': complete,
            'finish_step_raw': finish_step,
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

  
# LANG_TOKENS = "lang_tokens"
# LANG_MASKS  = "lang_masks"

# def pad_dataprotos_lang(dp_list, pad_id: int, pad_to: int | None = None, LANG_TOKENS = "lang_tokens", LANG_MASKS  = "lang_masks"):

#     lengths = [dp.batch[LANG_TOKENS].shape[-1] for dp in dp_list]
#     max_L = max(lengths) if pad_to is None else int(pad_to)
    
#     out = []
#     for dp in dp_list:
#         bt = dp.batch.clone()  # tensordict 支持 clone；或者用 deepcopy(dp.batch) 也行
#         tok = bt[LANG_TOKENS]  # [B, L_i]
#         msk = bt[LANG_MASKS]   # [B, L_i]
#         B, N, L = tok.shape

#         if L < max_L:
#             pad_tok = tok.new_full((B, N, max_L - L), pad_id, dtype=tok.dtype)
#             pad_msk = msk.new_zeros((B, N, max_L - L), dtype=msk.dtype)
#             tok = torch.cat([tok, pad_tok], dim=-1)
#             msk = torch.cat([msk, pad_msk], dim=-1)

#         bt[LANG_TOKENS] = tok
#         bt[LANG_MASKS]  = msk

#         new_dp = type(dp)(batch=bt, non_tensor_batch=dp.non_tensor_batch,)
#         out.append(new_dp)
#     return out

def pad_dataprotos_lang(
    dp_list,
    pad_id: int,
    pad_to: int | None = None,
    LANG_TOKENS: str = "lang_tokens",
    LANG_MASKS:  str = "lang_masks",
):
    """
    左填充到 max_L（或 pad_to）。若某样本长度 L > max_L，则右对齐截断保留尾部（最后 max_L 个 token）。
    形状约定：tok/msk 为 [B, N, L]，mask 的 1 表示有效，0 表示 PAD。
    """
    # 收集最大长度
    lengths = [dp.batch[LANG_TOKENS].shape[-1] for dp in dp_list]
    max_L = max(lengths) if pad_to is None else int(pad_to)

    out = []
    for dp in dp_list:
        bt = dp.batch.clone()
        tok = bt[LANG_TOKENS]  # [B, N, L]
        msk = bt[LANG_MASKS]   # [B, N, L]
        B, N, L = tok.shape

        if L == max_L:
            # 不变
            pass

        elif L < max_L:
            # 左填充：在左侧补 PAD / 0，使得右侧与原序列对齐
            pad_len = max_L - L
            pad_tok = torch.full((B, N, pad_len), pad_id, dtype=tok.dtype, device=tok.device)
            pad_msk = torch.zeros((B, N, pad_len), dtype=msk.dtype, device=msk.device)
            tok = torch.cat([pad_tok, tok], dim=-1)
            msk = torch.cat([pad_msk, msk], dim=-1)

        else:  # L > max_L
            # 右对齐截断：保留序列尾部（与左填充对齐的语义一致）
            tok = tok[..., L - max_L :]
            msk = msk[..., L - max_L :]

        bt[LANG_TOKENS] = tok
        bt[LANG_MASKS]  = msk

        new_dp = type(dp)(batch=bt, non_tensor_batch=getattr(dp, "non_tensor_batch", None))
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

def _env_entry(task_name, t_id, tr_id, in_state, config, input_q, output_q, is_valid, global_steps, max_steps):
    # 无头渲染（robosuite/mujoco）
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("DISPLAY", None)

    # 只在子进程里触碰 CUDA
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)                 # Ray 下本 actor 的第一个可见 GPU
        torch.backends.cudnn.benchmark = False   # 可选：稳定性
        torch.backends.cudnn.deterministic = False

    # 延迟导入/创建 env & 模型，避免父进程提前初始化 CUDA
    return env_worker_smolvla(
        task_name, t_id, tr_id, in_state, config, input_q, output_q, is_valid, global_steps, max_steps
    )
    
def _env_entry_vlm_reward(task_name, t_id, tr_id, in_state, config, input_q, output_q, is_valid, global_steps, max_steps):
    # 无头渲染（robosuite/mujoco）
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("DISPLAY", None)

    # 只在子进程里触碰 CUDA
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)                 # Ray 下本 actor 的第一个可见 GPU
        torch.backends.cudnn.benchmark = False   # 可选：稳定性
        torch.backends.cudnn.deterministic = False

    # 延迟导入/创建 env & 模型，避免父进程提前初始化 CUDA
    return env_worker_smolvla_vlm_reward(
        task_name, t_id, tr_id, in_state, config, input_q, output_q, is_valid, global_steps, max_steps
    )  

SUCCESS_VALUE_THRESH = 95.0      # 认为成功的 value 阈值
NO_PROGRESS_M = 500                # 连续 m 步 Δvalue <= 0 判定为无进展
BETA = 0.05                      # r_t = BETA * Δvalue 的缩放
CLIP_CRITIC_MIN = -90.0          # critic 数值裁剪下界，避免 -100 邻域不稳定
CLIP_CRITIC_MAX = 100.0          # critic 数值裁剪上界
CLIP_VALUE = True                # 是否把 value clip 到 [0, 100]

class RobHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, action_head, noisy_action_projector, config):
        super().__init__()
        self.config = config
        self.module = module
        self.action_head = action_head
        self.noisy_action_projector = noisy_action_projector
        self.max_steps = {"libero_spatial": 300,   # max step length 193
                            "libero_object": 512,    # max step length 254
                            "libero_goal": 512,      # max step length 270
                            # "libero_10": 1024,        # max step length 505
                            "libero_10": 560,        # max step length 505
                            "libero_90": 512,         # max step length 373 org 400 now change to 512
                            "bridge_orig": 30,
                        }
        if self.config.vla in ["smolvla", 'pi05']:
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        self.vla_preprocess()
        # breakpoint()
        self._rank = torch.distributed.get_rank(
        ) if torch.distributed.is_initialized() else 0
        self._num_gpus_per_node = config.n_gpus_per_node

        # self.num_workers = self.config.get('num_env_workers', 8)
        if config.unnorm_key not in ['bridge_orig']:
            self.num_workers = 16
            self.adapter = LIBEROAdapter(
                task_suite_name=self.config.task_suite_name,
                num_envs=self.num_workers,
                max_steps=self.max_steps[self.config.task_suite_name],
                num_steps_wait=self.config.num_steps_wait,
                model_family=self.config.model_family,
                gpu_ids=[self._rank % self._num_gpus_per_node], # Run all workers on the same assigned GPU
                # delta_action=self.config.delta_action
                flip=True if self.config.vla == "smolvla" else False,
            )
        #oft add
        # breakpoint()
        if noisy_action_projector is not None:
            unnorm_key=config.unnorm_key
            if  unnorm_key not in self.module.norm_stats and f"{unnorm_key}_no_noops" in self.module.norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"
            assert unnorm_key in self.module.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
            self.config.unnorm_key = unnorm_key
        #add end
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     for gpu in gpus:  
        #         tf.config.experimental.set_memory_growth(gpu, True)
    
    def close(self):
        """Gracefully shuts down the environment adapter."""
        if hasattr(self, 'adapter') and self.adapter:
            self.adapter.close()

    def __del__(self):
        # Ensure workers are closed when the object is garbage collected
        self.close()

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
        # breakpoint()
        if self.config.vla == "smolvla":
            if self.use_world_model and is_train:
                if self.config.reward_type == 'vlm':
                    output = [self._generate_minibatch_smolvla_wm(p) for p in batch_prompts]
                elif self.config.reward_type == 'vlac':  # use_reward_model
                    output = [self._generate_minibatch_smolvla_wm_vlac(p) for p in batch_prompts]
            elif self.config.reward_type == 'vlm' and is_train:
                output = [self._generate_minibatch_smolvla_vlm_reward(p) for p in batch_prompts]
            elif self.config.reward_type == 'vlac' and is_train:
                output = [self._generate_minibatch_smolvla_vlac_reward(p) for p in batch_prompts]
            elif self.config.only_for_gen_rm_data and (not is_train):
                output = [self._generate_minibatch_smolvla_vlm_reward_gendata(p) for p in batch_prompts]
            else:
                output = [self._generate_minibatch_smolvla(p) for p in batch_prompts]
                # output = [self._generate_minibatch_smolvla_vlm_reward(p) for p in batch_prompts]
            output = pad_dataprotos_lang(output, pad_id=self.module.language_tokenizer.pad_token_id, pad_to=None)
        elif self.config.vla == "pi05":
            if self.use_world_model and is_train:
                if self.config.reward_type == 'vlm':
                    output = [self._generate_minibatch_pi05_wm_bridge(p) for p in batch_prompts]
                else:
                    raise
            else:
                raise
            output = pad_dataprotos_lang(output, pad_id=2, pad_to=None)
        elif self.config.vla == "vla-adapter":
            if self.use_world_model and is_train:
                if self.config.unnorm_key in ['bridge_orig']:
                    output = [self._generate_minibatch_vla_adapter_wm_bridge(p) for p in batch_prompts]
                else:
                    if self.env_rollout:
                        output = [self._generate_minibatch_vla_adapter_wm_env_rollout(p) for p in batch_prompts]
                    else:
                        output = [self._generate_minibatch_vla_adapter_wm(p) for p in batch_prompts]
            elif self.config.reward_type == 'vlm' and is_train:
                output = [self._generate_minibatch_vla_adapter_vlm_reward(p) for p in batch_prompts]
            elif self.config.only_for_gen_rm_data and (not is_train):
                output = [self._generate_minibatch_vla_adapter_vlm_reward_gendata(p) for p in batch_prompts]
            else:
                output = [self._generate_minibatch_vla_adapter(p) for p in batch_prompts]
            output = pad_dataprotos_lang(
                output, 
                pad_id=self.processor.tokenizer.pad_token_id, 
                pad_to=None, 
                LANG_TOKENS="input_ids", 
                LANG_MASKS="attention_mask",
            )
        elif self.config.vla == "openvla-oft-flow":
            if self.use_world_model and is_train:
                if self.config.unnorm_key in ['bridge_orig']:
                    output = [self._generate_minibatch_openvla_oft_flow_wm_bridge(p) for p in batch_prompts]
                else:
                    raise
                    # if self.env_rollout:
                    #     output = [self._generate_minibatch_vla_adapter_wm_env_rollout(p) for p in batch_prompts]
                    # else:
                    #     output = [self._generate_minibatch_vla_adapter_wm(p) for p in batch_prompts]
            else:
                raise
            # elif self.config.reward_type == 'vlm' and is_train:
            #     output = [self._generate_minibatch_vla_adapter_vlm_reward(p) for p in batch_prompts]
            # elif self.config.only_for_gen_rm_data and (not is_train):
            #     output = [self._generate_minibatch_vla_adapter_vlm_reward_gendata(p) for p in batch_prompts]
            # else:
            #     output = [self._generate_minibatch_vla_adapter(p) for p in batch_prompts]
            output = pad_dataprotos_lang(
                output, 
                pad_id=self.processor.tokenizer.pad_token_id, 
                pad_to=None, 
                LANG_TOKENS="input_ids", 
                LANG_MASKS="attention_mask",
            )
                
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
    
    def process_input_pi(self, inputs:list, task_descriptions:list):
        
        batchdata = {"observation/image":[], 
                     "observation/image_is_pad": [],
                     "prompt": []}  
        if "state" in inputs[0].keys():
            batchdata["observation/state"] = []
            batchdata["observation/state_is_pad"] = []
        # if "wrist_image" in inputs[0].keys():
        #     batchdata["observation.images.wrist_image"] = []
        #     batchdata["observation.images.wrist_image_is_pad"] = []
            
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
            batchdata["prompt"].append(task_description)
           
            # image = Image.fromarray(input["full_image"]).convert("RGB")
            # if self.config.center_crop:
            #     image = center_crop_image(image)
            # image_tensor = transform(image).to(torch.bfloat16)
            # batchdata["observation/image"].append(image_tensor)
            
            image = torch.from_numpy(input["full_image"])
            batchdata["observation/image"].append(image)
            batchdata["observation/image_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
            
                
            # if "wrist_image" in input.keys():
            #     wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
            #     if self.config.center_crop:
            #         wrist_image = center_crop_image(wrist_image)
            #     wrist_image_tensor = transform(wrist_image).to(torch.bfloat16)
            #     batchdata["observation.images.wrist_image"].append(wrist_image_tensor) 
            #     batchdata["observation.images.wrist_image_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
            
            if "state" in input.keys():
                state_tensor = torch.tensor(input['state']).unsqueeze(0).to(torch.bfloat16)
                batchdata["observation/state"].append(state_tensor)
                batchdata["observation/state_is_pad"].append(torch.tensor([0.]).to(torch.bfloat16))
        
        device = torch.device('cuda') 
        
        batchdata["observation/image"] = torch.stack([x for x in batchdata["observation/image"]])  # .to(device)
        batchdata["observation/image_is_pad"] = torch.stack([x for x in batchdata["observation/image_is_pad"]]).to(device)
        if "state" in input.keys():
            batchdata["observation/state"] = torch.stack([x for x in batchdata["observation/state"]]).to(device)
            batchdata["observation/state_is_pad"] = torch.stack([x for x in batchdata["observation/state_is_pad"]]).to(device)
        # if "wrist_image" in input.keys():
        #     batchdata["observation.images.wrist_image"] = torch.stack([x for x in batchdata["observation.images.wrist_image"]]).to(device)
        #     batchdata["observation.images.wrist_image_is_pad"] = torch.stack([x for x in batchdata["observation.images.wrist_image_is_pad"]]).to(device)
        
        return batchdata
   
    def process_input_vla_adapter(self,inputs:list, task_descriptions:list):
        # breakpoint()
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        # breakpoint()
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            # prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_description.lower()}?<|im_end|>\n<|im_start|>assistant\n'
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
            
            # if not torch.all(input_ids[:, -1] == 29871):
            #     input_ids = torch.cat(
            #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            #     )
            #     if self.config.vla in ["openvla-oft", "vla-adapter"]:
            #         attention_mask = torch.cat(
            #             (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
            #         )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft", "vla-adapter"]:
        # if False:
            # breakpoint()
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
    
    def process_input_openvla_oft_flow(self,inputs:list, task_descriptions:list):
        # breakpoint()
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        # breakpoint()
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            # prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_description.lower()}?<|im_end|>\n<|im_start|>assistant\n'
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
            
            # if not torch.all(input_ids[:, -1] == 29871):
            #     input_ids = torch.cat(
            #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            #     )
            #     if self.config.vla in ["openvla-oft", "vla-adapter"]:
            #         attention_mask = torch.cat(
            #             (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
            #         )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft-flow", "vla-adapter"]:
        # if False:
            # breakpoint()
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
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
        # breakpoint()

        # This is a blocking call
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy()
        )
        
        
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], is_train, flip=True))
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
            # breakpoint()
            
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
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
            })
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], is_train, flip=True)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
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
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
            
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        
        # ctx = mp.get_context("spawn")

        # processes = []
        # input_queues = []
        # output_queues = []

        # for idx in range(batch_size):
        #     task_name = task_suite_name[idx]

        #     # 这些如果是 torch 张量，转成 Python 标量更稳妥
        #     t_id = int(task_id[idx][0].item())
        #     tr_id = int(trial_id[idx][0].item())

        #     # 彻底拷贝到普通 CPU 内存，避免继承父进程的 pinned 区域
        #     in_state = (init_state[idx]
        #                 .detach()
        #                 .cpu()
        #                 .contiguous()
        #                 .numpy()
        #                 .copy())

        #     input_q = ctx.Queue()
        #     output_q = ctx.Queue()

        #     p = ctx.Process(
        #         target=_env_entry_vlm_reward,
        #         args=(task_name, t_id, tr_id, in_state, self.config, input_q, output_q, is_valid, global_steps, max_steps),
        #         daemon=True,   # 可选：随父进程退出
        #     )
        #     p.start()

        #     processes.append(p)
        #     input_queues.append(input_q)
        #     output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], flip=True))
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
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
                },
                use_vlm_rm=True,
            )
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                # result = output_queues[idx].get(timeout=30)
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], flip=True)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete_raw'] = result['complete']
                task_records[idx]['finish_step_raw'] = result['finish_step']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        # for q in input_queues:
        #     q.put(None)
        # for p in processes:
        #     p.join(timeout=20)
        #     if p.is_alive():
        #         p.terminate()
        torch.cuda.empty_cache()
        # breakpoint()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
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
        
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        batch["step_images"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            batch["step_images"].append(np.stack(k["step_images"]))
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        batch["step_images"] = padded_step_images.to(device=batch['observation.images.image'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['observation.images.image'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_vla_adapter_vlm_reward(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
            
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], flip=False))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "step_images": [init_data['obs']['agentview_image'][::-1, ::-1]]
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
           
            
            vla_input = self.process_input_vla_adapter(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train)
            actions = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in current_inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
                },
                use_vlm_rm=True,
            )
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                # result = output_queues[idx].get(timeout=30)
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], flip=False)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete_raw'] = result['complete']
                task_records[idx]['finish_step_raw'] = result['finish_step']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        # for q in input_queues:
        #     q.put(None)
        # for p in processes:
        #     p.join(timeout=20)
        #     if p.is_alive():
        #         p.terminate()
        torch.cuda.empty_cache()
        # breakpoint()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
                )
        self.module.train()
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        batch["step_images"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            batch["step_images"].append(np.stack(k["step_images"]))
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['action_tensor'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        padded_step_images = padded_step_images.flip(dims=[3])
        batch["step_images"] = padded_step_images.to(device=batch['action_tensor'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['action_tensor'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_smolvla_vlm_reward_gendata(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
            
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
    
        traj_buf = [
            dict(
                task_file_name=None,
                agentview_images=[],
                eye_in_hand_images=[],
                gripper_qpos=[],
                joint_pos=[],
                eef_pos=[],
                eef_quat=[],
                sim_state=[],
                actions_applied=[],   # 每 env step 的实际动作
                complete_raw=False,
                finish_step_raw=None,
            )
            for _ in range(batch_size)
        ]
        


        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], flip=True))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "step_images": [init_data['obs']['agentview_image'][::-1]]
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
            traj_buf[idx]["task_file_name"] = init_data['task_file_name']
        
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
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
                },
                use_vlm_rm=True,
            )
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                # result = output_queues[idx].get(timeout=30)
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], flip=True)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete_raw'] = result['complete']
                task_records[idx]['finish_step_raw'] = result['finish_step']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                # breakpoint()
                ch = result.get("traj_chunk", None)
                if ch is not None:
                    # 拼接各序列
                    traj_buf[idx]["agentview_images"].extend(ch.get("agentview_images", []))
                    traj_buf[idx]["eye_in_hand_images"].extend(ch.get("eye_in_hand_images", []))
                    traj_buf[idx]["gripper_qpos"].extend(ch.get("gripper_qpos", []))
                    traj_buf[idx]["joint_pos"].extend(ch.get("joint_pos", []))
                    traj_buf[idx]["eef_pos"].extend(ch.get("eef_pos", []))
                    traj_buf[idx]["eef_quat"].extend(ch.get("eef_quat", []))
                    traj_buf[idx]["sim_state"].extend(ch.get("sim_state", []))
                    traj_buf[idx]["actions_applied"].extend(ch.get("actions_applied", []))
                # 记录完成信息
                traj_buf[idx]["complete_raw"] = bool(result['complete'])
                traj_buf[idx]["finish_step_raw"] = int(result['finish_step'])
                
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        torch.cuda.empty_cache()
        # breakpoint()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
                )
                
            target_root = getattr(self.config, "libero_target_dir", "./LIBERO/libero/datasets/generated")
            os.makedirs(target_root, exist_ok=True)

            for idx in range(batch_size):
                tb = traj_buf[idx]
                # 任务名解析
                base = os.path.splitext(os.path.basename(tb["task_file_name"] or "unknown_task"))[0]
                if base.endswith("_demo"):
                    task_name = base
                elif base.endswith("_demo.hdf5"):
                    task_name = base[:-5]
                else:
                    task_name = base + "_demo"
                # breakpoint()
                h5_path = os.path.join(target_root, f"{task_name}.hdf5")

                # 把 list -> np.array，并做必要转换/兜底
                def _stack(name, allow_empty=False, default=None):
                    seq = tb.get(name, [])
                    if len(seq) == 0:
                        if allow_empty:
                            return default
                        raise ValueError(f"Empty sequence for {name} (idx={idx})")
                    return np.stack(seq, axis=0)

                agentview_rgb = _stack("agentview_images")  # (T,H,W,3)
                eye_in_hand_rgb = _stack("eye_in_hand_images", allow_empty=True, default=None)
                if eye_in_hand_rgb is None:
                    eye_in_hand_rgb = np.zeros_like(agentview_rgb)

                gripper_states = _stack("gripper_qpos")
                joint_states   = _stack("joint_pos")
                ee_pos         = _stack("eef_pos")
                ee_quat        = _stack("eef_quat")

                # quat(wxyz)->axis-angle(3)
                ee_ori_axisangle = np.stack([T.quat2axisangle(q) for q in ee_quat], axis=0).astype(np.float32)

                actions_applied = _stack("actions_applied")  # 真正送入 env 的动作（建议存这个）
                robot_states = np.concatenate([
                    gripper_states,
                    ee_pos,
                    ee_quat,
                ], axis=-1).astype(np.float32)

                # states：优先 sim_state；若无，用 robot_states 兜底
                try:
                    states = _stack("sim_state")
                except Exception:
                    states = robot_states

                Tlen = actions_applied.shape[0]
                rewards = np.zeros((Tlen,), dtype=np.uint8)
                dones   = np.zeros((Tlen,), dtype=np.uint8)
                fs = tb.get("finish_step_raw", None)
                if isinstance(fs, (int, np.integer)) and 0 <= fs < Tlen:
                    rewards[fs] = 1; dones[fs] = 1
                else:
                    rewards[-1] = 1; dones[-1] = 1

                # 旋转 180° 与再生脚本对齐；并转 uint8
                agentview_rgb   = as_uint8(rotate_180(agentview_rgb))
                eye_in_hand_rgb = as_uint8(rotate_180(eye_in_hand_rgb))

                # 写 H5
                f = open_or_create_h5(h5_path)
                try:
                    # breakpoint()
                    demo_i = len(f["data"])
                    g = f["data"].create_group(f"demo_{demo_i}")
                    obs = g.create_group("obs")

                    h5_create_dset(obs, "gripper_states", gripper_states.astype(np.float32, copy=False))
                    h5_create_dset(obs, "joint_states",   joint_states.astype(np.float32, copy=False))
                    ee_states = np.hstack((ee_pos, ee_ori_axisangle)).astype(np.float32, copy=False)
                    h5_create_dset(obs, "ee_states",      ee_states)
                    h5_create_dset(obs, "ee_pos",         ee_pos.astype(np.float32, copy=False))
                    h5_create_dset(obs, "ee_ori",         ee_ori_axisangle.astype(np.float32, copy=False))
                    h5_create_dset(obs, "agentview_rgb",  agentview_rgb,  img_like=True)
                    h5_create_dset(obs, "eye_in_hand_rgb",eye_in_hand_rgb,img_like=True)
                    
                    h5_create_dset(g, "actions",      actions_applied.astype(np.float32, copy=False))
                    h5_create_dset(g, "states",       states.astype(np.float32, copy=False))
                    
                    h5_create_dset(g, "robot_states", robot_states.astype(np.float32, copy=False))
                    h5_create_dset(g, "rewards",      rewards)
                    h5_create_dset(g, "dones",        dones)
                    try:
                        h5_create_dset(g, "env_states",        init_state[idx].cpu().numpy().astype(np.float32, copy=False))
                    except:
                        breakpoint()
                finally:
                    f.close()
        # breakpoint()
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
        
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        batch["step_images"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            batch["step_images"].append(np.stack(k["step_images"]))
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        batch["step_images"] = padded_step_images.to(device=batch['observation.images.image'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['observation.images.image'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_vla_adapter_vlm_reward_gendata(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
            
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
    
        traj_buf = [
            dict(
                task_file_name=None,
                agentview_images=[],
                eye_in_hand_images=[],
                gripper_qpos=[],
                joint_pos=[],
                eef_pos=[],
                eef_quat=[],
                sim_state=[],
                actions_applied=[],   # 每 env step 的实际动作
                complete_raw=False,
                finish_step_raw=None,
            )
            for _ in range(batch_size)
        ]
        


        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], flip=False))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "step_images": [init_data['obs']['agentview_image'][:, ::-1]]
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
            traj_buf[idx]["task_file_name"] = init_data['task_file_name']
        
        step = 0
        vla_history = []
        while step < max_steps:
            # active_indices = [i for i, r in enumerate(task_records) if r['active']]
            active_indices = [i for i, r in enumerate(task_records)]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
           
            
            vla_input = self.process_input_vla_adapter(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train)
            actions = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)

            vla_history.append(step_data)
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
                },
                use_vlm_rm=True,
            )
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                # result = output_queues[idx].get(timeout=30)
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], flip=False)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete_raw'] = result['complete']
                task_records[idx]['finish_step_raw'] = result['finish_step']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                result['valid_images'] = [img[:, ::-1] for img in result['valid_images']]
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                ch = result.get("traj_chunk", None)
                if ch is not None:
                    # agentview = [img[:, ::-1] for img in result['traj_chunk']['agentview_images']]
                    # 拼接各序列
                    traj_buf[idx]["agentview_images"].extend(result['traj_chunk']['agentview_images'])
                    # traj_buf[idx]["eye_in_hand_images"].extend(ch.get("eye_in_hand_images", []))
                    traj_buf[idx]["gripper_qpos"].extend(ch.get("gripper_qpos", []))
                    traj_buf[idx]["joint_pos"].extend(ch.get("joint_pos", []))
                    traj_buf[idx]["eef_pos"].extend(ch.get("eef_pos", []))
                    traj_buf[idx]["eef_quat"].extend(ch.get("eef_quat", []))
                    traj_buf[idx]["sim_state"].extend(ch.get("sim_state", []))
                    traj_buf[idx]["actions_applied"].extend(ch.get("actions_applied", []))
                # 记录完成信息
                traj_buf[idx]["complete_raw"] = bool(result['complete'])
                traj_buf[idx]["finish_step_raw"] = int(result['finish_step'])
                
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        torch.cuda.empty_cache()
        
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
                )
                
            target_root = getattr(self.config, "libero_target_dir", "./LIBERO/libero/datasets/generated")
            os.makedirs(target_root, exist_ok=True)
            for idx in range(batch_size):
                tb = traj_buf[idx]
                # 任务名解析
                base = os.path.splitext(os.path.basename(tb["task_file_name"] or "unknown_task"))[0]
                if base.endswith("_demo"):
                    task_name = base
                elif base.endswith("_demo.hdf5"):
                    task_name = base[:-5]
                else:
                    task_name = base + "_demo"
                # breakpoint()
                h5_path = os.path.join(target_root, f"{task_name}.hdf5")

                # 把 list -> np.array，并做必要转换/兜底
                def _stack(name, allow_empty=False, default=None):
                    seq = tb.get(name, [])
                    if len(seq) == 0:
                        if allow_empty:
                            return default
                        raise ValueError(f"Empty sequence for {name} (idx={idx})")
                    return np.stack(seq, axis=0)

                agentview_rgb = _stack("agentview_images")  # (T,H,W,3)
                eye_in_hand_rgb = _stack("eye_in_hand_images", allow_empty=True, default=None)
                if eye_in_hand_rgb is None:
                    eye_in_hand_rgb = np.zeros_like(agentview_rgb)

                gripper_states = _stack("gripper_qpos")
                joint_states   = _stack("joint_pos")
                ee_pos         = _stack("eef_pos")
                ee_quat        = _stack("eef_quat")

                # quat(wxyz)->axis-angle(3)
                ee_ori_axisangle = np.stack([T.quat2axisangle(q) for q in ee_quat], axis=0).astype(np.float32)

                actions_applied = _stack("actions_applied")  # 真正送入 env 的动作（建议存这个）
                robot_states = np.concatenate([
                    gripper_states,
                    ee_pos,
                    ee_quat,
                ], axis=-1).astype(np.float32)

                # states：优先 sim_state；若无，用 robot_states 兜底
                try:
                    states = _stack("sim_state")
                except Exception:
                    states = robot_states

                Tlen = actions_applied.shape[0]
                rewards = np.zeros((Tlen,), dtype=np.uint8)
                dones   = np.zeros((Tlen,), dtype=np.uint8)
                fs = tb.get("finish_step_raw", None)
                if isinstance(fs, (int, np.integer)) and 0 <= fs < Tlen:
                    rewards[fs] = 1; dones[fs] = 1
                else:
                    rewards[-1] = 1; dones[-1] = 1

                # 旋转 180° 与再生脚本对齐；并转 uint8
                agentview_rgb   = as_uint8(agentview_rgb[:, ::-1, ::-1])
                # eye_in_hand_rgb = as_uint8(rotate_180(eye_in_hand_rgb))

                f = open_or_create_h5(h5_path)
                try:
                    # breakpoint()
                    demo_i = len(f["data"])
                    g = f["data"].create_group(f"demo_{demo_i}")
                    obs = g.create_group("obs")

                    h5_create_dset(obs, "gripper_states", gripper_states.astype(np.float32, copy=False))
                    h5_create_dset(obs, "joint_states",   joint_states.astype(np.float32, copy=False))
                    ee_states = np.hstack((ee_pos, ee_ori_axisangle)).astype(np.float32, copy=False)
                    h5_create_dset(obs, "ee_states",      ee_states)
                    h5_create_dset(obs, "ee_pos",         ee_pos.astype(np.float32, copy=False))
                    h5_create_dset(obs, "ee_ori",         ee_ori_axisangle.astype(np.float32, copy=False))
                    h5_create_dset(obs, "agentview_rgb",  agentview_rgb,  img_like=True)
                    # h5_create_dset(obs, "eye_in_hand_rgb",eye_in_hand_rgb,img_like=True)
                    
                    h5_create_dset(g, "actions",      actions_applied.astype(np.float32, copy=False))
                    h5_create_dset(g, "states",       states.astype(np.float32, copy=False))
                    
                    h5_create_dset(g, "robot_states", robot_states.astype(np.float32, copy=False))
                    h5_create_dset(g, "rewards",      rewards)
                    h5_create_dset(g, "dones",        dones)
                    try:
                        h5_create_dset(g, "env_states",        init_state[idx].cpu().numpy().astype(np.float32, copy=False))
                    except:
                        breakpoint()
                finally:
                    f.close()
        # breakpoint()
        self.module.train()
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        batch["step_images"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            batch["step_images"].append(np.stack(k["step_images"]))
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['action_tensor'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        padded_step_images = padded_step_images.flip(dims=[3])
        batch["step_images"] = padded_step_images.to(device=batch['action_tensor'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['action_tensor'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _build_chunk_pairs_and_map(self, task_records, task_descriptions, chunk_len):
        """
        把本 chunk 的所有 env 的相邻帧对打平，返回 pairs 与 map，供批量 RM 推理。
        约定：用每步的“末帧”代表该步的后状态：
        - 对于每个 env：我们取它的 step_images 中“上一 chunk 的末帧” + “本 chunk 的 20 张末帧”，
            形成 20 个相邻对：(prev_last, img0), (img0, img1), ..., (img18, img19)
        """
        pairs = []
        pair_task_texts = []
        pair_map = []  # (env_idx, local_step_in_chunk)

        for env_idx, rec in enumerate(task_records):
            imgs = rec['step_images']
            # 确保本 chunk 有 chunk_len 张新帧已 append
            # 假设 step_images 结构：开头 init 1 帧 + 每个 step append 1 张末帧
            # 则本 chunk 结束后，新增了 chunk_len 张；上一 chunk 的末帧 = imgs[-chunk_len-1]
            if len(imgs) < 2:
                continue  # 还没积累到帧
            if rec.get('active', True) is False:
                continue  # 已经停了的不参与
            if rec.get('complete', False) is True:
                continue

            # 最近 chunk 的 20 张末帧
            recent = imgs[-chunk_len:]             # 长度 chunk_len
            prev_last = imgs[-chunk_len-1]         # 前一帧（上一 chunk 的末帧）

            # 组 20 个相邻对
            cur_pairs = [(prev_last, recent[0])]
            for j in range(1, len(recent)):
                cur_pairs.append((recent[j-1], recent[j]))

            # 打平到全局 pairs
            task_text = task_descriptions[env_idx]
            for j, (img_t, img_tp1) in enumerate(cur_pairs):
                pairs.append({"img_t": img_t, "img_tp1": img_tp1})
                pair_task_texts.append(task_text)
                pair_map.append((env_idx, j))  # j ∈ [0, chunk_len-1]

        return pairs, pair_task_texts, pair_map

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
            imgs = rec.get('step_images', [])
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
        task_records, 
        pair_map, 
        critic_chunks,
        beta: float = 0.05,
        success_window: int = 2,
        max_steps: int = 500,
    ):
        """
        输入：
        - pair_map: [(env_idx, start_step, seg_len), ...]
        - critic_chunks: 与 pair_map 对齐的段级critic（百分比，正进负退）
        过程：
        1) 把每个段级 critic 按 seg_len 均匀摊到段内每一步的 per-step critic
        2) 对每个 env 的 per-step critic（长度 T）累计成 value，并计算 Δvalue
        3) 写回 rec['critic'] / rec['value'] / rec['delta'] / rec['reward']，并标记 complete/active
        """
        # 先聚合：为每个 env 准备 per-step critic 容器
        per_env_step_critic = {}  # env_idx -> list[length T]，T由该env的 step_images 决定
        for env_idx, rec in enumerate(task_records):
            imgs = rec.get('step_images', [])
            T = max(len(imgs) - 1, 0)
            per_env_step_critic[env_idx] = [0.0] * T

        # 1) 段级 critic 均匀摊回每步
        for (env_idx, start, seg_len), c in zip(pair_map, critic_chunks):
            if seg_len <= 0:
                continue
            contrib = float(c) / float(seg_len)
            
            step_list = per_env_step_critic.get(env_idx, None)
            if step_list is None:
                continue
            for t in range(start, start + seg_len):
                if 0 <= t < len(step_list):
                    step_list[t] += contrib

        # 2) 累计每个 env 的 value/Δvalue，并写 reward
        for env_idx, step_critic in per_env_step_critic.items():
            rec = task_records[env_idx]
            T = len(step_critic)
            if T == 0:
                # 没有步就跳过
                rec['critic'] = []
                rec['delta'] = []
                rec['value'] = [0.0]
                rec['reward'] = []
                rec['active'] = False
                continue

            v0 = 0.0
            value, deltas = self._accumulate_value_delta(step_critic, v0=v0)
            rewards = [beta * d for d in deltas]
            
            rec['critic'] = step_critic 
            rec['delta'] = deltas 
            rec['value'] = value 
            rec['reward'] = rewards
            
            finish_step = None
            consec = 0
            # 遍历 1..T（value 的索引），i 对应“完成第 i 步后的进度”
            for i in range(1, T + 1):
                if value[i] >= SUCCESS_VALUE_THRESH:
                    consec += 1
                else:
                    consec = 0
                if consec >= success_window:
                    # 第一次满足窗口长度时，finish_step = i（即原始第 i 步）
                    finish_step = i
                    break

            if finish_step is not None:
                rec['complete'] = True
                rec['finish_step'] = int(finish_step)
            else:
                rec['complete'] = False
                # 若希望把最后一步记为 finish_step（未成功但到头），可如下设置；否则置 0/None 皆可
                rec['finish_step'] = max_steps

            # 整段评估完成，统一置 inactive
            rec['active'] = False
            
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

    def _scatter_progress_back_and_update(self, task_records, pair_map, progress_list, chunk_len):
        """
        按 pair_map 回填 progress，累计成 value/Δvalue，写入 task_records，并据此更新 active/complete。
        假设每个 env 我们只关心“本 chunk 的 20 个 critic”，用 env 的 last_value 作为 v0。
        """
        # 先按 env 聚合
        env_chunk_critic = {}  # env_idx -> list length chunk_len
        for (env_idx, j), prog in zip(pair_map, progress_list):
            env_chunk_critic.setdefault(env_idx, [None]*chunk_len)
            env_chunk_critic[env_idx][j] = float(prog)

        for env_idx, critic_seq in env_chunk_critic.items():
            # 初始化 per-env 的缓存槽位
            rec = task_records[env_idx]
            if 'critic' not in rec: rec['critic'] = []
            if 'value'  not in rec: rec['value']  = [0.0]   # 从 0 开始
            if 'delta'  not in rec: rec['delta']  = []
            if 'no_progress_count' not in rec: rec['no_progress_count'] = 0

            # 用上一时刻 value 的最后一个作为 v0
            v0 = rec['value'][-1]

            # 计算本 chunk 的 value/Δvalue
            value_seq, delta_seq = self._accumulate_value_delta(critic_seq, v0=v0)

            # 追加到全局（注意 value_seq 比 critic_seq 多 1 个）
            rec['critic'].extend(critic_seq)
            rec['delta'].extend(delta_seq)
            rec['value'].extend(value_seq[1:])  # 去掉 v0，只追加新 20 个

            # 成功/早停判断
            # 规则1：若本 chunk 末 value >= 阈值，则标成功并 inactive
            if rec['value'][-1] >= SUCCESS_VALUE_THRESH:
                rec['complete'] = True
                rec['active'] = False
                rec['finish_step'] = len(rec['critic'])   # 累计步数（近似）
                continue

            # 规则2：若连续 m 步 Δvalue <= 0，则 early stop（无进展）
            # 只检查本 chunk 新增的 Δvalue
            for d in delta_seq:
                if d <= 0:
                    rec['no_progress_count'] += 1
                else:
                    rec['no_progress_count'] = 0

            if rec['no_progress_count'] >= NO_PROGRESS_M:
                rec['active'] = False  # 早停
                # 不标 complete，表示失败停
                # 可选：把 no_progress_count 归零
                rec['no_progress_count'] = 0

            # 也可以在这里顺便把逐步奖励写进 rec['reward']，供 GRPO/PPO 用
            if 'reward' not in rec: rec['reward'] = [0]
            rec['reward'].extend([BETA * d for d in delta_seq])
            
    def _generate_minibatch_smolvla_vlac_reward_step(self, prompts):
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
            
        ctx = mp.get_context("spawn")

        processes = []
        input_queues = []
        output_queues = []

        for idx in range(batch_size):
            task_name = task_suite_name[idx]

            # 这些如果是 torch 张量，转成 Python 标量更稳妥
            t_id = int(task_id[idx][0].item())
            tr_id = int(trial_id[idx][0].item())

            # 彻底拷贝到普通 CPU 内存，避免继承父进程的 pinned 区域
            in_state = (init_state[idx]
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                        .copy())

            input_q = ctx.Queue()
            output_q = ctx.Queue()

            p = ctx.Process(
                target=_env_entry_vlm_reward,
                args=(task_name, t_id, tr_id, in_state, self.config, input_q, output_q, is_valid, global_steps, max_steps),
                daemon=True,   # 可选：随父进程退出
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
                "active": True,
                "active_raw": init_data['active'],
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
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            # active_indices = [i for i, r in enumerate(task_records)]
            
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
                task_records[idx]['active_raw'] = result['active']
                task_records[idx]['complete_raw'] = result['complete_raw']
                task_records[idx]['finish_step_raw'] = result['finish_step_raw']
                task_records[idx]['step_images'].extend(result['valid_images']) 
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            # breakpoint()
            # compute_vlac_reward
            # self.reward_model.reward_step()
            chunk_len = self.config.action_chunks_len
            # 1) 组装 pairs（跨所有仍 active 的 env）
            pairs, pair_task_texts, pair_map = self._build_chunk_pairs_and_map(
                task_records, task_descriptions, chunk_len
            )
            
            if len(pairs) > 0:
                # 2) 批量 RM 推理 —— TODO: 用你的接口替换这行
                #    例如 progress = self.reward_model_predict(pairs, pair_task_texts)
                # progress = self.reward_model.reward_step(pairs, pair_task_texts)  # -> list/np.array, len == len(pairs)
                critic_list = self.reward_model.reward_step(pairs, pair_task_texts,
                                                        batch_num=256,
                                                        temperature=0.0,
                                                        top_k=None,
                                                        addition_scale=1.0,
                                                        divide_skip=1,
                                                        value_simple="simple",
                                                        related_critic=False,
                                                        return_value=False,
                                                        rich=False)

                # 3) 回填 + 累计 + 成功/早停判定 + 写逐步 reward
                self._scatter_progress_back_and_update(
                    task_records, pair_map, critic_list, chunk_len
                )
                # breakpoint()
                # 4) 根据更新后的 active 状态，决定下一轮 active_indices
                for idx in active_indices:
                    vals = task_records[idx].get('value', [])
                    last5 = vals[-5:] if len(vals) >= 5 else vals
                    if len(last5) >= 1 and all(v >= 95.0 for v in last5):  # 0-100 尺度
                        task_records[idx]['active'] = False
            else:
                pass

                
            
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        torch.cuda.empty_cache()
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
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
        
        breakpoint()
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        batch["step_images"] = []
        batch["critic"] = []
        batch["value"] = []
        batch["delta"] = []
        batch["reward"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            batch["step_images"].append(np.stack(k["step_images"]))
            batch["critic"].append(np.stack(k["critic"]))
            batch["value"].append(np.stack(k["value"]))
            batch["delta"].append(np.stack(k["delta"]))
            batch["reward"].append(np.stack(k["reward"]))
        
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        batch["step_images"] = padded_step_images.to(device=batch['observation.images.image'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['observation.images.image'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    

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
    
    def _generate_minibatch_smolvla_vlac_reward(self, prompts):
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
            
        ctx = mp.get_context("spawn")

        processes = []
        input_queues = []
        output_queues = []

        for idx in range(batch_size):
            task_name = task_suite_name[idx]

            # 这些如果是 torch 张量，转成 Python 标量更稳妥
            t_id = int(task_id[idx][0].item())
            tr_id = int(trial_id[idx][0].item())

            # 彻底拷贝到普通 CPU 内存，避免继承父进程的 pinned 区域
            in_state = (init_state[idx]
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                        .copy())

            input_q = ctx.Queue()
            output_q = ctx.Queue()

            p = ctx.Process(
                target=_env_entry_vlm_reward,
                args=(task_name, t_id, tr_id, in_state, self.config, input_q, output_q, is_valid, global_steps, max_steps),
                daemon=True,   # 可选：随父进程退出
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
                "active": True,
                "active_raw": init_data['active'],
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
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            # active_indices = [i for i, r in enumerate(task_records)]
            
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
                task_records[idx]['active_raw'] = result['active']
                task_records[idx]['complete_raw'] = result['complete_raw']
                task_records[idx]['finish_step_raw'] = result['finish_step_raw']
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
                complete = any(r['complete_raw'] for r in task_records if r['task_file_name'] == task_file)
                # breakpoint()
                finish_step = [r['finish_step_raw'] for r in task_records if r['task_file_name'] == task_file][0]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                    finish_step
                )
        
        # step_interval = getattr(self.config, "rm_step_interval", self.config.action_chunks_len)
        
        # ref_video_path_list = [Path(REF_DICT[task_name][task_id[i].item()]) for i in range(len(task_id))]
        # ref_frames_list = []
        # for ref_video_path in ref_video_path_list:
        #     ref_frames, ref_frame_indices = self.process_video_to_PIL_frames_with_indices(ref_video_path, num_frames=10)
        #     ref_frames_list.append(ref_frames)
        # pairs, pair_task_texts, pair_map = self._build_pairs_full_trajectory(
        #     task_records, task_descriptions, step_interval=step_interval, ref_frames_list=ref_frames_list, ref_num=10
        # )
        
        # if len(pairs) > 0:
    
        #     critic_chunks = self.reward_model.reward_step(
        #         pairs, pair_task_texts,
        #         use_ref=True,
        #         batch_num=256,
        #         addition_scale=1.0,
        #         divide_skip=1,
        #         related_critic=False,
        #         return_value=False,
        #         rich=False,
        #     )

        #     self._distribute_and_finalize_rewards(
        #         task_records, pair_map, critic_chunks, beta=0.05, success_window=2, max_steps=max_steps
        #     )
        # else:
        #     # 没有帧对就全置空
        #     for rec in task_records:
        #         rec['critic'] = []
        #         rec['delta']  = []
        #         rec['value']  = [0.0]
        #         rec['reward'] = []
        #         rec['active'] = False
    
                
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
        
        # breakpoint()
        batch["complete_raw"] = []
        # batch["complete"] = []
        batch["finish_step_raw"] = []
        # batch["finish_step"] = []
        batch["step_images"] = []
        # batch["critic"] = []
        # batch["value"] = []
        # batch["delta"] = []
        # batch["reward"] = []
        for k in task_records:
            batch["complete_raw"].append(k["complete_raw"])
            # batch["complete"].append(k["complete"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            # batch["finish_step"].append(k["finish_step"])
            batch["step_images"].append(np.stack(k["step_images"]))
            # batch["critic"].append(np.stack(k["critic"]))
            # batch["value"].append(np.stack(k["value"]))
            # batch["delta"].append(np.stack(k["delta"]))
            # batch["reward"].append(np.stack(k["reward"]))
        
            
        
        batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['observation.images.image'].device)
        # batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['observation.images.image'].device)
        # batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["complete"] = (torch.zeros_like(batch["complete_raw"]) == 1)
        batch["finish_step"] = torch.ones_like(batch["finish_step_raw"]) * max_steps
        # breakpoint()
        padded_step_images, padded_step_images_mask = pad_dataprotos_step_images(batch["step_images"])
        batch["step_images"] = padded_step_images.to(device=batch['observation.images.image'].device)
        batch["step_images_mask"] = padded_step_images_mask.to(device=batch['observation.images.image'].device)
        # batch["critic"] = torch.tensor(batch["critic"], dtype=torch.float32, device=batch['observation.images.image'].device)
        # batch["value"] = torch.tensor(batch["value"], dtype=torch.float32, device=batch['observation.images.image'].device)
        # batch["delta"] = torch.tensor(batch["delta"], dtype=torch.float32, device=batch['observation.images.image'].device)
        # batch["reward"] = torch.tensor(batch["reward"], dtype=torch.float32, device=batch['observation.images.image'].device)
        # breakpoint()
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
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        current_obs_batch_np_list = []
        for idx in range(len(init_data_list)):
            current_obs_batch_np_list.append(init_data_list[idx]['obs']['agentview_image'][::-1])
        current_obs_batch_np = np.stack(current_obs_batch_np_list)
            

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
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
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
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
                # r['step_images'].extend()
                                
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
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
            
        # breakpoint()
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
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['observation.images.image'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['observation.images.image'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_pi05_wm_bridge(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        current_obs_batch_np = init_state.cpu().numpy()
        self.world_model.reset(current_obs_batch_np)
            

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                # "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
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
                img = current_obs_batch_np[idx][::-1, :, :] 
                # valid_video[task_file].append(img)
        
        step = 0
        vla_history = []
        trajectory_video_batch = [current_obs_batch_np]
        vla_timings = []
        wm_timings = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            if not active_indices:
                break
            
            inputs = [self._obs_to_input(obs, flip=True) for obs in current_obs_batch_np]
            
            vla_input = self.process_input_pi(inputs, task_descriptions)
            vla_input.update(meta_info)
            step_data = vla_input.copy()
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_pi(vla_input, use_sde=is_train)
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            
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
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            # trajectory_video_batch.append(next_obs_batch_np[:, :, :, ::-1, :])
            trajectory_video_batch.append(next_obs_batch_np)

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :]  # [::-1, :, :]
                        # valid_video[task_file].append(img)
            
            for r in task_records:
                if r['active']:
                    r['finish_step'] = step
                    if r['finish_step'] >= max_steps:
                        r['active'] = False
                        r['complete'] = True 
                # r['step_images'].extend()
                                
        torch.cuda.empty_cache()
        # if is_valid:
        #     for task_file, images in valid_video.items():
        #         complete_flags = [r['complete'] for r in task_records if r['task_file_name'] == task_file]
        #         complete = complete_flags[0] if complete_flags else False
                
        #         save_rollout_video(
        #             images,
        #             self.config.experiment_name,
        #             task_file,
        #             global_steps,
        #             complete
        #         )
        initial_frame_expanded = np.expand_dims(trajectory_video_batch[0], axis=1) # -> (B, 1, H, W, C)
        video_chunks = [initial_frame_expanded] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
            
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
        batch = {"observation/image":[], 
                "observation/image_is_pad": [],
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
        for k in ["observation/image", "observation/image_is_pad",
                #   "observation.state", "observation.state_is_pad", 
                  "action_tensor", "x_t", "t", "x_next", "lang_tokens", "lang_masks"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['x_t'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['x_t'].device)
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['x_t'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['x_t'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_vla_adapter_wm(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        current_obs_batch_np_list = []
        for idx in range(len(init_data_list)):
            current_obs_batch_np_list.append(init_data_list[idx]['obs']['agentview_image'][::-1])
        current_obs_batch_np = np.stack(current_obs_batch_np_list)
        self.world_model.reset(current_obs_batch_np)

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
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
                img = current_obs_batch_np[idx][::-1, :, :] 
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
            
            vla_input = self.process_input_vla_adapter(inputs, task_descriptions)
            vla_input.update(meta_info)
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train)
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions_batch
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)
            if self.world_model is None:
                raise ValueError("World Model Worker Group has not been set!")
            
            with Timer(name="World_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            trajectory_video_batch.append(next_obs_batch_np[:, :, :, ::-1, :])

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :][::-1, :, :]
                        valid_video[task_file].append(img)
            
            for r in task_records:
                if r['active']:
                    r['finish_step'] = step
                    if r['finish_step'] >= max_steps:
                        r['active'] = False
                        r['complete'] = True 
                # r['step_images'].extend()
                                
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
        video_chunks = [initial_frame_expanded[:, :, :, ::-1, :] ] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
            
        # breakpoint()
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
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
                    
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['action_tensor'].device)
        full_trajectory_video = np.flip(full_trajectory_video, axis=3).copy()
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['action_tensor'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['action_tensor'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_vla_adapter_wm_bridge(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        # breakpoint()
        # current_obs_batch_np = np.stack(current_obs_batch_np_list)
        current_obs_batch_np = init_state.cpu().numpy()
        self.world_model.reset(current_obs_batch_np)

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                # "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
            })
            
        valid_video = defaultdict(list)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        if is_valid:
            # current_obs_batch_np 的形状是 [B, H, W, 3]
            # 我们需要把它拆开，存到每个任务对应的视频列表中
            for idx in range(batch_size):
                # task_file = task_records[idx]['task_file_name']
                # 从批次中取出第 idx 帧图像，并进行上下翻转（如果需要的话，这模仿了原函数的行为）
                img = current_obs_batch_np[idx][::-1, :, :] 
                # valid_video[task_file].append(img)
        
        step = 0
        vla_history = []
        trajectory_video_batch = [current_obs_batch_np]
        vla_timings = []
        wm_timings = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            if not active_indices:
                break
            # breakpoint()
            inputs = [self._obs_to_input(obs, flip=True) for obs in current_obs_batch_np]
            
            vla_input = self.process_input_vla_adapter(inputs, task_descriptions)
            vla_input.update(meta_info)
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train, a_shape=(5,7))
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions_batch
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)
            if self.world_model is None:
                raise ValueError("World Model Worker Group has not been set!")
            
            with Timer(name="World_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            # trajectory_video_batch.append(next_obs_batch_np[:, :, :, ::-1, :])
            trajectory_video_batch.append(next_obs_batch_np)

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    # task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :]  # [::-1, :, :]
                        # valid_video[task_file].append(img)
            
            for r in task_records:
                if r['active']:
                    r['finish_step'] = step
                    if r['finish_step'] >= max_steps:
                        r['active'] = False
                        r['complete'] = True 
                # r['step_images'].extend()
                                
        torch.cuda.empty_cache()
        # if is_valid:
        #     for task_file, images in valid_video.items():
        #         complete_flags = [r['complete'] for r in task_records if r['task_file_name'] == task_file]
        #         complete = complete_flags[0] if complete_flags else False
                
        #         save_rollout_video(
        #             images,
        #             self.config.experiment_name,
        #             task_file,
        #             global_steps,
        #             complete
        #         )
        initial_frame_expanded = np.expand_dims(trajectory_video_batch[0], axis=1) # -> (B, 1, H, W, C)
        video_chunks = [initial_frame_expanded] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
        print(task_lang)
        # breakpoint()
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
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
                    
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['action_tensor'].device)
        # full_trajectory_video = np.flip(full_trajectory_video, axis=3).copy()
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['action_tensor'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['action_tensor'].device)
        # breakpoint()
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    def _generate_minibatch_openvla_oft_flow_wm_bridge(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        # breakpoint()
        # current_obs_batch_np = np.stack(current_obs_batch_np_list)
        current_obs_batch_np = init_state.cpu().numpy()
        self.world_model.reset(current_obs_batch_np)

        task_records = []
        for idx in range(batch_size):
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                # "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
            })
            
        valid_video = defaultdict(list)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        if is_valid:
            # current_obs_batch_np 的形状是 [B, H, W, 3]
            # 我们需要把它拆开，存到每个任务对应的视频列表中
            for idx in range(batch_size):
                # task_file = task_records[idx]['task_file_name']
                # 从批次中取出第 idx 帧图像，并进行上下翻转（如果需要的话，这模仿了原函数的行为）
                img = current_obs_batch_np[idx][::-1, :, :] 
                # valid_video[task_file].append(img)
        
        step = 0
        vla_history = []
        trajectory_video_batch = [current_obs_batch_np]
        vla_timings = []
        wm_timings = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            if not active_indices:
                break
            # breakpoint()
            inputs = [self._obs_to_input(obs, flip=True) for obs in current_obs_batch_np]
            
            vla_input = self.process_input_openvla_oft_flow(inputs, task_descriptions)
            vla_input.update(meta_info)
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_openvla_oft_flow(vla_input, use_sde=is_train, a_shape=(5,7))
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions_batch
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["mean"] = torch.stack(vla_output["return_dict"]["mean"], dim=1)
            # step_data["v_t"] = torch.stack(vla_output["return_dict"]["v_t_all"], dim=1)
            # step_data["hs"] = torch.stack(vla_output["return_dict"]["hs_all"], dim=1)
            # step_data['task_latent_states'] = vla_output["return_dict"]["task_latent_states"]
            # step_data['actions_hidden_states'] = vla_output["return_dict"]["actions_hidden_states"]
            # step_data['last_hidden_states'] = vla_output["return_dict"]["last_hidden_states"]
            
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)
            if self.world_model is None:
                raise ValueError("World Model Worker Group has not been set!")
            
            with Timer(name="World_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            # trajectory_video_batch.append(next_obs_batch_np[:, :, :, ::-1, :])
            trajectory_video_batch.append(next_obs_batch_np)

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    # task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :]  # [::-1, :, :]
                        # valid_video[task_file].append(img)
            
            for r in task_records:
                if r['active']:
                    r['finish_step'] = step
                    if r['finish_step'] >= max_steps:
                        r['active'] = False
                        r['complete'] = True 
                # r['step_images'].extend()
                                
        torch.cuda.empty_cache()
        # if is_valid:
        #     for task_file, images in valid_video.items():
        #         complete_flags = [r['complete'] for r in task_records if r['task_file_name'] == task_file]
        #         complete = complete_flags[0] if complete_flags else False
                
        #         save_rollout_video(
        #             images,
        #             self.config.experiment_name,
        #             task_file,
        #             global_steps,
        #             complete
        #         )
        initial_frame_expanded = np.expand_dims(trajectory_video_batch[0], axis=1) # -> (B, 1, H, W, C)
        video_chunks = [initial_frame_expanded] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
        print(task_lang)
        # breakpoint()
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
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                # 'task_latent_states': [],
                # 'actions_hidden_states': [],
                # 'last_hidden_states': [],

                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                # "mean": [],
                # "v_t": [],
                # "hs": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
                    
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                #   'task_latent_states', 'actions_hidden_states', 'last_hidden_states', "mean", "v_t", "hs",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['action_tensor'].device)
        # full_trajectory_video = np.flip(full_trajectory_video, axis=3).copy()
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['action_tensor'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['action_tensor'].device)
        # breakpoint()
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    
    def _generate_minibatch_vla_adapter_wm_env_rollout(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        task_lang = np.repeat(prompts.non_tensor_batch['task_lang'], n_samples)
        task_descriptions = [name for name in task_lang]
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_train = meta_info.get('is_train', False)
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy(),
        )
        current_obs_batch_np_list = []
        for idx in range(len(init_data_list)):
            current_obs_batch_np_list.append(init_data_list[idx]['obs']['agentview_image'][::-1])
        current_obs_batch_np = np.stack(current_obs_batch_np_list)
        self.world_model.reset(current_obs_batch_np)

        task_records = []
        for idx in range(batch_size):
            init_data = init_data_list[idx]
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                # "step_images": [current_obs_batch_np],
                "step_images_env": [init_data['obs']['agentview_image'][::-1, ::-1]]
            })
        # breakpoint()
        valid_video = defaultdict(list)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        if is_valid:
            # current_obs_batch_np 的形状是 [B, H, W, 3]
            for idx in range(batch_size):
                task_file = task_records[idx]['task_file_name']
                img = current_obs_batch_np[idx][::-1, :, :] 
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
            
            vla_input = self.process_input_vla_adapter(inputs, task_descriptions)
            vla_input.update(meta_info)
            
            with Timer(name="VLA_Inference", text="{name} mean: {:.4f}s") as timer:
                vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train)
            vla_timings.append(timer.last)
            
            actions_batch = vla_output["action"]
            
            step_data = vla_input.copy()
            step_data["action"] = actions_batch
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)

            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions_batch,
                },
                use_vlm_rm=True,
            )
            
            if self.world_model is None:
                raise ValueError("World Model Worker Group has not been set!")
            
            with Timer(name="World_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            # breakpoint()
            # current_obs_batch_np = next_obs_batch_np
            current_obs_batch_np = next_obs_batch_np[:, -1, :, :, :]

            step += self.config.action_chunks_len
            trajectory_video_batch.append(next_obs_batch_np[:, :, :, ::-1, :])

            if is_valid:
                num_frames_in_chunk = next_obs_batch_np.shape[1]
                
                for idx in range(batch_size):
                    task_file = task_records[idx]['task_file_name']
                    for f_idx in range(num_frames_in_chunk):
                        img = next_obs_batch_np[idx, f_idx, :, :, :][::-1, :, :]
                        valid_video[task_file].append(img)
            
            # for r in task_records:
            #     if r['active']:
            #         r['finish_step'] = step
            #         if r['finish_step'] >= max_steps:
            #             r['active'] = False
            #             r['complete'] = True 
            for idx in active_indices:
                # result = output_queues[idx].get(timeout=30)
                result = step_results_list[idx]
                task_records[idx]['active_raw'] = result['active']
                task_records[idx]['complete_raw'] = result['complete']
                task_records[idx]['finish_step_raw'] = result['finish_step']
                task_records[idx]['step_images_env'].extend(result['valid_images']) 
                if task_records[idx]['active']:
                    task_records[idx]['finish_step'] = step
                    if task_records[idx]['finish_step'] >= max_steps:
                        task_records[idx]['active'] = False
                        task_records[idx]['complete'] = True 
                # r['step_images'].extend()
                                
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
        video_chunks = [initial_frame_expanded[:, :, :, ::-1, :] ] + trajectory_video_batch[1:]
        full_trajectory_video = np.concatenate(video_chunks, axis=1)
        # breakpoint()
        # self.world_model.save_trajectory_grid_image(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
            
        # breakpoint()
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
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
                    
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        batch["complete_raw"] = []
        batch["finish_step_raw"] = []
        # batch["step_images_env"] = []
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            batch["complete_raw"].append(k["complete_raw"])
            batch["finish_step_raw"].append(k["finish_step_raw"])
            # batch["step_images_env"].append(np.stack(k["step_images_env"]))
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['action_tensor'].device)
        full_trajectory_video = np.flip(full_trajectory_video, axis=3).copy()
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['action_tensor'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['action_tensor'].device)
        
        # batch["complete_raw"] = torch.tensor(batch["complete_raw"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step_raw"] = torch.tensor(batch["finish_step_raw"], dtype=torch.int64, device=batch['action_tensor'].device)
        batch["complete_raw"] = batch["finish_step_raw"] < max_steps
        # padded_step_images_env, padded_step_images_mask_env = pad_dataprotos_step_images(batch["step_images_env"])
        # padded_step_images_env = padded_step_images_env.flip(dims=[3])
        # batch["step_images_env"] = padded_step_images_env.to(device=batch['action_tensor'].device)
        # batch["step_images_mask_env"] = padded_step_images_mask_env.to(device=batch['action_tensor'].device)
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    
    def _generate_minibatch_smolvla_wm_vlac(self, prompts):
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
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_suite_name[idx]}_task_{task_id[idx][0].item()}_trial_{trial_id[idx][0].item()}",
                "step_images": [current_obs_batch_np]
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
        rm_timings = []
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
                next_obs_batch_np = self.world_model.step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            wm_timings.append(timer.last)
            breakpoint()
            with Timer(name="Reward_Model_Step", text="{name} mean: {:.4f}s") as timer:
                next_obs_batch_np = self.world_model.reward_step(current_obs_batch_np, actions_batch, step)  # B, chunk_size, H, W, C
            rm_timings.append(timer.last)
            
            
            
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
                # r['step_images'].extend()
                                
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
        #     f"work_dirs/{self.config.experiment_name}/trajectory_grid_{global_steps}.png"
        # )
        # ran_id = random.randint(1, 10000)
        # self.world_model.save_video_grid(
        #     full_trajectory_video, 
        #     f"work_dirs/{self.config.experiment_name}_train_rollouts/trajectory_grid_{global_steps}_rand{ran_id}.mp4"
        # )
            
        # breakpoint()
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
        
        # breakpoint()
        batch["complete"] = []
        batch["finish_step"] = []
        # breakpoint()
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
            
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['observation.images.image'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['observation.images.image'].device)
        batch["step_images"] = torch.from_numpy(full_trajectory_video[:, :max_steps]).to(dtype=torch.uint8, device=batch['observation.images.image'].device)
        batch["step_images_mask"] = torch.ones([batch_size, max_steps], dtype=torch.int64, device=batch['observation.images.image'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    
    def _generate_minibatch_vla_adapter(self, prompts):
        # print('Waiting for debugger 5678'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        init_state = prompts.batch['init_state'].repeat_interleave(n_samples, dim=0)
        init_state_len = prompts.batch['init_state_len'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_train = meta_info.get('is_train', False)
        # breakpoint()

        # This is a blocking call
        # self.adapter.env = None
        init_data_list = self.adapter._blocking_reset(
            task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
            trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            init_state=init_state.cpu().numpy(),
            init_state_len=init_state_len.cpu().numpy()
        )
        
        
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            # init_data = output_queues[idx].get(timeout=120)
            init_data = init_data_list[idx]
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], is_train))
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
           
            # breakpoint()
            vla_input = self.process_input_vla_adapter(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step_vla_adapter(vla_input, use_sde=is_train)
            actions = vla_output["action"]
            # breakpoint()
            
            step_data = vla_input.copy()
            step_data["action"] = actions
            step_data["action_tensor"] = vla_output["action_tensor"]
            step_data["step"] = step
            step_data["x_t"] = torch.stack(vla_output["return_dict"]["x_t"], dim=1)
            step_data["t"] = torch.stack(vla_output["return_dict"]["t"], dim=1)
            step_data["x_next"] = torch.stack(vla_output["return_dict"]["x_next"], dim=1)
            # step_data["lang_tokens"] = vla_output["lang_tokens"]
            # step_data["lang_masks"] = vla_output["lang_masks"]
            step_data["full_image"] = torch.from_numpy(np.stack([c['full_image'] for c in current_inputs])).to(vla_output["action_tensor"].device)

            vla_history.append(step_data)
            
            # for idx in active_indices:
            #     input_queues[idx].put(actions[idx])
            step_results_list = self.adapter._blocking_step({
                "indices": active_indices,
                "actions": actions,
            })
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = step_results_list[idx]
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], is_train)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
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
        # self.adapter.close()
        
        self.module.train()
        # breakpoint()
        batch = {"input_ids":[], 
                "attention_mask": [],
                # "pixel_values": [], 
                "full_image": [],
                "x_t": [],
                "t": [],
                "x_next": [],
                "action_tensor": [],
                # "lang_tokens": [],
                # "lang_masks": []
                }  
                    
        for k in ["input_ids", "attention_mask",
                #   "pixel_values", 
                  "full_image",
                  "action_tensor", "x_t", "t", "x_next"]:
            for h in vla_history:
                batch[k].append(h[k])
                
        for k,v in batch.items():
            batch[k] = torch.stack(v, dim=1) 
        
        batch["complete"] = []
        batch["finish_step"] = []
        
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['action_tensor'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['action_tensor'].device)
        # print(batch)
        # breakpoint()
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
    
    @torch.no_grad()
    def _generate_one_step_pi(self, prompts: dict, use_sde: bool = False):
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # actions = self.module.select_action(prompts)
                actions, lang_tokens, lang_masks, return_dict = self.module.predict_action_chunk(prompts, use_sde=use_sde)
        
        batch = prompts.copy()
        batch["action_tensor"] = torch.from_numpy(actions)
        batch["action"] = actions  # .to(torch.float32).cpu().numpy()
        batch["return_dict"] = return_dict
        batch["lang_tokens"] = lang_tokens
        batch["lang_masks"] = lang_masks
        
        return batch
    
    @torch.no_grad()
    def _generate_one_step_vla_adapter(self, prompts: dict, use_sde: bool = False, a_shape=(20,7)):
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # actions = self.module.select_action(prompts)
                # actions, lang_tokens, lang_masks, return_dict = self.module.predict_action(prompts, use_sde=use_sde)
                # breakpoint()
                idx = prompts['input_ids']  # (bs, prompt_length)
                attention_mask = prompts['attention_mask']  # left-padded attention_mask
                pixel_values = prompts["pixel_values"]
                # breakpoint()
                action, return_dict = self.module.predict_action(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    unnorm_key=self.config.unnorm_key,
                    do_sample=False,
                    proprio=None,
                    proprio_projector=None,
                    noisy_action_projector=self.noisy_action_projector,
                    action_head=self.action_head,
                    use_film=False,
                    use_sde=use_sde,
                    a_shape=a_shape,
                    # use_sde=True,
                )
                # breakpoint()
                # action_len = 8
                # action = action[:, :action_len]

        
        batch = prompts.copy()
        batch["action_tensor"] = torch.from_numpy(action).to(pixel_values.device)
        batch["action"] = action  #.to(torch.float32).cpu().numpy()
        batch["return_dict"] = return_dict
        # batch["lang_tokens"] = lang_tokens
        # batch["lang_masks"] = lang_masks
        
        return batch
    
    @torch.no_grad()
    def _generate_one_step_openvla_oft_flow(self, prompts: dict, use_sde: bool = False, a_shape=(20,7)):
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # actions = self.module.select_action(prompts)
                # actions, lang_tokens, lang_masks, return_dict = self.module.predict_action(prompts, use_sde=use_sde)
                # breakpoint()
                idx = prompts['input_ids']  # (bs, prompt_length)
                attention_mask = prompts['attention_mask']  # left-padded attention_mask
                pixel_values = prompts["pixel_values"]
                # breakpoint()
                action, return_dict = self.module.predict_action(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    unnorm_key=self.config.unnorm_key,
                    do_sample=False,
                    proprio=None,
                    proprio_projector=None,
                    noisy_action_projector=self.noisy_action_projector,
                    action_head=self.action_head,
                    use_film=False,
                    use_sde=use_sde,
                    a_shape=a_shape,
                    # use_sde=True,
                )
                action = action[0]
                # breakpoint()
                # action_len = 8
                # action = action[:, :action_len]

        
        batch = prompts.copy()
        batch["action_tensor"] = torch.from_numpy(action).to(pixel_values.device)
        batch["action"] = action  #.to(torch.float32).cpu().numpy()
        batch["return_dict"] = return_dict
        # batch["lang_tokens"] = lang_tokens
        # batch["lang_masks"] = lang_masks
        
        return batch

    def _obs_to_input(self, obs, is_train=True, flip=False):
        if self.use_world_model and is_train:
            return {
                "full_image": obs[:, ::-1] if not flip else obs,
            }
        else:
            if self.config.num_images_in_input > 1:
                return {
                    "full_image": get_libero_image(obs, 224, flip=flip),
                    "wrist_image": get_libero_wrist_image(obs, 224),
                    "state": np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]
                    ])
                }
            else:
                return {
                    "full_image": get_libero_image(obs, 224, flip=flip),
                    "state": np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]
                    ])
                }