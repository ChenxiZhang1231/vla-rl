#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
from collections import Counter

import base64
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from functools import partial
from PIL import Image

from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os

import openai
import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/LIBERO")
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev")
from libero.libero import benchmark
from verl_vla.utils.prompt_utils.prompt import build_system_prompt

@dataclass
class RewardTask:
    frames: List[str]
    frame_indices: List[int]
    ref_frames: List[str]
    ref_frame_indices: List[int]
    description: str
    score_gt: int            # 0/1
    video_name: str
    finish_step_gt_raw: Optional[int] = None
    finish_step_gt_mapped: Optional[int] = None
    pred_success: Optional[int] = None
    pred_finish_step: Optional[int] = None
    score_text: str = ""

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
# =========================
# Utilities
# =========================
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def is_video_file(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS

def parse_task_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"task[_\-]?(\d+)", name, flags=re.I)
    if m:
        return int(m.group(1))
    parts = re.split(r"[^\d]+", name)
    nums = [p for p in parts if p.isdigit()]
    return int(nums[-1]) if nums else None

def parse_success_from_name(name: str) -> int:
    lower = name.lower()
    if "success=true" in lower:
        return 1
    if "success=false" in lower:
        return 0
    raise ValueError(f"文件名未包含 success=True/False: {name}")

def parse_finish_step_from_name(name: str) -> Optional[int]:
    """
    尝试从文件名中解析完成步。如: ...finish_step=123..., ...finish_step-123..., ...finish123...
    若未找到则返回 None（表示没有‘完成步’标签）。
    """
    lower = name.lower()
    
    # 修改后的模式，增加了对等号 '=' 的支持
    # [_\-=]? 匹配一个可选的下划线、连字符 或 等号
    for pat in [r"finish[_\-]?step[_\-=]?(\d+)",  # 匹配 finish_step=256, finish-step-256, finish_step_256
                r"finished[_\-=]?(\d+)",         # 匹配 finished=123
                r"finish[_\-=]?(\d+)"]:           # 匹配 finish=123
        
        m = re.search(pat, lower)
        if m:
            # group(1) 捕获的是括号内的数字部分
            return int(m.group(1))
            
    return None

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_binary_like(scores: List[int | float]) -> bool:
    uniq = set(float(x) for x in scores)
    return uniq.issubset({0.0, 1.0})

def to_pred_labels(y_score: List[float], threshold: float) -> List[int]:
    return [1 if s >= threshold else 0 for s in y_score]

def compute_confusion(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def compute_basic_metrics(cm: Dict[str, int]) -> Dict[str, float]:
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

def auc_mann_whitney(y_true: List[int], y_score: List[float]) -> Optional[float]:
    pos_idx = [i for i, y in enumerate(y_true) if y == 1]
    neg_idx = [i for i, y in enumerate(y_true) if y == 0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted([(s, i) for i, s in enumerate(y_score)], key=lambda x: x[0])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            _, idx = pairs[k]
            ranks[idx] = avg_rank
        i = j
    sum_ranks_pos = sum(ranks[i] for i in pos_idx)
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)

def compute_all_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
    cm = compute_confusion(y_true, y_pred)
    basic = compute_basic_metrics(cm)
    return {"confusion": cm, "metrics": basic}

def process_video_to_base64_frames_with_indices(video_path: Path, num_frames: int = 10) -> Tuple[List[str], List[int]]:
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
    base64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 不缩放，保持原图；如需加速可打开 128x128
            resized_frame = cv2.resize(frame, (128, 128))
            _, buffer = cv2.imencode('.jpg', resized_frame)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")
    cap.release()
    # 若读帧失败导致数量不一致，截齐
    L = min(len(base64_frames), len(frame_indices))
    return base64_frames[:L], frame_indices[:L]

def process_video_to_PIL_frames_with_indices(video_path: Path, num_frames: int = 10) -> Tuple[List[str], List[int]]:
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

def map_finish_step_to_sampled_idx(finish_step_raw: int, sampled_indices: List[int]) -> int:
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

def build_question(task_lang: str, step_ids: list[int]) -> str:
    """
    生成与你 SFT 一致的 user 文本：先 PROMPT，然后 Task，再逐行
    'frame_step{step_id}-<image>'，不使用 system prompt。
    注意：这里的 step_ids 必须与后续附加的图片顺序一一对应。
    """
    # 如果视频解码有丢帧，务必用 len(frames) 截齐 step_ids，以保证一一对应
    frame_str = "".join([f"frame_step{sid}-<image>\n" for sid in step_ids])
    return f"{PROMPT}\nTask: {task_lang}\n{frame_str}"


def try_parse_json_response(text: str) -> Optional[Dict]:
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


# =========================
# VLM 调用（SFT 版）
# =========================
def fetch_one_reward_sync(client, task: RewardTask, task_index: int, mode: str, temperature, top_p, seed) -> Tuple[int, int, int, str]:
    """
    对单个样本调用 judge。
    返回: (task_index, pred_success (0/1), pred_finish_step (int or -1), raw_text)
    """
    if not task.frames:
        return task_index, 0, -1, "Empty"

    step_ids = task.frame_indices[:len(task.frames)]
    question = build_question(task.description, step_ids=step_ids)
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

    data = try_parse_json_response(resp)
    if data is None:
        return task_index, 0, -1, resp

    succ = 1 if int(data.get("success", 0)) == 1 else 0
    fs_raw = int(data.get("finish_step", -1))
    fs_mapped = map_finish_step_to_sampled_idx(fs_raw, step_ids) if succ == 1 else -1
    return task_index, succ, fs_mapped, resp


def get_rewards_from_judge_batch_sync(
    client,
    tasks: List[RewardTask],
    max_workers: int = 10,
    mode: str = "v2",
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
                fetch_one_reward_sync, client, task, i, mode,
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


# =========================
# finish_step 评测
# =========================
def compute_finish_step_metrics(tasks: List[RewardTask]) -> Dict[str, float]:
    """
    仅对 (GT=1 且 预测=1) 的样本统计完成步误差。
    """
    errs: List[int] = []
    n_gt_pos = 0
    n_pred_pos = 0
    for t in tasks:
        if t.score_gt == 1:
            n_gt_pos += 1
        if t.pred_success == 1:
            n_pred_pos += 1
        if t.score_gt == 1 and t.pred_success == 1 and t.finish_step_gt_mapped is not None and t.finish_step_gt_mapped >= 0 and t.pred_finish_step is not None and t.pred_finish_step >= 0:
            errs.append(abs(int(t.pred_finish_step) - int(t.finish_step_gt_mapped)))
        
    if not errs:
        return {
            "count": 0,
            "mae": None,
            "med_ae": None,
            "rmse": None,
            "within_0": None,
            "within_1": None,
            "within_2": None,
            "gt_success_count": n_gt_pos,
            "pred_success_count": n_pred_pos,
            "coverage_pred_success_over_gt_success": 0.0 if n_gt_pos > 0 else None,
        }

    arr = np.asarray(errs, dtype=float)
    mae = float(np.mean(arr))
    med = float(np.median(arr))
    rmse = float(np.sqrt(np.mean(arr**2)))
    within_0 = float(np.mean(arr <= 0.5))       # 精确命中
    within_1 = float(np.mean(arr <= 1.0))       # 误差≤1帧
    within_2 = float(np.mean(arr <= 2.0))       # 误差≤2帧

    # 覆盖率：在 GT=1 的样本中，被判为成功的比例
    cov = None
    if n_gt_pos > 0:
        # 注意：覆盖率不是 errs 占比，因为 errs 仅统计了 (GT=1 & pred=1)
        # 覆盖率 = (#(GT=1 & pred=1)) / (#(GT=1))
        n_gt_pred = sum(1 for t in tasks if t.score_gt == 1 and t.pred_success == 1)
        cov = n_gt_pred / n_gt_pos

    return {
        "count": int(len(errs)),
        "mae": mae,
        "med_ae": med,
        "rmse": rmse,
        "within_0": within_0,
        "within_1": within_1,
        "within_2": within_2,
        "gt_success_count": n_gt_pos,
        "pred_success_count": n_pred_pos,
        "coverage_pred_success_over_gt_success": cov,
    }

def process_single_video(name, eval_folder, task_name, num_frames, ref_num_frames):
    """
    处理单个视频文件，提取所有必要信息。
    这个函数将在一个独立的进程中运行。
    
    返回:
        一个包含 RewardTask 和 csv_row 的元组，如果处理失败则返回 None。
    """
    video_path = eval_folder / name
    try:
        gt = parse_success_from_name(name)
    except Exception:
        print(f"[SKIP] {name}（未找到 success=True/False）")
        return None

    task_lang_lookup = {}
    if benchmark is not None:
        try:
            bdict = benchmark.get_benchmark_dict()
            if task_name in bdict:
                suite = bdict[task_name]()
                task_lang_lookup["__suite__"] = suite
        except Exception as e:
            print(f"[WARN] benchmark 加载失败：{e}")
            
    task_id = parse_task_id_from_name(name)
    task_lang = ""
    if "__suite__" in task_lang_lookup and task_id is not None:
        try:
            lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
            task_lang = str(lang)
        except Exception:
            pass  # 保持 task_lang 为空字符串

    # 瓶颈所在：I/O 和 CPU 密集型操作
    frames, frame_indices = process_video_to_PIL_frames_with_indices(video_path, num_frames=num_frames)

    finish_raw = parse_finish_step_from_name(name)
    finish_mapped = map_finish_step_to_sampled_idx(finish_raw, frame_indices) if finish_raw is not None else -1

    ref_video_path = Path(REF_DICT[task_name][task_id])
    ref_frames, ref_frame_indices = process_video_to_PIL_frames_with_indices(ref_video_path, num_frames=ref_num_frames)
    
    task = RewardTask(
        frames=frames,
        frame_indices=frame_indices,
        ref_frames=ref_frames,
        ref_frame_indices=ref_frame_indices,
        description=task_lang,
        score_gt=gt,
        video_name=name,
        finish_step_gt_raw=finish_raw,
        finish_step_gt_mapped=finish_mapped if gt == 1 else -1,
    )

    row_for_csv = {
        "video_name": name,
        "task_id": task_id,
        "task_lang": task_lang,
        "gt_success": gt,
        "gt_finish_step_raw": finish_raw,
        "gt_finish_step_mapped": finish_mapped if gt == 1 else -1,
    }

    return (task, row_for_csv)

def main_processing_loop(videos, eval_folder, args):
    tasks = []
    rows_for_csv = []
    
    # 如果是 debug 模式，只处理前10个视频
    if args.debug:
        videos_to_process = sorted(videos)[:10]
    else:
        videos_to_process = sorted(videos)

    # 设置工作进程数，通常设置为 CPU 核心数，或者可以由 args 控制
    num_workers = args.max_workers if hasattr(args, 'max_workers') else os.cpu_count()
    num_workers = 1
    # num_workers = os.cpu_count() # 使用所有CPU核心
    print(f"使用 {num_workers} 个进程进行并行处理...")

    # 使用 functools.partial 来预先填充 process_single_video 函数的固定参数
    worker_func = partial(
        process_single_video,
        eval_folder=eval_folder,
        task_name=args.task_name,
        num_frames=args.num_frames,
        ref_num_frames=args.ref_num_frames,
    )

    # 创建一个进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 executor.map 来将任务分配给进程池
        # executor.map 会保持原始输入的顺序
        # 将其包裹在 tqdm 中以显示进度条
        results_iterator = executor.map(worker_func, videos_to_process)
        
        for result in tqdm(results_iterator, total=len(videos_to_process), desc="Processing videos"):
            if result is not None:
                task, row_for_csv = result
                tasks.append(task)
                rows_for_csv.append(row_for_csv)

    return tasks, rows_for_csv


def aggregate_success_and_finish(pred_success_list: List[int], pred_finish_list: List[int], m: int, strategy: str="min") -> Tuple[int,int]:
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
    
    
def get_vlac_model(args):
    Critic=GAC_model(tag='critic')
    Critic.init_model(model_path=args.rm_model_path, model_type='internvl2', device_map=f'cuda:0')
    Critic.temperature=0.5
    Critic.top_k=1
    Critic.set_config()
    Critic.set_system_prompt()
    return Critic

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-style reward model (JSON success + finish_step).")
    parser.add_argument("--eval_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_val",
                        help="存放评测视频的目录")
    parser.add_argument("--output_dir", type=str, default="work_dirs/vlac-oneshot", help="结果保存目录")
    parser.add_argument("--rm_model_path", type=str, default="/inspire/ssd/project/robotsimulation/public/huggingface_models/VLAC", help="")
    parser.add_argument("--tag", type=str, default="sft-7b-lora64-60steps-vote5-pass3", help="子目录名")
    parser.add_argument("--mode", type=str, default="", help="传给 build_system_prompt 的模式字段")
    parser.add_argument("--task_name", type=str, default="libero_spatial", help="benchmark 名")
    parser.add_argument("--num_frames", type=int, default=20, help="每段视频抽帧数")
    parser.add_argument("--ref_num_frames", type=int, default=20, help="每段视频抽帧数")
    parser.add_argument("--max_workers", type=int, default=128, help="judge 并行线程数")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--dry_run", action="store_true", help="只做解析不过 judge（用于快速检查）")
    parser.add_argument("--vote_n", type=int, default=5, help="每个视频生成多少个变体做投票 (n)")
    parser.add_argument("--vote_m", type=int, default=3, help="至少多少票成功算通过 (m)")
    parser.add_argument("--vote_finish_agg", type=str, default="min", choices=["min","median","mode"],
                        help="对成功变体的 finish_step 聚合方式")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度（>0 才有随机性）")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling 截断")
    parser.add_argument("--seed_base", type=int, default=-1, help=">=0 时每次采样用 seed_base+i；<0 则不传 seed")
    args = parser.parse_args()

    eval_folder = Path(args.eval_folder)
    out_dir = Path(os.path.join(args.output_dir, args.tag))
    safe_makedirs(out_dir)

    videos = [f for f in os.listdir(eval_folder) if is_video_file(f)]
    if not videos:
        raise FileNotFoundError(f"目录中未找到视频: {eval_folder}")

    tasks, rows_for_csv = main_processing_loop(videos, eval_folder, args)
    
    if args.dry_run:
        print(f"[DRY-RUN] 已解析 {len(tasks)} 段视频，停止于此。")
        return

    vlac_model = get_vlac_model(args)
    for i, task in enumerate(tasks):
        if i >= 10:
            break
        done_list = vlac_model.eval_trajectory(
            task,
            batch_num=10,
            ref_num=10,
            skip=5,
            frame_skip=True, #whether to skip frames(if false, each frame while be evaluated, cost more time)
            done_threshold=0.9,#done threshold
            video_output=True,
            output_path=str(out_dir).replace(args.tag, f"{args.tag}_saved_video")
        )
        print(done_list)        
    
    
    y_true = [t.score_gt for t in tasks]
    y_pred = [t.pred_success for t in tasks]
    cls_stat = compute_all_metrics(y_true, y_pred)

    step_stat = compute_finish_step_metrics(tasks)

    import csv
    csv_path = out_dir / "reward_eval_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "video_name", "task_id", "task_lang",
            "gt_success", "pred_success",
            "gt_finish_step_raw", "gt_finish_step_mapped",
            "pred_finish_step", "step_abs_err",
            "raw_text",
        ]
        writer.writerow(header)
        for t in tasks:
            step_err = (
                abs(int(t.pred_finish_step) - int(t.finish_step_gt_mapped))
                if (t.score_gt == 1 and t.pred_success == 1 and t.finish_step_gt_mapped is not None and t.finish_step_gt_mapped >= 0 and t.pred_finish_step is not None and t.pred_finish_step >= 0)
                else ""
            )
            writer.writerow([
                t.video_name,
                parse_task_id_from_name(t.video_name),
                t.description,
                t.score_gt,
                t.pred_success,
                t.finish_step_gt_raw,
                t.finish_step_gt_mapped if t.score_gt == 1 else -1,
                t.pred_finish_step if t.pred_success == 1 else -1,
                step_err,
                t.score_text,
            ])

    json_path = out_dir / "reward_eval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"classification": cls_stat, "finish_step": step_stat}, f, ensure_ascii=False, indent=2)

    txt_path = out_dir / "reward_eval_summary.txt"
    cm = cls_stat["confusion"]
    m = cls_stat["metrics"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("== Classification (Success) ==\n")
        f.write(f"TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}\n")
        f.write(f"Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  Recall/TPR={m['recall']:.4f}  F1={m['f1']:.4f}\n")
        f.write(f"FPR={m['fpr']:.4f}  TNR/Specificity={m['tnr_specificity']:.4f}  FNR={m['fnr']:.4f}  Balanced-Acc={m['balanced_accuracy']:.4f}  MCC={m['mcc']:.4f}\n\n")

        f.write("== Finish Step (only on GT=1 & Pred=1) ==\n")
        f.write(f"Count={step_stat['count']}  GT_pos={step_stat['gt_success_count']}  Pred_pos={step_stat['pred_success_count']}\n")
        f.write(f"Coverage (Pred=1 among GT=1)={step_stat['coverage_pred_success_over_gt_success']}\n")
        f.write(f"MAE={step_stat['mae']}  MedianAE={step_stat['med_ae']}  RMSE={step_stat['rmse']}\n")
        f.write(f"Within±0={step_stat['within_0']}  Within±1={step_stat['within_1']}  Within±2={step_stat['within_2']}\n")

    print(f"[OK] 评测完成，共 {len(tasks)} 段。")
    print(f"明细: {csv_path}")
    print(f"汇总: {json_path}")
    print(f"摘要: {txt_path}")


if __name__ == "__main__":
    main()
