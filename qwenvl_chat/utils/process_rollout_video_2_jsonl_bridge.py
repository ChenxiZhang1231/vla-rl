import argparse
from pathlib import Path
import os
import re
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm 
import cv2 
import numpy as np
import json

import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/LIBERO")
from libero.libero import benchmark


def parse_task_id_from_name(name: str) -> Optional[int]:
    """
    支持多种命名：...task_12... 或者 ..._12_...（你原先 split('_')[3] 的做法容易出错）
    """
    m = re.search(r"task[_\-]?(\d+)", name)
    if m: 
        return int(m.group(1))
    # 退路：尝试抓最后一个纯数字片段
    parts = re.split(r"[^\d]+", name)
    nums = [p for p in parts if p.isdigit()]
    return int(nums[-1]) if nums else None

def parse_success_from_name(name: str) -> int:
    """
    文件名中包含 success=True / success=False
    """
    lower = name.lower()
    if "success=true" in lower:
        return 1
    if "success=false" in lower:
        return 0
    raise ValueError(f"文件名未包含 success=True/False: {name}")

def parse_finish_step_from_name(name: str) -> int:
    parts = name.split('--')
    
    for part in parts:
        if part.startswith("finish_step="):
            try:
                value_str = part.split('=')[1]
                return int(value_str)
            except (IndexError, ValueError):
                return None
                
    return None
    
def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def is_video_file(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS
 
 

def process_video_to_frames(video_path: Path, num_frames: int = 10) -> list[str]:
    """
    """
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return []

    if num_frames > total_frames:
        num_frames = total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (128, 128))
            frames.append(resized_frame)

    cap.release()
    return frames, frame_indices, num_frames


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

def map_finish_step_to_sampled_idx(finish_step_raw: int, sampled_indices: list[int]) -> int:
    """
    """
    if finish_step_raw is None or finish_step_raw < 0 or len(sampled_indices) == 0:
        return -1
    for v in sampled_indices:
        if v >= finish_step_raw:
            return int(v)
    return int(sampled_indices[-1])

def build_question(task_lang: str, step_ids: list[int]) -> str:
    frame_str = ""
    for step_id in step_ids:
        frame_str += f"frame_step{step_id}-<image>\n"
    return (
        f"{PROMPT}\n"
        f"Task: {task_lang}\n"
        f"{frame_str}"
    )

def build_answer(complete: int, finish_step_raw: int, sampled_indices: list[int]) -> str:
    if int(complete) == 1:
        mapped = map_finish_step_to_sampled_idx(finish_step_raw, sampled_indices)
        payload = {"success": 1, "finish_step": int(mapped if mapped is not None else -1)}
    else:
        payload = {"success": 0, "finish_step": -1}
    import json
    return json.dumps(payload, ensure_ascii=False)

def normalize_basename(name: str) -> str:
    """
    统一成用于匹配的基名：
    - 去目录
    - 去扩展名
    - 去掉结尾的 `_biased` 或 `_origin`（若存在）
    """
    base = os.path.basename(name)
    base = re.sub(r'\.(mp4|avi|mov|mkv)$', '', base, flags=re.IGNORECASE)
    base = re.sub(r'_(biased|origin)$', '', base, flags=re.IGNORECASE)
    return base

def build_caption_index(dataset_json_path: str) -> dict:
    """
    读取 dataset.json，返回 {规范化基名: caption}
    其中规范化基名来自 media_path 的文件名处理后。
    """
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    idx = {}
    for item in data:
        media_path = item.get("media_path", "")
        caption = item.get("caption", "")
        base = normalize_basename(media_path)
        # 如果后面出现重复 key，保留第一个或做去重策略均可；这里采用后者覆盖前者
        idx[base] = caption
    return idx


def map_videos_to_captions(video_list: list, dataset_json_path: str) -> dict:
    caption_idx = build_caption_index(dataset_json_path)
    result = {}
    missing = []

    for v in video_list:
        key = normalize_basename(v)
        cap = caption_idx.get(key)
        if cap is None:
            missing.append(v)
        else:
            result[v] = cap

    # 打印未匹配的，方便排查
    if missing:
        print("未在 dataset.json 中找到 caption 的视频（规范化后未匹配到）:")
        for m in missing:
            print("  -", m)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on rollout videos.")
    parser.add_argument("--video_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/data/rm_bridge/generation/predict2_video2world_2b_action_conditioned_finetuning_generation_bridge_2025-10-20_17-11-12/videos/00000",
                        help="")
    parser.add_argument("--output_dir", type=str, default="/inspire/ssd/project/robotsimulation/public/data/rm_for_wm_train_jsonl_bridge", help="")
    parser.add_argument("--dataset_json", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/bridge_orig/dataset.json", help="")
    parser.add_argument("--num_frames", type=int, default=50, help="")
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    out_dir = Path(args.output_dir)
    frames_save_path = out_dir / "images"
    safe_makedirs(out_dir)
    safe_makedirs(frames_save_path)



    videos = [f for f in os.listdir(video_folder) if is_video_file(f)]
    
    mapping = map_videos_to_captions(videos, args.dataset_json)
    
    if not videos:
        raise FileNotFoundError(f"目录中未找到视频: {video_folder}")

    jsonl_list = []
    for vid, (name, cap) in tqdm(enumerate(mapping.items())):
        video_path = video_folder / name
        complete = False if "biased" in name else True
        lang = cap
        task_lang = str(lang)
            
        frames, frame_indices, num_frames_real = process_video_to_frames(video_path, num_frames=args.num_frames)
        
        for frame, frame_id in zip(frames, frame_indices):
            frame_save_path = frames_save_path / f"video{vid}" / f"{frame_id}.png"
            safe_makedirs(frame_save_path.parent)
            cv2.imwrite(str(frame_save_path), frame)
        
        entry = {
            "id": f"video{vid}",
            "image": [],
            "width_list": [],
            "height_list": [],
            "conversations": [],
        }

        for i in range(num_frames_real):
            entry['image'].append(f"images/video{vid}/{frame_indices[i]}.png")
            entry['width_list'].append(128)
            entry['height_list'].append(128)

        question = build_question(task_lang, step_ids=frame_indices)
        finish_step = num_frames_real-1
        answer = build_answer(complete=complete, finish_step_raw=finish_step, sampled_indices=frame_indices)

        q = {"from": "human", "value": question}
        a = {"from": "gpt", "value": answer}
        entry["conversations"].append(q)
        entry["conversations"].append(a)
        
        jsonl_list.append(entry)
                
    jsonl_file_name = f'train_rm_{len(jsonl_list)}.jsonl'
    jsonl_file_path = os.path.join(str(out_dir), jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in jsonl_list:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 

if __name__ == "__main__":
    main()