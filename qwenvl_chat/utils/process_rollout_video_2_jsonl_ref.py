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

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (128, 128))
            frames.append(resized_frame)

    cap.release()
    return frames, frame_indices


PROMPT = """You are a task-conditioned video rollout success judge.

You will be given two sets of ordered frames:
1.  **Reference Video:** This video demonstrates a successful completion of the task. Use it to understand what success looks like.
2.  **Rollout Video:** This is the video you must evaluate.

Your job is to judge the **Rollout Video** and decide:
(1) whether the task is successfully completed,
and (2) at which step index (from the provided Rollout Video step_id list) the success is FIRST
visibly satisfied.

Principles
- Base your judgment of the Rollout Video on the example of success shown in the Reference Video.
- Use only the provided frames. Do not assume off-camera facts.
- Success requires visible, decisive evidence in the Rollout Video frames.
- Do NOT infer “about to succeed” (e.g., hovering ≠ ON/IN).
- If a required condition seen in the Reference Video cannot be verified in the Rollout Video, choose Failure.
- The reported finish_step must be one of the provided Rollout Video step_ids; if Failure, use -1.

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

def build_question_ref(task_lang: str, step_ids: list[int], ref_step_ids: list[int]) -> str:
    """
    构建包含参考视频和执行视频的 prompt 字符串。
    """
    # 1. 构建参考视频的帧字符串
    ref_frame_str = "Reference Video Frames:\n"
    for step_id in ref_step_ids:
        ref_frame_str += f"ref_frame_step{step_id}-<image>\n"

    # 2. 构建待评判视频的帧字符串
    rollout_frame_str = "Rollout Video Frames (to be judged):\n"
    for step_id in step_ids:
        rollout_frame_str += f"frame_step{step_id}-<image>\n"
    
    # 3. 组合最终的 prompt
    separator = "\n---\n\n"
    
    return (
        f"{PROMPT}\n"
        f"Task: {task_lang}\n\n"
        f"{ref_frame_str}"
        f"{separator}"
        f"{rollout_frame_str}"
    )

def build_answer(complete: int, finish_step_raw: int, sampled_indices: list[int]) -> str:
    if int(complete) == 1:
        mapped = map_finish_step_to_sampled_idx(finish_step_raw, sampled_indices)
        payload = {"success": 1, "finish_step": int(mapped if mapped is not None else -1)}
    else:
        payload = {"success": 0, "finish_step": -1}
    import json
    return json.dumps(payload, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on rollout videos.")
    parser.add_argument("--video_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train",
                        help="")
    parser.add_argument("--output_dir", type=str, default="/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_ref/rm_train", help="")
    parser.add_argument("--num_frames", type=int, default=50, help="")
    parser.add_argument("--num_ref_frames", type=int, default=20, help="")
    parser.add_argument("--task_name", type=str, default="libero_spatial", help="benchmark 名")
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    out_dir = Path(args.output_dir)
    frames_save_path = out_dir / "images"
    safe_makedirs(out_dir)
    safe_makedirs(frames_save_path)


    task_lang_lookup = {}
    if benchmark is not None:
        bdict = benchmark.get_benchmark_dict()
        if args.task_name in bdict:
            suite = bdict[args.task_name]()
            # suite.get_task(task_id) -> (something, language)
            task_lang_lookup["__suite__"] = suite


    videos = [f for f in os.listdir(video_folder) if is_video_file(f)]
    if not videos:
        raise FileNotFoundError(f"目录中未找到视频: {video_folder}")

    jsonl_list = []
    for vid, name in tqdm(enumerate(sorted(videos))):
        video_path = video_folder / name
        complete = parse_success_from_name(name)
        finish_step = parse_finish_step_from_name(name)
        task_id = parse_task_id_from_name(name)
        lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
        task_lang = str(lang)
            
        frames, frame_indices = process_video_to_frames(video_path, num_frames=args.num_frames)
        ref_video_path = Path(REF_DICT[args.task_name][task_id])
        ref_frames, ref_frame_indices = process_video_to_frames(ref_video_path, num_frames=args.num_ref_frames)
        
        for frame, frame_id in zip(frames, frame_indices):
            frame_save_path = frames_save_path / f"video{vid}" / f"{frame_id}.png"
            safe_makedirs(frame_save_path.parent)
            cv2.imwrite(str(frame_save_path), frame)
        
        for ref_frame, ref_frame_id in zip(ref_frames, ref_frame_indices):
            frame_save_path = frames_save_path / f"ref_video_task_{task_id}" / f"{ref_frame_id}.png"
            safe_makedirs(frame_save_path.parent)
            if not frame_save_path.exists():
                cv2.imwrite(str(frame_save_path), ref_frame)
        
        entry = {
            "id": f"video{vid}",
            "image": [],
            "width_list": [],
            "height_list": [],
            "conversations": [],
        }

        for i in range(args.num_ref_frames):
            entry['image'].append(f"images/ref_video_task_{task_id}/{ref_frame_indices[i]}.png")
            entry['width_list'].append(128)
            entry['height_list'].append(128)
            
        for i in range(args.num_frames):
            entry['image'].append(f"images/video{vid}/{frame_indices[i]}.png")
            entry['width_list'].append(128)
            entry['height_list'].append(128)

        question = build_question_ref(task_lang, step_ids=frame_indices, ref_step_ids=ref_frame_indices)
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