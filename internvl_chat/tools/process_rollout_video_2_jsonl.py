import argparse
from pathlib import Path
import os
import re
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm 
import cv2 
import numpy as np
import json
import uuid

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

def process_video_to_frames_step(
    video_path: Path,
    step: int = 5,                 # 间隔帧数：每 step 取一帧
    finish_step: int = -1,
    max_frames: int | None = None, # 可选：最多取多少帧
    resize_hw: tuple[int, int] = (128, 128),
):
    """
    按固定间隔采样视频帧（默认每 5 帧取 1 帧），返回 (frames, frame_indices)

    frames: list[np.ndarray]，每帧为 BGR 128x128
    frame_indices: np.ndarray[int]，对应原视频中的帧号
    """
    if step <= 0:
        raise ValueError("step 必须为正整数")
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # if total_frames <= 0 or not cap.isOpened():
    #     cap.release()
    #     return [], np.array([], dtype=int)
    total_frames = finish_step

    # 计划采样的索引
    # indices = np.arange(0, total_frames, step, dtype=int)
    indices = np.arange(0, total_frames, step, dtype=int)
    last = total_frames - 1
    if last >= 0:
        indices = np.unique(np.append(indices, last)) 
        
    if max_frames is not None:
        indices = indices[:max_frames]

    frames: list[np.ndarray] = []
    target_ptr = 0  # 指向下一个需要的索引
    cur_idx = 0

    # 顺序读取，遇到需要的帧就保存（避免反复 seek）
    while target_ptr < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if cur_idx == indices[target_ptr]:
            frames.append(cv2.resize(frame, resize_hw))
            target_ptr += 1
        cur_idx += 1

    cap.release()
    # 实际返回的索引（防止读到中途断流）
    return frames, indices[:len(frames)]


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

def trojectory_example_prompt(images,task):
    prompt=f"<trajectory> <task> {task} </task>:"
    t_len=len(images)-1
    for i,one in enumerate(range(len(images))):
        temp_p=int((i/t_len)*100)
        prompt=prompt+f" {temp_p}% <image>\n"
    prompt=prompt+'</trajectory>'
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on rollout videos.")
    parser.add_argument("--video_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/rm_train",
                        help="")
    parser.add_argument("--output_dir", type=str, default="/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_vlac/rm_train", help="")
    parser.add_argument("--frame_step", type=int, default=5, help="")
    parser.add_argument("--num_ref_frames", type=int, default=10, help="")
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
        if complete == 0:
            continue
        finish_step = parse_finish_step_from_name(name)
        task_id = parse_task_id_from_name(name)
        lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
        task_lang = str(lang)
            
        frames, frame_indices = process_video_to_frames_step(video_path, step=args.frame_step, finish_step=finish_step)
        ref_video_path = Path(REF_DICT[args.task_name][task_id])
        ref_frames, ref_frame_indices = process_video_to_frames(ref_video_path, num_frames=args.num_ref_frames)
        
        for frame, frame_id in zip(frames, frame_indices):
            frame_save_path = frames_save_path / f"video{vid}" / f"{frame_id}.png"
            safe_makedirs(frame_save_path.parent)
            cv2.imwrite(str(frame_save_path), frame)
        
        ref_frame_path_list = []
        for ref_frame, ref_frame_id in zip(ref_frames, ref_frame_indices):
            ref_frame_save_path = frames_save_path / f"video_ref" / f"{args.task_name}-task{task_id}-frame{ref_frame_id}.png"
            ref_frame_path = Path("images") / f"video_ref" / f"{args.task_name}-task{task_id}-frame{ref_frame_id}.png"
            ref_frame_path_list.append(ref_frame_path)
            safe_makedirs(ref_frame_save_path.parent)
            if not ref_frame_save_path.exists():
                cv2.imwrite(str(ref_frame_save_path), ref_frame)

        trajectory_prompt = trojectory_example_prompt(list(range(len(ref_frame_indices))), task=task_lang)
        prompt_templete = "0% <image>\nThis image is the trajectory beginning of the following two images\nImage-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        question = trajectory_prompt + prompt_templete.format(task_lang)
        
        frame_pairs = []
        for i in range(len(frame_indices)-3):
            frame_pairs.append((i, i+1, frame_indices[i], frame_indices[i+1]))
            frame_pairs.append((i, i+2, frame_indices[i], frame_indices[i+2]))
        frame_pairs.append((len(frame_indices)-3, len(frame_indices)-2, frame_indices[len(frame_indices)-3], frame_indices[len(frame_indices)-2]))
        frame_pairs.append((len(frame_indices)-3, len(frame_indices)-1, frame_indices[len(frame_indices)-3], frame_indices[len(frame_indices)-1]))
        frame_pairs.append((len(frame_indices)-2, len(frame_indices)-1, frame_indices[len(frame_indices)-2], frame_indices[len(frame_indices)-1]))
        
        frame0_idx = frame_indices[0]
        frame0_path = Path("images") / f"video{vid}" / f"{frame0_idx}.png"
        for frame_pair in frame_pairs:
            frame1_idx = frame_indices[frame_pair[0]]
            frame2_idx = frame_indices[frame_pair[1]]
            frame1_path = Path("images") / f"video{vid}" / f"{frame1_idx}.png"
            frame2_path = Path("images") / f"video{vid}" / f"{frame2_idx}.png"
            
            img_list = ref_frame_path_list + [frame0_path] + [frame1_path] + [frame2_path]
            progress = frame_pair[3] - frame_pair[2]
            answer = '+' + str(progress) if progress > 0 else str(progress)
            
            entry = {
                "id": f"video{vid}_{frame1_idx}_to_{frame2_idx}_{str(uuid.uuid4())[:8]}",
                "image": [str(p) for p in img_list],
                "conversations": [],
            }
            
            q = {"from": "human", "value": question}
            a = {"from": "gpt", "value": answer}
            entry["conversations"].append(q)
            entry["conversations"].append(a)
            
            jsonl_list.append(entry)
            
            img_list_inverse = ref_frame_path_list + [frame0_path] + [frame2_path] + [frame1_path]
            progress_inverse = frame_pair[2] - frame_pair[3]
            answer_inverse = '+' + str(progress_inverse) if progress_inverse > 0 else str(progress_inverse)
            
            entry_inverse = {
                "id": f"video{vid}_{frame2_idx}_to_{frame1_idx}_{str(uuid.uuid4())[:8]}",
                "image": [str(p) for p in img_list_inverse],
                "conversations": [],
            }
            
            q = {"from": "human", "value": question}
            a = {"from": "gpt", "value": answer_inverse}
            entry_inverse["conversations"].append(q)
            entry_inverse["conversations"].append(a)
        
            jsonl_list.append(entry_inverse)
                
    # jsonl_file_name = f'train_rm_{len(jsonl_list)}.jsonl'
    jsonl_file_name = f'train_rm.jsonl'
    jsonl_file_path = os.path.join(str(out_dir), jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in jsonl_list:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 

if __name__ == "__main__":
    main()