import argparse
from pathlib import Path
import os
import re
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm 
import cv2 
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def process_video_and_create_entry(video_info: dict):
    """
    处理单个视频文件，并返回构建好的JSON entry。
    这个函数将在一个独立的进程中运行。
    
    Args:
        video_info (dict): 包含处理单个视频所需全部信息的字典。
    """
    # 从字典中解包所有需要的参数
    vid = video_info["vid"]
    name = video_info["name"]
    video_folder = video_info["video_folder"]
    frames_save_path = video_info["frames_save_path"]
    lang = video_info["lang"]
    num_frames = video_info["num_frames"]
    
    try:
        video_path = video_folder / name
        
        complete = parse_success_from_name(name)
        finish_step = parse_finish_step_from_name(name)
        task_id = parse_task_id_from_name(name)
        # lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
        task_lang = str(lang)
        
        frames, frame_indices = process_video_to_frames(video_path, num_frames=num_frames)

        # 如果未能提取到帧，则直接返回None表示失败
        if not frames:
            print(f"警告: 未能从视频 {name} 中提取任何帧。")
            return None

        # 3. 【耗时操作】保存帧为图片
        for frame, frame_id in zip(frames, frame_indices):
            frame_save_path_full = frames_save_path / f"video{vid}" / f"{frame_id}.png"
            safe_makedirs(frame_save_path_full.parent)
            cv2.imwrite(str(frame_save_path_full), frame)
        
        # 4. 构建JSON entry
        entry = {
            "id": f"video{vid}",
            "image": [f"images/video{vid}/{frame_id}.png" for frame_id in frame_indices],
            "width_list": [128] * len(frame_indices),
            "height_list": [128] * len(frame_indices),
            "conversations": [
                {"from": "human", "value": build_question(task_lang, step_ids=frame_indices)},
                {"from": "gpt", "value": build_answer(complete=complete, finish_step_raw=finish_step, sampled_indices=frame_indices)}
            ],
        }
        
        return entry
    
    except Exception as e:
        # 捕获任何可能的错误，打印并返回None
        print(f"处理视频 {name} 时发生错误: {e}")
        return None


def run_parallel_processing(args):
    """
    主函数，用于设置和运行并行处理流程。
    """
    # --- 配置区域 ---
    video_folder = Path(args.video_folder)
    frames_save_path = Path(args.output_dir) / "images"


    task_lang_lookup = {}
    if benchmark is not None:
        bdict = benchmark.get_benchmark_dict()
        if args.task_name in bdict:
            suite = bdict[args.task_name]()
            task_lang_lookup["__suite__"] = suite
            
    videos = [f for f in os.listdir(video_folder) if is_video_file(f)]
    if not videos:
        raise FileNotFoundError(f"目录中未找到视频: {video_folder}")

    # 准备要分发给每个工作进程的任务列表
    tasks_to_process = []
    for vid, name in enumerate(videos):
        task_id = parse_task_id_from_name(name)
        lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
        task_info = {
            "vid": vid,
            "name": name,
            "video_folder": video_folder,
            "frames_save_path": frames_save_path,
            "lang": lang,
            "num_frames": args.num_frames,
        }
        tasks_to_process.append(task_info)
        
    jsonl_list = []
    
    # --- 并行执行 ---
    # 设置为 None 来使用所有 CPU 核心。可以设置为具体数字，如 12。
    num_workers = args.num_workers
    # process_video_and_create_entry(tasks_to_process[0])
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_video_and_create_entry, task): task for task in tasks_to_process}
        
        # 使用tqdm显示进度，并收集已完成任务的结果
        for future in tqdm(as_completed(futures), total=len(videos), desc="处理视频"):
            result_entry = future.result()
            # 只有当任务成功返回entry时才添加到列表中
            if result_entry:
                jsonl_list.append(result_entry)

    # 【重要】并行处理的结果是无序的。如果需要，处理完后根据id进行排序。
    jsonl_list.sort(key=lambda x: int(x['id'].replace('video', '')))

    # --- 后续处理 ---
    # 例如，将结果写入一个 .jsonl 文件
    output_jsonl_path = Path(args.output_dir) / "data.jsonl"
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_list:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n处理完成！共生成 {len(jsonl_list)} 条记录。")
    print(f"数据已保存到: {output_jsonl_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on rollout videos.")
    parser.add_argument("--video_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/gen_rm_for_wm_goal_repeat_rm_data-fixedbug",
                        help="")
    parser.add_argument("--output_dir", type=str, default="/inspire/ssd/project/robotsimulation/public/data/rm_train_jsonl_wm/rm_train_goal", help="")
    parser.add_argument("--num_frames", type=int, default=50, help="")
    parser.add_argument("--task_name", type=str, default="libero_spatial", help="benchmark 名")
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    out_dir = Path(args.output_dir)
    frames_save_path = out_dir / "images"
    safe_makedirs(out_dir)
    safe_makedirs(frames_save_path)
    
    class MockArgs:
        video_folder = args.video_folder
        output_dir = args.output_dir
        num_frames = args.num_frames
        num_workers = 32
        task_name = args.task_name
    
    run_parallel_processing(MockArgs())

    # jsonl_list = []
    # for vid, name in tqdm(enumerate(sorted(videos))):
    #     video_path = video_folder / name
    #     complete = parse_success_from_name(name)
    #     finish_step = parse_finish_step_from_name(name)
    #     task_id = parse_task_id_from_name(name)
    #     lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
    #     task_lang = str(lang)
            
    #     frames, frame_indices = process_video_to_frames(video_path, num_frames=args.num_frames)
        
    #     for frame, frame_id in zip(frames, frame_indices):
    #         frame_save_path = frames_save_path / f"video{vid}" / f"{frame_id}.png"
    #         safe_makedirs(frame_save_path.parent)
    #         cv2.imwrite(str(frame_save_path), frame)
        
    #     entry = {
    #         "id": f"video{vid}",
    #         "image": [],
    #         "width_list": [],
    #         "height_list": [],
    #         "conversations": [],
    #     }

    #     for i in range(args.num_frames):
    #         entry['image'].append(f"images/video{vid}/{frame_indices[i]}.png")
    #         entry['width_list'].append(128)
    #         entry['height_list'].append(128)

    #     question = build_question(task_lang, step_ids=frame_indices)
    #     answer = build_answer(complete=complete, finish_step_raw=finish_step, sampled_indices=frame_indices)

    #     q = {"from": "human", "value": question}
    #     a = {"from": "gpt", "value": answer}
    #     entry["conversations"].append(q)
    #     entry["conversations"].append(a)
        
    #     jsonl_list.append(entry)
                
    # jsonl_file_name = f'train_rm_{len(jsonl_list)}.jsonl'
    # jsonl_file_path = os.path.join(str(out_dir), jsonl_file_name)

    # with open(jsonl_file_path, "w") as jsonl_file:
    #     for entry in jsonl_list:
    #         json.dump(entry, jsonl_file)
    #         jsonl_file.write('\n') 

if __name__ == "__main__":
    main()