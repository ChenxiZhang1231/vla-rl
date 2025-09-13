import json
import os
import statistics
import numpy as np
import cv2
import base64
import argparse
from typing import List, Tuple, Optional, Dict
import re
from dataclasses import dataclass
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/LIBERO")
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev")
from libero.libero import benchmark
from verl.utils.prompt_utils.prompt import build_system_prompt

@dataclass
class RewardTask:
    frames: List[str]
    description: str
    score_gt: int
    video_name: str
    score_vlm: int = 0
    score_text: str = ""
    
def fetch_one_reward_sync(client, task: RewardTask, task_index: int) -> tuple[int, float, str]:
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
    system_prompt = build_system_prompt(mode="v4") 

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
        completion = client.chat.completions.create(
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
        reward = parse_reward_from_box(response_text)  # 你已有的解析函数
        return task_index, reward, response_text

    except Exception as e:
        print(f"Task {task_index} - Callback API Error: {e}")
        return task_index, 0.0, "Empty"

# def parse_reward_from_box(response: str) -> float:
#     """
#     Parses a response to find a boxed answer (\box{...} or \boxed{...}) and returns a reward.
#     Returns 1.0 if "success" is found inside the box, otherwise 0.0.
#     """
#     # The new regex r"\\box(ed)?\{(.*?)\}" handles both cases.
#     # (ed)? makes the letters "ed" an optional group.
#     # The content ("success" or "failure") is now in the second capture group, group(2).
#     match = re.search(r"\\box(ed)?\{(.*?)\}", response.lower())
    
#     if match:
#         # group(1) would be "ed" or None, group(2) is the content we want.
#         answer = match.group(2).strip()
#         if answer == "success":
#             return 1.0
            
#     return 0.0

def parse_reward_from_box(response: str) -> float:
    """
    """
    match = re.search(r"\\box\{(.*?)\}", response.lower())
    if match:
        answer = match.group(1).strip()
        if answer == "success":
            return 1.0
    return 0.0


def process_video_to_base64_frames(video_path: Path, num_frames: int = 10) -> list[str]:
    """
    读取视频文件，均匀下采样到指定帧数，并返回Base64编码的帧列表。

    Args:
        video_path (Path): 视频文件的路径。
        num_frames (int): 要采样的帧数。

    Returns:
        list[str]: Base64编码的JPEG图像字符串列表。
    """
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    base64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # resized_frame = cv2.resize(frame, (128, 128))
            # _, buffer = cv2.imencode('.jpg', resized_frame)
            _, buffer = cv2.imencode('.jpg', frame)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")

    cap.release()
    return base64_frames

def get_rewards_from_judge_batch_sync(
    client,
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
            executor.submit(fetch_one_reward_sync, client, task, i): i
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

    
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def is_video_file(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS

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

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_binary_like(scores: List[float]) -> bool:
    # 全是 0/1 则视为二值
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
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 也叫 TPR / Sensitivity
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0     # Specificity
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    bal_acc = (recall + tnr) / 2.0
    # Matthews Correlation Coefficient
    import math
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr_specificity": tnr,
        "fnr": fnr,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
    }

def auc_mann_whitney(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """
    AUROC 的一种等价计算：MW 统计量 / (pos * neg)
    对分数的相同值使用平均秩（tie-aware）。无 sklearn 依赖。
    """
    pos_idx = [i for i, y in enumerate(y_true) if y == 1]
    neg_idx = [i for i, y in enumerate(y_true) if y == 0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return None

    # 排名（从小到大），相同分数取平均秩
    pairs = sorted([(s, i) for i, s in enumerate(y_score)], key=lambda x: x[0])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # rank 从 1 开始
        for k in range(i, j):
            _, idx = pairs[k]
            ranks[idx] = avg_rank
        i = j

    sum_ranks_pos = sum(ranks[i] for i in pos_idx)
    # MW 统计：U = sum_ranks_pos - n_pos*(n_pos+1)/2
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)

def compute_all_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, object]:
    if is_binary_like(y_score):
        y_pred = [int(s) for s in y_score]
        auc = None  # 全 0/1 分数没法产生阈值曲线（只有一个点），AUROC 没意义
    else:
        y_pred = to_pred_labels(y_score, threshold)
        auc = auc_mann_whitney(y_true, y_score)

    cm = compute_confusion(y_true, y_pred)
    basic = compute_basic_metrics(cm)
    return {"threshold": threshold, "confusion": cm, "metrics": basic, "auroc": auc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on rollout videos.")
    parser.add_argument("--eval_folder", type=str, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/rollouts/debug",
                        help="存放评测视频的目录")
    parser.add_argument("--output_dir", type=str, default="output", help="结果保存目录（CSV、JSON）")
    parser.add_argument("--tag", type=str, default="v4-72b", help="")
    parser.add_argument("--task_name", type=str, default="libero_spatial", help="benchmark 名")
    parser.add_argument("--num_frames", type=int, default=20, help="每段视频抽帧数")
    parser.add_argument("--threshold", type=float, default=0.5, help="连续分数二值化阈值")
    parser.add_argument("--max_workers", type=int, default=64, help="judge 并行线程数")
    parser.add_argument("--dry_run", action="store_true", help="只做解析不过 judge（用于快速检查）")
    args = parser.parse_args()

    eval_folder = Path(args.eval_folder)
    out_dir = Path(os.path.join(args.output_dir, args.tag))
    safe_makedirs(out_dir)

    client = openai.OpenAI(
        base_url="http://localhost:18901/v1", # 请确保这是您的服务地址
        api_key="not-needed"
    )


    # 加载 benchmark 词典
    task_lang_lookup = {}
    if benchmark is not None:
        try:
            bdict = benchmark.get_benchmark_dict()
            if args.task_name in bdict:
                suite = bdict[args.task_name]()
                # suite.get_task(task_id) -> (something, language)
                task_lang_lookup["__suite__"] = suite
        except Exception as e:
            print(f"[WARN] benchmark 加载失败：{e}")

    # 收集视频
    videos = [f for f in os.listdir(eval_folder) if is_video_file(f)]
    if not videos:
        raise FileNotFoundError(f"目录中未找到视频: {eval_folder}")

    tasks: List[RewardTask] = []
    rows = []  # 用于落盘 CSV
    for name in tqdm(sorted(videos)):
        video_path = eval_folder / name
        try:
            gt = parse_success_from_name(name)
        except Exception:
            print(f"[SKIP] {name}（未找到 success=True/False）")
            continue

        task_id = parse_task_id_from_name(name)
        if "__suite__" in task_lang_lookup and task_id is not None:
            try:
                lang = task_lang_lookup["__suite__"].get_task(task_id)[1]
                task_lang = str(lang)
            except Exception:
                task_lang = ""
        else:
            task_lang = ""

        # 转帧
        frames = process_video_to_base64_frames(video_path, num_frames=args.num_frames)

        tasks.append(RewardTask(
            frames=frames,
            description=task_lang,
            score_gt=gt,
            video_name=name,
        ))

        rows.append({
            "video_name": name,
            "task_id": task_id,
            "task_lang": task_lang,
            "score_gt": gt,
        })

    if args.dry_run:
        print(f"[DRY-RUN] 已解析 {len(tasks)} 段视频，停止于此。")
        return

    # 批量评测
    scores, texts = get_rewards_from_judge_batch_sync(client, tasks, max_workers=args.max_workers)
    if len(scores) != len(tasks):
        raise RuntimeError(f"返回分数数量与任务数量不一致: {len(scores)} vs {len(tasks)}")
    # 回填
    for i, (s, t) in enumerate(zip(scores, texts)):
        tasks[i].score_vlm = float(s)
        tasks[i].score_text = str(t)

    # 统计
    y_true = [t.score_gt for t in tasks]
    y_score = [t.score_vlm for t in tasks]
    stat = compute_all_metrics(y_true, y_score, threshold=args.threshold)

    # 落盘
    # 1) 明细 CSV
    import csv
    csv_path = out_dir / "reward_eval_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["video_name", "task_id", "task_lang", "score_gt", "score_vlm", "pred_label", "score_text"]
        writer.writerow(header)
        # 计算预测标签（用于明细）
        if is_binary_like(y_score):
            y_pred = [int(s) for s in y_score]
        else:
            y_pred = to_pred_labels(y_score, stat["threshold"])
        for t, yl in zip(tasks, y_pred):
            writer.writerow([t.video_name, parse_task_id_from_name(t.video_name), t.description, t.score_gt, t.score_vlm, yl, t.score_text])

    # 2) 汇总 JSON
    json_path = out_dir / "reward_eval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    # 3) 同时保存一个可读的 txt
    txt_path = out_dir / "reward_eval_summary.txt"
    cm = stat["confusion"]
    m = stat["metrics"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("== Confusion Matrix ==\n")
        f.write(f"TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}\n\n")
        f.write("== Metrics ==\n")
        f.write(f"Accuracy={m['accuracy']:.4f}\nPrecision={m['precision']:.4f}\nRecall/TPR={m['recall']:.4f}\nF1={m['f1']:.4f}\n")
        f.write(f"FPR={m['fpr']:.4f}\nTNR/Specificity={m['tnr_specificity']:.4f}\nFNR={m['fnr']:.4f}\n")
        f.write(f"Balanced-Acc={m['balanced_accuracy']:.4f}\nMCC={m['mcc']:.4f}\n")
        f.write(f"Threshold={stat['threshold']}\n")
        f.write(f"AUROC={stat['auroc'] if stat['auroc'] is not None else 'N/A'}\n")

    print(f"[OK] 评测完成，共 {len(tasks)} 段。")
    print(f"明细: {csv_path}")
    print(f"汇总: {json_path}")
    print(f"摘要: {txt_path}")


if __name__ == "__main__":
    main()