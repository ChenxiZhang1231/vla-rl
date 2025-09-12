import openai
import re
from dataclasses import dataclass
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from pathlib import Path
import time
import cv2
import numpy as np

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
            _, buffer = cv2.imencode('.jpg', frame)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")

    cap.release()
    return base64_frames

@dataclass
class RewardTask:
    frames: List[str]
    description: str

client = openai.OpenAI(
    base_url="http://localhost:18901/v1", # 请确保这是您的服务地址
    api_key="not-needed"
)

# --- 2. (与之前相同) 健壮的响应解析函数 ---
def parse_reward_from_box(response: str) -> float:
    """
    从模型的响应中解析出 \box{} 里的内容，并转换为奖励分数。
    """
    match = re.search(r"\\box\{(.*?)\}", response.lower())
    if match:
        answer = match.group(1).strip()
        if answer == "success":
            return 1.0
    return 0.0

# --- 3. 用于单次推理的 "worker" 函数 (这是被多线程调用的核心) ---
def fetch_one_reward_sync(task: RewardTask, task_index: int) -> tuple[int, float]:
    """
    执行单次API调用，并返回结果和原始索引。
    这个函数会被多个线程并发执行。
    """
    if not task.frames:
        print(f"Task {task_index}: Input list is empty")
        return task_index, 0.0

    prompt_text = f"""Please analyze the following image sequence to determine task completion.\n
Task Description: {task.description}\n
Input: A sequence of {len(task.frames)} temporally ordered image frames.\n
Instruction: Based on the visual evidence in the sequence, judge if the task objective has been met.\n
Required Output Format: Output your thought process first, then output the final answer. Your final answer must be strictly one of the following two words: 'Success' or 'Failure', and it must be enclosed in \\box{{}}.\n
Example: \\box{{Success}}"""

#     prompt_text = f"""请分析以下图像序列以判断任务是否完成。
# 任务描述：{task.description}
# 输入：一个由 {len(task.frames)} 帧按时间顺序排列的图像序列。
# 指令要求：请根据序列中的视觉证据，判断任务目标是否已经达成。
# 要求输出格式：先输出你的思考过程，然后输出最终答案。你的最终答案必须也只能是'成功'或'失败'这两个词中的一个，并且必须用 \\box{{}} 包裹。
# 例如：\\box{{成功}}"""

    content = [{"type": "text", "text": prompt_text}]
    for frame_url in task.frames:
        content.append({"type": "image_url", "image_url": {"url": frame_url}})

    try:
        print(f"Sending request for task {task_index}...")
        completion = client.chat.completions.create(
            model="judge",
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.0
        )
        response_text = completion.choices[0].message.content
        print(f"Task {task_index} response: '{response_text}'")
        reward = parse_reward_from_box(response_text)
        return task_index, reward, response_text

    except Exception as e:
        print(f"Task {task_index} - Callback API Error: {e}")
        return task_index, 0.0, "Empty"

# --- 4. 核心的批量推理函数 (同步版本) ---
def get_rewards_from_judge_batch_sync(
    tasks: List[RewardTask], 
    max_workers: int = 10
) -> List[float]:
    """
    使用多线程并发地为多个任务获取奖励判断。
    这个函数是同步的，调用它会阻塞直到所有结果返回。

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
            executor.submit(fetch_one_reward_sync, task, i): i
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

# --- 5. 如何调用这个同步批量函数 ---
if __name__ == "__main__":
    print("Starting synchronous batch reward inference...")
    video_file = Path("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/world_model/ActionWorldModel/output_debug/generated_video_action_whole.mp4")
    sampled_frames = process_video_to_base64_frames(video_file, num_frames=10)
    
    # desc = "picked up the bowl on the table"
    desc = "picked up the bowl"
    # desc = "抓起了一个碗"
    tasks_to_process = [
        RewardTask(frames=sampled_frames, description=desc)
        for i in range(256)
    ]
    start_time = time.time()
    rewards, results_text = get_rewards_from_judge_batch_sync(tasks_to_process, max_workers=64)
    gen_time = time.time() - start_time
    print(f"Model inference time: {gen_time}")
    

    # print("\n--- Batch Inference Complete ---")
    for i, (task, reward) in enumerate(zip(tasks_to_process, rewards)):
        print(f"Task {i} ('{task.description}') -> Reward: {reward}")