import openai
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


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def get_reward_from_judge(frames: list[str], task_description: str) -> float:
    """
    """
    if not frames:
        print("Input list is empty")
        return 0.0

    prompt_text = f"""Please analyze the following image sequence to determine task completion.\n
Task Description: {task_description}\n
Input: A sequence of {len(frames)} temporally ordered image frames.\n
Instruction: Based on the visual evidence in the sequence, judge if the task objective has been met.\n
Required Output Format: Your final answer must be strictly one of the following two words: 'Success' or 'Failure', and it must be enclosed in \\box{{}}.\n
Example: \\box{{Success}}"""

#     prompt_text = f"""请分析以下图像序列以判断任务是否完成。
# 任务描述：{task_description}
# 输入：一个由 {len(frames)} 帧按时间顺序排列的图像序列。
# 指令要求：请根据序列中的视觉证据，判断任务目标是否已经达成。
# 要求输出格式：你的最终答案必须也只能是'成功'或'失败'这两个词中的一个，并且必须用 \\box{{}} 包裹。
# 例如：\\box{{成功}}"""

    # prompt_text = f"""机械臂是否抓起来一个碗"""

    content = [{"type": "text", "text": prompt_text}]
    for frame_url in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_url}
        })

    try:
        completion = client.chat.completions.create(
            model="judge",
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=500,
            temperature=0.0
        )
        response_text = completion.choices[0].message.content.strip().lower()
        print(f"Model response: '{response_text}'")
        return response_text

    except Exception as e:
        print(f"Callback API Error: {e}")
        return 0.0



if __name__ == "__main__":
    client = openai.OpenAI(
        base_url="http://localhost:18901/v1",
        api_key="not-needed"
    )
    
    # try:
    #     image_path = Path("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/stitched_observations_grid.png")
    #     base64_image = image_to_base64(image_path)
    #     image_url = f"data:image/jpeg;base64,{base64_image}"
    # except FileNotFoundError:
    #     print(f"错误: 请将 'path/to/your/screenshot.jpg' 替换为一张真实图片的路径。")
    #     exit()
    
    video_file = Path("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/world_model/ActionWorldModel/output_debug/generated_video_action_whole.mp4")
    sampled_frames = process_video_to_base64_frames(video_file, num_frames=10)
    

            
    task_desc = "抓起了一个碗"

    for i in range(10):
        start_time = time.time()
        reward = get_reward_from_judge(sampled_frames, task_desc)
        infer_time = time.time() - start_time

        print(f"Model infer time: {infer_time}")

    print(f"\n获取到的奖励分数为: {reward}")