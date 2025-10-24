import os
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def check_ffmpeg_exists():
    """检查 ffmpeg 命令是否存在于系统 PATH 中。"""
    if shutil.which("ffmpeg") is None:
        print("错误: 'ffmpeg' 命令未找到。")
        print("请确保 FFmpeg 已正确安装，并且其路径已添加到系统的 PATH 环境变量中。")
        return False
    return True

def process_single_video(input_file: Path, output_file: Path):
    """
    处理单个视频文件的函数，供并行工作进程调用。
    
    返回一个元组 (bool, str, str or None)，分别代表：
    - 是否成功 (True/False)
    - 输入文件的名称
    - 如果失败，则为错误信息；否则为 None
    """
    command = [
        "ffmpeg",
        "-i", str(input_file),
        "-c:v", "libx264",
        "-preset", "medium",  # 在速度和质量之间取得良好平衡
        "-crf", "23",
        "-y",  # 自动覆盖输出文件
        "-hide_banner", # 隐藏 ffmpeg 的版本信息
        "-loglevel", "error", # 只在发生真正错误时打印日志
        str(output_file)
    ]
    
    try:
        # 执行命令，如果失败则会抛出异常
        subprocess.run(
            command,
            check=True,
            capture_output=True, # 捕获输出
            text=True,
            encoding='utf-8'
        )
        return True, input_file.name, None
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，返回错误信息
        error_message = f"FFmpeg 错误:\n{e.stderr}"
        return False, input_file.name, error_message

def repair_videos_parallel(input_folder: str, output_folder: str, num_workers: int = None):
    """
    使用并行处理，批量修复文件夹中的所有 .mp4 视频文件。

    Args:
        input_folder (str): 包含损坏视频的源文件夹路径。
        output_folder (str): 用于保存修复后视频的目标文件夹路径。
        num_workers (int, optional): 使用的并行工作进程数。
                                     如果为 None，则默认为系统 CPU 核心数。
                                     建议设置为核心数-1，以保留系统流畅。
    """
    if not check_ffmpeg_exists():
        return

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.is_dir():
        print(f"错误：输入文件夹不存在 -> {input_folder}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_path.glob("*.mp4"))
    
    if not video_files:
        print(f"在 {input_folder} 中没有找到任何 .mp4 文件。")
        return

    # 如果未指定工作进程数，则使用 CPU 核心数
    if num_workers is None:
        # os.cpu_count() 可能为 None，提供一个默认值
        num_workers = os.cpu_count() or 4 
        print(f"未指定工作进程数，将使用所有可用的 CPU 核心: {num_workers} 个")

    print(f"找到 {len(video_files)} 个视频文件，将使用 {num_workers} 个进程并行处理...")

    tasks = []
    for video_file in video_files:
        output_file_path = output_path / video_file.name
        tasks.append((video_file, output_file_path))

    success_count = 0
    failure_count = 0
    failed_files = []

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 创建一个 future 对象列表
        futures = {executor.submit(process_single_video, in_file, out_file): in_file for in_file, out_file in tasks}
        
        # 使用 tqdm 创建进度条，并处理已完成的任务
        for future in tqdm(as_completed(futures), total=len(video_files), desc="修复视频", unit="个"):
            try:
                success, filename, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    failed_files.append((filename, error_msg))
            except Exception as exc:
                # 处理任务执行期间可能发生的其他异常
                print("aa")
                filename = futures[future].name
                failure_count += 1
                failed_files.append((filename, str(exc)))

    # 打印最终的总结报告
    print("\n" + "="*30)
    print("      处理完成！")
    print("="*30)
    print(f"  成功: {success_count} 个")
    print(f"  失败: {failure_count} 个")
    print(f"修复后的文件已保存至: {output_folder}")
    
    if failed_files:
        print("\n--- 失败文件列表 ---")
        for filename, error in failed_files:
            print(f"\n文件名: {filename}")
            print(f"  错误信息: {error}")
        print("--------------------")


if __name__ == '__main__':
    # --- 请在这里配置您的文件夹路径 ---
    SOURCE_FOLDER = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/gen_rm_for_wm_10_repeat_rm_data"
    DESTINATION_FOLDER = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/gen_rm_for_wm_10_repeat_rm_data-fixedbug"

    # --- 配置并行数量 ---
    # 设置为 None 来使用所有 CPU 核心。
    # 如果您的 CPU 有 16 个核心，可以设置为 12 或 14，以避免系统完全卡死。
    NUM_PARALLEL_JOBS = 8

    # 执行修复函数
    repair_videos_parallel(SOURCE_FOLDER, DESTINATION_FOLDER, num_workers=NUM_PARALLEL_JOBS)