#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DROID -> 统一切片/对齐导出器

输入目录结构（示例）:
  <scene_root>/
    success/
      2023-08-01/
        12-30-15/
          trajectory.h5
          *.json (包含 *_cam_serial / uuid / current_task 等)
          recordings/
            MP4/
              <serial>-stereo.mp4 或 <serial>.mp4
    failure/
      ...

输出:
  <output_dir>/<scene_name>/<split>/<date>/<timestamp>/
    clips/<stem>_clip_00000.mp4 ...
    metadata/<stem>_clip_00000.npz ...
    dataset.json  （[{caption, media_path}, ...]）

npz 字段：
- extrinsics: (F,4,4)   # 世界->相机（若源为相机->世界，会自动取逆）
- intrinsics: (3,3)
- action:     (F,2,A)   # 第二路全 0
- obs_state:  (F,2,8)   # [cartesian(6), gripper(1), pad(1)], 第二路全 0
- start_frame, clip_len, fps
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# ----------------- 可调参数 -----------------
EXTRINSIC_IS_WORLD_TO_CAMERA = False   # DROID camera_extrinsics 存的是 [tx,ty,tz,euler(xyz)] in world? 如果是 camera->world 就设 False（会自动 inv）
QUAT_IS_W_FIRST = False                # 若未来你要画夹爪朝向，可切换四元数顺序；当前未用

# --------------------------------------------------
# 基础工具
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return res.stdout

def ffprobe_fps(path: Path) -> float:
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=avg_frame_rate","-of","default=nokey=1:noprint_wrappers=1",str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return 0.0
    fr = res.stdout.strip()
    try:
        if "/" in fr:
            num, den = fr.split("/")
            num, den = float(num), float(den)
            return 0.0 if den == 0 else num/den
        return float(fr)
    except:
        return 0.0

def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return 0.0
    try:
        return float(res.stdout.strip())
    except:
        return 0.0

def ffprobe_safe_total_frames(path: Path, fps: int) -> int:
    duration = ffprobe_duration(path)
    if duration <= 0:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n
        return 0
    return max(0, int(round(duration * max(fps, 1))))

def resample_to_fps(input_path: Path, output_path: Path, fps: int, crf: int = 20, preset: str = "veryfast"):
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg","-y","-i",str(input_path),"-r",str(fps),
        "-c:v","libx264","-crf",str(crf),"-preset",preset,"-pix_fmt","yuv420p","-an",str(output_path)
    ]
    run_cmd(cmd)

def resample_to_fps_left_half(
    input_path: Path,
    output_path: Path,
    fps: int,
    crf: int = 20,
    preset: str = "veryfast",
    threads: int = 1,
):
    """
    1) 取左半幅（竖直一刀两半，保留左边）
    2) 重采样到目标 fps（恒定帧率）
    3) 重新编码为 H.264
    """
    ensure_dir(output_path.parent)

    # 裁切：宽度=原始宽度的一半，高度不变；左上角起点(0,0)
    # 重采样：在滤镜里做 fps，搭配 -vsync cfr，行为更可控
    vf = f"crop=iw/2:ih:0:0,fps={fps}"

    cmd: List[str] = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter:v", vf,        # 先裁左半，再按时间戳抽帧到目标fps
        "-vsync", "cfr",        # 强制恒定帧率
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", str(threads),
        "-an",
        str(output_path),
    ]
    run_cmd(cmd)
    
def cut_clip_by_frames(input_path: Path, start_frame: int, num_frames: int, fps: int,
                       output_path: Path, crf: int = 20, preset: str = "veryfast"):
    start_time = start_frame / max(fps, 1e-6)
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg","-y","-ss",f"{start_time:.6f}","-i",str(input_path),
        "-frames:v",str(num_frames),"-c:v","libx264","-crf",str(crf),
        "-preset",preset,"-pix_fmt","yuv420p","-an",str(output_path)
    ]
    run_cmd(cmd)

def indices_for_resampled(num_resampled: int, orig_fps: float, new_fps: float, n_orig: int) -> np.ndarray:
    t_res = np.arange(num_resampled, dtype=np.float64) / max(new_fps, 1e-6)
    idx = np.round(t_res * max(orig_fps, 1e-6)).astype(int)
    return np.clip(idx, 0, n_orig - 1)

def euler_to_T(tx_ty_tz_exyz: np.ndarray) -> np.ndarray:
    """[tx,ty,tz, ex,ey,ez] -> 4x4"""
    t = tx_ty_tz_exyz[:3]
    r = Rotation.from_euler("xyz", tx_ty_tz_exyz[3:], degrees=False).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = r
    T[:3, 3] = t
    return T

# --------------------------------------------------
# DROID 装载
def find_episode_video(recordings_mp4: Path, serial: str) -> Path:
    c1 = recordings_mp4 / f"{serial}-stereo.mp4"
    c2 = recordings_mp4 / f"{serial}.mp4"
    if c1.exists(): return c1
    if c2.exists(): return c2
    raise FileNotFoundError(f"No mp4 for serial={serial} under {recordings_mp4}")

def scale_K(K: np.ndarray, src_wh: Tuple[int,int], dst_wh: Tuple[int,int]) -> np.ndarray:
    sw, sh = src_wh
    dw, dh = dst_wh
    sx = dw / sw
    sy = dh / sh
    S = np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=np.float64)
    K2 = K.copy()
    K2 = S @ K2
    return K2

def build_action_and_obs(trajectory_h5) -> Tuple[np.ndarray, np.ndarray]:
    """返回 action:(N,2,A7), obs_state:(N,2,8)"""
    obs = trajectory_h5['observation']
    act = trajectory_h5['action']

    proprio = obs['robot_state']['cartesian_position'][:]   # (N,6)
    abs_action = act['cartesian_position'][:]               # (N,6)

    rel_xyz = abs_action[:, :3] - proprio[:, :3]
    abs_R = Rotation.from_euler('xyz', abs_action[:, 3:], degrees=False).as_matrix()
    pro_R = Rotation.from_euler('xyz', proprio[:, 3:],   degrees=False).as_matrix()
    rel_R = abs_R @ np.transpose(pro_R, (0,2,1))
    rel_euler = Rotation.from_matrix(rel_R).as_euler('xyz', degrees=False)

    rel_action = np.concatenate([rel_xyz, rel_euler], axis=1)  # (N,6)
    gr = act['gripper_position'][:].reshape(-1,1)              # (N,1)
    rel_action7 = np.concatenate([rel_action, gr], axis=1)     # (N,7)

    # (N,2,7)
    action = np.stack([rel_action7, np.zeros_like(rel_action7)], axis=1)

    # obs_state 取 cartesian(6) + gripper(1) + pad(1) = 8
    gr_obs = obs['robot_state']['gripper_position'][:].reshape(-1,1)
    # pad1 = np.zeros_like(gr_obs)
    obs7 = np.concatenate([proprio, gr_obs], axis=1)      # (N,8)
    obs_state = np.stack([obs7, np.zeros_like(obs7)], axis=1)   # (N,2,8)

    return action.astype(np.float32), obs_state.astype(np.float32)

def build_extrinsics_per_frame(trajectory_h5, serial: str) -> np.ndarray:
    """从 observation/camera_extrinsics/{serial}_left 读取 [tx,ty,tz,ex,ey,ez] → (N,4,4)"""
    ds = trajectory_h5['observation']['camera_extrinsics'][f"{serial}_left"][:]  # (N,6)
    Ts = np.stack([euler_to_T(x) for x in ds], axis=0)                            # (N,4,4)
    # 若源是 camera->world，这里不改；在消费侧需要世界->相机可设 EXTRINSIC_IS_WORLD_TO_CAMERA=False 并在投影时取 inv
    return Ts.astype(np.float64)

EXTRINSIC_IS_WORLD_TO_CAMERA = False  # set False if your extrinsics are camera->world
QUAT_IS_W_FIRST = False               # set False if quaternions are (x,y,z,w)
AXIS_LEN_UNITS = 0.15                # length of drawn XYZ axes in world units (tweak as needed)
AXIS_THICKNESS = 8                   # px for axes
POINT_RADIUS = 8                     # px for effector point


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix. q shape (..., 4). Assumes (w,x,y,z) if QUAT_IS_W_FIRST else (x,y,z,w)."""
    if QUAT_IS_W_FIRST:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # normalize to be safe
    norm = np.sqrt(w*w + x*x + y*y + z*z) + 1e-8
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    # rotation matrix
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R

def world_to_cam_points(T_wc: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """Project world 3D points to camera coordinates.
    If EXTRINSIC_IS_WORLD_TO_CAMERA is False, invert T to obtain world->camera.
    Accepts a single point (3,) or a stack (..., M, 3). Returns matching shape with last dim 3.
    """
    Twc = T_wc
    if not EXTRINSIC_IS_WORLD_TO_CAMERA:
        Twc = np.linalg.inv(T_wc)

    R = Twc[..., :3, :3]
    t = Twc[..., :3, 3]

    # Single point: shape (3,)
    if Xw.ndim == 1 and Xw.shape[-1] == 3:
        x = Xw.reshape(3, 1)  # (3,1)
        Xc = (R @ x).squeeze(-1) + t  # (3,)
        return Xc

    # Multiple points: (..., M, 3)
    # Ensure last dims are (M,3)
    if Xw.ndim == 2 and Xw.shape[-1] == 3:
        # (M,3) -> (M,3,1)
        Xc = (R[..., None, :, :] @ Xw[:, :, None]).squeeze(-1) + t[None, :]
        return Xc
    else:
        # General batched case (..., M, 3)
        Xc = (R[..., None, :, :] @ Xw[..., :, :, None]).squeeze(-1) + t[..., None, :]
        return Xc

def cam_to_pixels(K: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Pin-hole projection. Xc (..., M, 3) -> (..., M, 2)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = np.clip(Xc[..., 2], 1e-6, None)
    u = fx * (Xc[..., 0] / Z) + cx
    v = fy * (Xc[..., 1] / Z) + cy
    return np.stack([u, v], axis=-1)

# def _scalar_to_bgr(v: float, vmin: float, vmax: float) -> tuple[int, int, int]:
#     if np.isnan(v):
#         v = vmin
#     t = (v - vmin) / (vmax - vmin + 1e-8)
#     t = float(np.clip(t, 0.0, 1.0))
#     gray = np.array([[int(round(t * 255))]], dtype=np.uint8)
#     bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[0, 0]
#     return int(bgr[0]), int(bgr[1]), int(bgr[2])  # (B,G,R)

def _scalar_to_bgr(
    v: float,
    v_open: float = 0.0,   # 打开时的读数（你的情况是 0）
    v_close: float = 50.0, # 关闭时的读数（你的情况是 40）
) -> tuple[int, int, int]:
    """
    将夹爪读数映射为颜色：越打开(接近 v_open) 颜色越“高端”（反向归一化）。
    返回 BGR (int,int,int) 以便 OpenCV 直接使用。
    """
    import numpy as np
    import cv2

    # NaN 直接按“最打开”的颜色处理（也可以换成固定颜色）
    if np.isnan(v):
        v = v_open

    # 计算“合拢比例” p ∈ [0,1]：0=完全打开，1=完全关闭
    denom = (v_close - v_open)
    if abs(denom) < 1e-8:
        p = 0.0
    else:
        p = (v - v_open) / denom
    p = float(np.clip(p, 0.0, 1.0))

    # 我们想要“越打开颜色越亮/越靠色条高端”，所以取 t = 1 - p
    t = 1.0 - p  # 0=最合，1=最开

    gray = np.array([[int(round(t * 255))]], dtype=np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])  # (B,G,R)

def _depth_to_radius(z: float,
                     radius_at_z_ref: float,
                     z_ref: float,
                     radius_min: int,
                     radius_max: int) -> int:
    if not np.isfinite(z) or z <= 1e-6:
        return radius_min
    r = radius_at_z_ref * (z_ref / z)
    r = int(np.clip(r, radius_min, radius_max))
    return max(r, 1)


def draw_overlay_on_clip(clip_path: Path, out_path: Path,
                         K: np.ndarray,
                         e_clip: np.ndarray,           # (F, 4, 4)
                         end_pos_clip: np.ndarray,     # (F, 2, 3)
                         end_quat_clip: np.ndarray,    # (F, 2, 4)
                         end_eff_clip: np.ndarray,     # (F, 2),
                         *,
                         vmin: float = 35.0,
                         vmax: float = 125.0,
                         radius_at_z_ref: int = 40,    # 在 z_ref 时的像素半径
                         z_ref: float = 1.0,           # 参考深度（与你场景单位一致，常见是米）
                         radius_min: int = 8,
                         radius_max: int = 140,
                         circle_alpha: float = 0.35
                         ) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {clip_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    F = e_clip.shape[0]
    axis = np.eye(3) * AXIS_LEN_UNITS  # 3 axis unit vectors scaled

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret or f >= F:
            break

        overlay = frame.copy()

        for ee in range(1):
            pos_w = end_pos_clip[f, ee, :]  # (3,)
            quat  = end_quat_clip[f, ee, :] # (4,)
            R_ee  = quat_to_rot(quat)       # (3,3)

            pts_axes_w = pos_w[None, :] + (R_ee @ axis).T  # (3,3)
            pts_w = np.vstack([pos_w[None, :], pts_axes_w])  # (4,3)

            Xc = world_to_cam_points(e_clip[f], pts_w)   # (4,3), 相机坐标
            uv = cam_to_pixels(K, Xc[None, ...])[0]      # (4,2), 像素坐标

            u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
            z = float(Xc[0, 2])
            circle_radius = _depth_to_radius(z, radius_at_z_ref, z_ref, radius_min, radius_max)

            val = float(end_eff_clip[f, ee]) * 100
            circle_color = _scalar_to_bgr(val, vmin, vmax)

            cv2.circle(overlay, (u0, v0), circle_radius, circle_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (u0, v0), POINT_RADIUS, (255, 255, 255), -1)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for k in range(3):
                uk, vk = int(round(uv[k + 1, 0])), int(round(uv[k + 1, 1]))
                cv2.line(frame, (u0, v0), (uk, vk), colors[k], AXIS_THICKNESS, cv2.LINE_AA)
            cv2.circle(frame, (u0, v0), circle_radius, (255,255,255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, circle_alpha, frame, 1.0 - circle_alpha, 0.0, dst=frame)
        writer.write(frame)
        f += 1

    writer.release()
    cap.release()
    
def draw_overlay_on_black(clip_path: Path, out_path: Path,
                         K: np.ndarray,
                         e_clip: np.ndarray,           # (F, 4, 4)
                         end_pos_clip: np.ndarray,     # (F, 2, 3)
                         end_quat_clip: np.ndarray,    # (F, 2, 4)
                         end_eff_clip: np.ndarray,     # (F, 2),
                         *,
                         vmin: float = 35.0,
                         vmax: float = 125.0,
                         radius_at_z_ref: int = 40,    # 在 z_ref 时的像素半径
                         z_ref: float = 1.0,           # 参考深度（与你场景单位一致，常见是米）
                         radius_min: int = 8,
                         radius_max: int = 140,
                         circle_alpha: float = 0.35
                         ) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {clip_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    F = e_clip.shape[0]
    axis = np.eye(3) * AXIS_LEN_UNITS  # 3 axis unit vectors scaled

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret or f >= F:
            break

        overlay = np.zeros_like(frame)
        frame = overlay.copy()

        for ee in range(1):
            pos_w = end_pos_clip[f, ee, :]  # (3,)
            quat  = end_quat_clip[f, ee, :] # (4,)
            R_ee  = quat_to_rot(quat)       # (3,3)

            pts_axes_w = pos_w[None, :] + (R_ee @ axis).T  # (3,3)
            pts_w = np.vstack([pos_w[None, :], pts_axes_w])  # (4,3)

            Xc = world_to_cam_points(e_clip[f], pts_w)   # (4,3), 相机坐标
            uv = cam_to_pixels(K, Xc[None, ...])[0]      # (4,2), 像素坐标

            u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
            z = float(Xc[0, 2])
            circle_radius = _depth_to_radius(z, radius_at_z_ref, z_ref, radius_min, radius_max)

            val = float(end_eff_clip[f, ee])
            circle_color = _scalar_to_bgr(val, vmin, vmax)

            cv2.circle(overlay, (u0, v0), circle_radius, circle_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (u0, v0), POINT_RADIUS, (255, 255, 255), -1)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for k in range(3):
                uk, vk = int(round(uv[k + 1, 0])), int(round(uv[k + 1, 1]))
                cv2.line(frame, (u0, v0), (uk, vk), colors[k], AXIS_THICKNESS, cv2.LINE_AA)
            cv2.circle(frame, (u0, v0), circle_radius, (255,255,255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, circle_alpha, frame, 1.0 - circle_alpha, 0.0, dst=frame)

        writer.write(frame)
        f += 1

    writer.release()
    cap.release()
    

def batch_axisangle2quat(vecs: np.ndarray) -> np.ndarray:
    """
    Converts a batch of scaled axis-angles to quaternions.

    Args:
        vecs (np.ndarray): (N, 3) array of axis-angle exponential coordinates.

    Returns:
        np.ndarray: (N, 4) array of (x, y, z, w) quaternions.
    """
    # 1. 批量计算所有向量的模长（角度）
    #    axis=-1 表示沿着最后一个维度（即3个坐标）计算模长。
    #    结果是一个 (N,) 形状的数组。
    angles = np.linalg.norm(vecs, axis=-1)

    # 2. 识别零旋转的情况
    #    np.isclose 会返回一个 (N,) 形状的布尔数组（mask）。
    zero_mask = np.isclose(angles, 0.0)

    # 3. 批量计算单位旋转轴
    #    为了避免“除以零”的警告，我们使用 np.divide 并通过 `where` 参数
    #    指定只在角度不为零的地方进行除法。
    #    angles[:, np.newaxis] 将 (N,) 的角度数组变形为 (N, 1)，以便与 (N, 3) 的 vecs 进行广播。
    #    对于角度为零的情况，轴向量将被置为 [0, 0, 0]，这是安全的。
    axes = np.divide(vecs, angles[:, np.newaxis], where=~zero_mask[:, np.newaxis], out=np.zeros_like(vecs))

    # 4. 批量应用转换公式
    half_angles = angles / 2.0
    sin_half_angles = np.sin(half_angles)
    cos_half_angles = np.cos(half_angles)

    # 初始化 (N, 4) 的输出数组
    quats = np.zeros((vecs.shape[0], 4))
    
    # 计算 [x, y, z] 部分
    quats[:, :3] = axes * sin_half_angles[:, np.newaxis]
    # 计算 w 部分
    quats[:, 3] = cos_half_angles

    # 5. 修正零旋转的情况
    #    使用之前创建的布尔掩码，将所有零旋转对应的四元数
    #    强制设置为单位四元数 [0, 0, 0, 1]。
    quats[zero_mask, :3] = 0.0
    quats[zero_mask, 3] = 1.0

    return quats

def process_one_camera_episode(
    mp4_path: Path,
    out_root: Path,
    nice_stem: str,
    task_lang: str,
    e_params_orig: np.ndarray,   # (N,4,4)
    k_params: np.ndarray,        # (3,3)
    action: np.ndarray,          # (N,2,A)
    obs_state: np.ndarray,       # (N,2,S)
    fps_out: int,
    clip_len: int,
    stride: int,
    crf: int,
    preset: str
) -> List[Dict[str,Any]]:
    # 1) 重采样
    resampled_path = out_root / "resampled" / f"{nice_stem}_resampled.mp4"
    ensure_dir(resampled_path.parent)
    if not resampled_path.exists():
        # resample_to_fps(mp4_path, resampled_path, fps=fps_out, crf=crf, preset=preset)
        resample_to_fps_left_half(mp4_path, resampled_path, fps=fps_out, crf=crf, preset=preset)

    # 2) 估计重采样后的总帧数
    total_frames_est = ffprobe_safe_total_frames(resampled_path, fps_out)

    # 3) 建立索引映射
    orig_fps = ffprobe_fps(mp4_path)
    N = e_params_orig.shape[0]
    idx_map = indices_for_resampled(total_frames_est, orig_fps if orig_fps>0 else fps_out, fps_out, N)
    e_res = e_params_orig[idx_map]
    action_res = action[idx_map]
    obs_state_res = obs_state[idx_map]

    # 4) 切片 + 存 npz
    clip_dir = out_root / "clips"
    meta_dir = out_root / "metadata"
    overlay_dir = out_root / "overlays"
    black_dir = out_root / "blacks"
    ensure_dir(clip_dir); ensure_dir(meta_dir)

    out_items: List[Dict[str,Any]] = []
    f0 = 0; i = 0; last_f0 = -1
    while f0 + clip_len <= total_frames_est:
        clip_path = clip_dir / f"{nice_stem}_clip_{i:05d}.mp4"
        if not clip_path.exists():
            cut_clip_by_frames(resampled_path, f0, clip_len, fps_out, clip_path, crf, preset)
        e_clip = e_res[f0:f0+clip_len]
        a_clip = action_res[f0:f0+clip_len]
        o_clip = obs_state_res[f0:f0+clip_len]
        npz_path = meta_dir / f"{nice_stem}_clip_{i:05d}.npz"
        np.savez_compressed(npz_path,
            extrinsics=e_clip, intrinsics=k_params,
            action=a_clip, obs_state=o_clip,
            start_frame=f0, clip_len=clip_len, fps=fps_out
        )
        overlay_path = overlay_dir / f"{nice_stem}_clip_{i:05d}_overlay.mp4"
        black_path = black_dir / f"{nice_stem}_clip_{i:05d}_black.mp4"
        try:
            pos_clip = o_clip[:, :, 0:3]
            rot_clip = o_clip[:, :, 3:6]
            eff_clip = o_clip[:, :, 6]
            quat_clip = batch_axisangle2quat(rot_clip[:, 0])
            quat_clip = np.stack([quat_clip, np.zeros_like(quat_clip)], axis=1)
            if not overlay_path.exists():
                draw_overlay_on_clip(clip_path, overlay_path, K=k_params, e_clip=e_clip,
                                    end_pos_clip=pos_clip, end_quat_clip=quat_clip, end_eff_clip=eff_clip,
                                    vmin=0.0, vmax=100)
            if not black_path.exists():
                draw_overlay_on_black(clip_path, black_path, K=k_params, e_clip=e_clip,
                                    end_pos_clip=pos_clip, end_quat_clip=quat_clip, end_eff_clip=eff_clip,
                                    vmin=0.0, vmax=100)
        except Exception as viz_err:
            print(f"[WARN] overlay failed for {clip_path}: {viz_err}")
        out_items.append({"caption": task_lang, "media_path": str(clip_path)})
        last_f0 = f0
        f0 += stride; i += 1

    if total_frames_est >= clip_len:
        final_f0 = total_frames_est - clip_len
        if final_f0 > last_f0:
            clip_path = clip_dir / f"{nice_stem}_clip_{i:05d}.mp4"
            if not clip_path.exists():
                cut_clip_by_frames(resampled_path, final_f0, clip_len, fps_out, clip_path, crf, preset)
            e_clip = e_res[final_f0:final_f0+clip_len]
            a_clip = action_res[final_f0:final_f0+clip_len]
            o_clip = obs_state_res[final_f0:final_f0+clip_len]
            npz_path = meta_dir / f"{nice_stem}_clip_{i:05d}.npz"
            np.savez_compressed(npz_path,
                extrinsics=e_clip, intrinsics=k_params,
                action=a_clip, obs_state=o_clip,
                start_frame=final_f0, clip_len=clip_len, fps=fps_out
            )
            out_items.append({"caption": task_lang, "media_path": str(clip_path)})

    return out_items

# --------------------------------------------------
def collect_camera_serials(meta_json: Dict[str,Any]) -> Dict[str,str]:
    # 自动收集所有 *_cam_serial
    out = {}
    for k,v in meta_json.items():
        if k.endswith("_cam_serial"):
            cam_name = k[:-len("_cam_serial")]
            out[cam_name] = str(v)
    return out


class StereoCamera:
    left_images: list[np.ndarray]
    right_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    width: float
    height: float
    left_dist_coeffs: np.ndarray
    left_intrinsic_mat: np.ndarray

    right_dist_coeffs: np.ndarray
    right_intrinsic_mat: np.ndarray

    def __init__(self, recordings: Path, serial: int, device_id: int, new_image_size: tuple[int, int] = (1280, 720)):
        
        # try:
        import pyzed.sl as sl
        init_params = sl.InitParameters()
        init_params.sdk_verbose = 0
        # print(f"device_id: {device_id}")
        init_params.sdk_gpu_id = device_id
        svo_path = recordings / "SVO" / f"{serial}.svo"
        init_params.set_from_svo_file(str(svo_path))
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.2

        zed = sl.Camera()
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error reading camera data: {err}")

        params = (
            zed.get_camera_information().camera_configuration.calibration_parameters
        )
        
        self.left_intrinsic_mat = np.array(
            [
                [params.left_cam.fx, 0, params.left_cam.cx],
                [0, params.left_cam.fy, params.left_cam.cy],
                [0, 0, 1],
            ]
        )
        self.right_intrinsic_mat = np.array(
            [
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ]
        )
        self.zed = zed

        if new_image_size != (1280, 720):
            scale_x = new_image_size[0] / 1280
            scale_y = new_image_size[1] / 720 
            self.left_intrinsic_mat = np.array(
                [
                    [params.left_cam.fx * scale_x, 0, params.left_cam.cx * scale_x],
                    [0, params.left_cam.fy * scale_y, params.left_cam.cy * scale_y],
                    [0, 0, 1],
                ]
            )
            self.right_intrinsic_mat = np.array(
                [
                    [params.right_cam.fx * scale_x, 0, params.right_cam.cx * scale_x],
                    [0, params.right_cam.fy * scale_y, params.right_cam.cy * scale_y],
                    [0, 0, 1],
                ]
            )

        # except ModuleNotFoundError:
        #     # pyzed isn't installed we can't find its intrinsic parameters
        #     # so we will have to make a guess.
        #     self.left_intrinsic_mat = np.array([
        #         [733.37261963,   0.,         625.26251221],
        #         [  0.,         733.37261963,  361.92279053],
        #         [  0.,           0.,           1.,        ]
        #     ])
        #     self.right_intrinsic_mat = self.left_intrinsic_mat
            
        #     mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
        #     if (recordings / "MP4" / f'{serial}-stereo.mp4').exists():
        #         mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
        #     elif (recordings / "MP4" / f'{serial}.mp4').exists():
        #         # Sometimes they don't have the '-stereo' suffix
        #         mp4_path = recordings / "MP4" / f'{serial}.mp4'
        #     else:
        #         raise Exception(f"unable to video file for camera {serial}")

        #     self.cap = cv2.VideoCapture(str(mp4_path))
        #     print(f"opening {mp4_path}")


    def get_next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Gets the the next from both cameras and maybe computes the depth."""

        if hasattr(self, "zed"):
            # We have the ZED SDK installed.
            import pyzed.sl as sl
            left_image = sl.Mat()
            # right_image = sl.Mat()
            depth_image = sl.Mat()

            rt_param = sl.RuntimeParameters()
            err = self.zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                left_image = np.array(left_image.numpy())

                # self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                # right_image = np.array(right_image.numpy())
                right_image = None

                self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                depth_image = np.array(depth_image.numpy())

                return (left_image, right_image, depth_image)
            else:
                return None
        else:
            # We don't have the ZED sdk installed
            ret, frame = self.cap.read()
            if ret:
                left_image = frame[:,:1280,:]
                right_image = frame[:,1280:,:]
                return (left_image, right_image, None)
            else:
                print("empty!")
                return None

def _combine_sec_nsec(sec, nsec):
    t = sec.astype(np.float64) + nsec.astype(np.float64) * 1e-9
    # 防止偶发逆序
    return np.maximum.accumulate(t)

def get_camera_frame_ts(H: h5py.File, serial: str) -> np.ndarray:
    """返回相机逐帧时间戳（float秒，shape (F,)），优先用 estimated_capture。"""
    grp = H["observation"]["timestamp"]["cameras"]
    # 按优先级选 key
    for suffix in ["estimated_capture", "frame_received", "read_end", "read_start"]:
        key = f"{serial}_{suffix}"
        if key not in grp:
            continue
        ds = grp[key]
        arr = ds[...]
        # 情况1：复合dtype（如含 'sec','nsec' 字段）
        if hasattr(arr, "dtype") and arr.dtype.fields:
            flds = arr.dtype.fields
            names = [n for n in flds.keys()]
            lower = [n.lower() for n in names]
            # 常见字段名匹配
            try:
                i_sec  = lower.index("sec")   if "sec"   in lower else lower.index("seconds")
                i_nsec = lower.index("nsec")  if "nsec"  in lower else lower.index("nanoseconds")
                sec  = arr[names[i_sec]]
                nsec = arr[names[i_nsec]]
                return _combine_sec_nsec(sec, nsec)
            except Exception:
                # 若字段名不标准，退化为把第0列当sec、第1列当nsec
                if arr.ndim == 1 and len(arr.dtype)==2:
                    sec  = arr[names[0]]
                    nsec = arr[names[1]]
                    return _combine_sec_nsec(sec, nsec)
                raise RuntimeError(f"Unsupported compound ts format in {key}: {arr.dtype}")

        # 情况2：二维 [sec, nsec]
        if arr.ndim == 2 and arr.shape[1] == 2:
            sec, nsec = arr[:,0], arr[:,1]
            return _combine_sec_nsec(sec, nsec)

        # 情况3：一维整型“秒”
        if np.issubdtype(arr.dtype, np.integer):
            # 只有整秒的话，就返回整秒（下游可配合 fps 做最近邻）
            t = arr.astype(np.float64)
            return np.maximum.accumulate(t)

        # 情况4：一维浮点“秒”
        if np.issubdtype(arr.dtype, np.floating):
            t = arr.astype(np.float64)
            return np.maximum.accumulate(t)

        raise RuntimeError(f"Unknown timestamp dtype/shape for {key}: {arr.dtype}, shape={arr.shape}")

    raise KeyError(f"No timestamp dataset found for serial={serial} "
                   f"among estimated_capture/frame_received/read_end/read_start")
    
def main():
    ap = argparse.ArgumentParser("DROID to clips+npz exporter")
    ap.add_argument("--scene_root", type=Path, default="/inspire/ssd/project/robotsimulation/public/data/droid_raw/1.0.1/RAD", help="目录：包含 success/ 和 failure/ 子目录")
    ap.add_argument("--output_dir", type=Path, default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/droid")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--clip_len", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--image_width", type=int, default=1280, help="假定左图宽；若不同可改")
    ap.add_argument("--image_height", type=int, default=720, help="假定左图高；若不同可改")
    args = ap.parse_args()

    scene_root: Path = args.scene_root
    scene_name = scene_root.name

    splits = []
    if (scene_root / "success").exists(): splits.append("success")
    if (scene_root / "failure").exists(): splits.append("failure")

    for split in splits:
        split_dir = scene_root / split
        date_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for date_dir in date_dirs:
            ts_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])
            tasks = []
            for ts_dir in ts_dirs:
                # 元数据
                json_list = list(ts_dir.glob("*.json"))
                if not json_list:
                    print(f"[WARN] no metadata json under {ts_dir}, skip")
                    continue
                with open(json_list[0], "r") as f:
                    meta = json.load(f)
                    
                agg_path = "/inspire/ssd/project/robotsimulation/public/data/droid_raw/1.0.1/aggregated-annotations-030724.json"
                with open(agg_path, "r") as f:
                    meta_agg = json.load(f)
                    
                traj_path = ts_dir / "trajectory.h5"
                if not traj_path.exists():
                    print(f"[WARN] {traj_path} not found, skip")
                    continue

                # 读取一次 episode 级别数组
                # with cv2.samples.findFileOrKeep:  # 只是避免静态分析告警
                #     pass

                with h5py.File(str(traj_path), "r") as H:
                    action, obs_state = build_action_and_obs(H)
                    cam_serials = collect_camera_serials(meta)
                    # cam_frame_ts = {}
                    # for cam_name in cam_serials:
                    #     frame_ts = get_camera_frame_ts(H, cam_serials[cam_name])
                    #     cam_frame_ts[cam_name] = frame_ts
                    
                # 为每个相机创建任务
                for cam_name, serial in cam_serials.items():
                    # if cam_name != 'ext1':
                    #     continue
                    try:
                        camera = StereoCamera(
                            ts_dir / "recordings",
                            serial,
                            device_id=0
                        )
                    except:
                        continue
                    K = camera.left_intrinsic_mat
                    try:
                        mp4 = ts_dir / "recordings" / "MP4"
                        video_path = find_episode_video(mp4, serial)
                    except Exception as e:
                        print(f"[WARN] {ts_dir} camera {cam_name} serial {serial}: {e}")
                        continue
                        
                    # 每个相机的 per-frame extrinsics
                    try:
                        with h5py.File(str(traj_path), "r") as H:
                            e_params = build_extrinsics_per_frame(H, serial)  # (N,4,4)
                    except:
                        continue

                    # nice stem：<date>_<timestamp>_<split>_<cam>
                    nice_stem = f"{date_dir.name}_{ts_dir.name}_{split}_{cam_name}"

                    out_root = args.output_dir / scene_name / split / date_dir.name / ts_dir.name
                    ensure_dir(out_root)
                    
                    tasks.append(dict(
                        mp4_path=video_path,
                        out_root=out_root,
                        nice_stem=nice_stem,
                        task_lang=meta_agg.get(meta['uuid'], ""),
                        e_params_orig=e_params,
                        k_params=K,
                        action=action,
                        obs_state=obs_state,
                        fps_out=args.fps,
                        clip_len=args.clip_len,
                        stride=args.stride,
                        crf=args.crf,
                        preset=args.preset
                    ))
                    # break
                
            if not tasks:
                continue
            
            # process_one_camera_episode(**tasks[1])
            # 并行跑
            items_all: List[Dict[str,Any]] = []
            print(f"[{split}/{date_dir.name}] start {len(tasks)} camera-episodes with {args.jobs} workers...")
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
                fut2t = {ex.submit(process_one_camera_episode, **t): t for t in tasks}
                for idx, fut in enumerate(as_completed(fut2t), 1):
                    t = fut2t[fut]
                    try:
                        items = fut.result()
                        items_all.extend(items)
                        print(f"  [{idx}/{len(tasks)}] OK: {t['nice_stem']} -> {len(items)} clips")
                    except Exception as e:
                        print(f"  [{idx}/{len(tasks)}] FAIL: {t['nice_stem']}: {e}")

            # 写 dataset.json（按 split/date 聚合）
            if items_all:
                meta_path = args.output_dir / scene_name / split / date_dir.name / "dataset.json"
                ensure_dir(meta_path.parent)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(items_all, f, ensure_ascii=False, indent=2)
                print(f"[{split}/{date_dir.name}] saved {meta_path}")

if __name__ == "__main__":
    main()
