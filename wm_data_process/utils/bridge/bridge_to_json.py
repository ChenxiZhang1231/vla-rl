#!/usr/bin/env python3
"""
Dataset video processor with synchronized camera/robot metadata and 2D projection overlays.

What it does (on top of your original script):
- Resamples raw videos to a target FPS, cuts them into fixed-length clips with stride.
- Resamples per-frame metadata (camera extrinsics, end-effector poses) to match the resampled frame
  timeline exactly (via time-based index mapping from original FPS -> target FPS).
- For every produced clip, saves strictly corresponding:
    * camera extrinsics (F, 4, 4)
    * camera intrinsics (3, 3)
    * end-effector positions (F, 2, 3)
    * end-effector orientations as quaternions (F, 2, 4)
  into NPZ sidecars inside output/metadata/.
- Creates a visualization overlay video per clip that projects each end effector and its XYZ axes onto
  the image plane using the (resampled, per-frame) extrinsics + shared intrinsics.

Assumptions / notes:
- Extrinsics in JSON are world->camera (R, t) so that X_cam = R * X_world + t. If your JSON stores
  camera->world, set EXTRINSIC_IS_WORLD_TO_CAMERA = False.
- End-effector orientation quaternion format is (w, x, y, z). If your data is (x, y, z, w), set
  QUAT_IS_W_FIRST = False.
- Units for positions are consistent across end_effector/world/camera (e.g., meters). The axis length
  used for drawing is AXIS_LEN_UNITS.
"""
import argparse
import subprocess
from pathlib import Path
import json
import os
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import h5py
import numpy as np
import torch
import uuid

import packaging.version
from lerobot.datasets.video_utils import get_video_info
from lerobot.datasets.utils import (
    get_episode_data_index,
    load_episodes,
    load_info,
    load_tasks,
    hf_transform_to_torch,
)
import datasets
from tqdm import tqdm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class LRMeta:
    """轻量封装，使用你提供的 load_* 接口 + HF parquet 访问。"""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)

        self.total_episodes = self.info["total_episodes"]
        self.fps = self.info["fps"]
        self.features = self.info["features"]
        self.video_path_fmt = self.info["video_path"]   # 可 format

        path = str(self.root / "data")
        self.hf = datasets.load_dataset("parquet", data_dir=path, split="train")
        # self.hf.set_transform(hf_transform_to_torch)
        # breakpoint()
        self.ep_index = get_episode_data_index(self.episodes, episodes=None)

        # names（便于自动解析 state 中的字段语义）
        self.names = {k: ft.get("names", None) for k, ft in self.features.items()}

    def video_keys(self) -> List[str]:
        return [k for k, ft in self.features.items() if ft["dtype"] == "video"]

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = ep_index // self.info["chunks_size"]
        fpath = self.video_path_fmt.format(
            episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index
        )
        return self.root / fpath

    def episode_frame_range(self, ep_idx: int) -> Tuple[int, int]:
        return self.ep_index["from"][ep_idx].item(), self.ep_index["to"][ep_idx].item()


EXTRINSIC_IS_WORLD_TO_CAMERA = False  # set False if your extrinsics are camera->world
QUAT_IS_W_FIRST = False               # set False if quaternions are (x,y,z,w)
AXIS_LEN_UNITS = 0.1                # length of drawn XYZ axes in world units (tweak as needed)
AXIS_THICKNESS = 2                   # px for axes
POINT_RADIUS = 4                     # px for effector point


def ffprobe_codec_name(path: Path) -> str:
    """Return codec_name of the first video stream (e.g., 'h264', 'av1')."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout.strip()
def h5_to_dict(h5obj):
    d = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Dataset):
            d[key] = item[()]
        elif isinstance(item, h5py.Group):
            d[key] = h5_to_dict(item)
    return d

def preprocess_e_params(
    raw_extrinsics,
    *,
    orthonormalize: bool = False,
    enforce_right_handed: bool = False,
    translation_scale: float = 1.0,
    dtype=np.float64,
):
    if isinstance(raw_extrinsics, dict):
        seq = [raw_extrinsics]
    else:
        seq = list(raw_extrinsics)

    Ts = []
    for i, item in enumerate(seq):
        extr = item.get("extrinsic", item)
        if "rotation_matrix" not in extr or "translation_vector" not in extr:
            raise KeyError(
                f"[index {i}] missing 'rotation_matrix' or 'translation_vector' in {list(extr.keys())}"
            )
        R = np.asarray(extr["rotation_matrix"], dtype=dtype)
        t = np.asarray(extr["translation_vector"], dtype=dtype).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"[index {i}] rotation_matrix must be 3x3, got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"[index {i}] translation_vector must have length 3, got {t.shape}")
        if orthonormalize:
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if enforce_right_handed and np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
        if translation_scale != 1.0:
            t = t * translation_scale

        T = np.eye(4, dtype=dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        Ts.append(T)

    return np.stack(Ts, axis=0)

def preprocess_k_params(raw_intrinsic, dtype=np.float64):
    intr = raw_intrinsic.get("intrinsic", raw_intrinsic)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=dtype)
    return K

def run_cmd(cmd: List[str]):
    cmd = ["ffmpeg" if cmd[0] == "ffmpeg" else cmd[0]] + cmd[1:]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return res.stdout

def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return 0.0
    out = res.stdout.strip()
    try:
        return float(out)
    except:
        return 0.0

def ffprobe_fps(path: Path) -> float:
    """Return average frame rate as float fps."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return 0.0
    fr = res.stdout.strip()
    try:
        if "/" in fr:
            num, den = fr.split("/")
            num = float(num)
            den = float(den)
            return 0.0 if den == 0 else num / den
        return float(fr)
    except:
        return 0.0
    
def resample_to_fps(input_path: Path, output_path: Path, fps: int, crf: int = 20, preset: str = "veryfast"):
    """
    Resample to FPS and re-encode to H.264. If the input is AV1 and the local FFmpeg
    lacks an AV1 decoder (common on older builds), we first try to force libdav1d as
    the decoder. If that still fails, we raise a clear error with remediation steps.
    """
    ensure_dir(output_path.parent)

    def try_cmd(cmd_list: List[str]):
        try:
            run_cmd(cmd_list)
            return True
        except RuntimeError as e:
            err = str(e)
            if "Decoder (codec av1) not found" in err or "Unknown decoder 'libdav1d'" in err:
                return False
            # re-raise other errors
            raise

    codec = ffprobe_codec_name(input_path)

    # 1) normal path
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path)
    ]
    if try_cmd(cmd):
        return

    # 2) AV1-specific retry with libdav1d decoder (if available) BEFORE -i
    if codec == "av1":
        cmd_dav1d = [
            "ffmpeg", "-y",
            "-c:v", "libdav1d",  # force AV1 decoder if present
            "-i", str(input_path),
            "-r", str(fps),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ]
        if try_cmd(cmd_dav1d):
            return
        # 3) fallback try libaom-av1 decoder
        cmd_libaom = [
            "ffmpeg", "-y",
            "-c:v", "libaom-av1",
            "-i", str(input_path),
            "-r", str(fps),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ]
        if try_cmd(cmd_libaom):
            return

    # If we got here, the local FFmpeg likely has no AV1 decoder compiled in.
    raise RuntimeError(
        "FFmpeg cannot decode AV1 on this machine."
    )

def cut_clip_by_frames(input_path: Path, start_frame: int, num_frames: int, fps: int,
                       output_path: Path, crf: int = 20, preset: str = "veryfast"):
    start_time = start_frame / fps
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.6f}",
        "-i", str(input_path),
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path)
    ]
    run_cmd(cmd)

def make_video_stem(video_path: Path) -> str:
    parts = video_path.parts
    try:
        idx = parts.index("observations")
        observation = parts[idx + 1]
        session = parts[idx + 2]
    except Exception:
        observation = video_path.parent.parent.name
        session = video_path.parent.name
    camera_stem = video_path.stem
    return f"{observation}_{session}_{camera_stem}"

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

# ---------------- Resampling helpers ----------------
def indices_for_resampled(num_resampled: int, orig_fps: float, new_fps: float, n_orig: int) -> np.ndarray:
    """Map each resampled frame index to nearest original frame index based on time alignment."""
    t_res = np.arange(num_resampled, dtype=np.float64) / max(new_fps, 1e-6)
    idx = np.round(t_res * orig_fps).astype(int)
    return np.clip(idx, 0, n_orig - 1)

# ---------------- Core processing ----------------
def draw_overlay_on_clip(clip_path: Path, out_path: Path,
                         K: np.ndarray,
                         e_clip: np.ndarray,          # (F, 4, 4)
                         end_pos_clip: np.ndarray,    # (F, 2, 3)
                         end_quat_clip: np.ndarray,    # (F, 2, 4)
                         ) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {clip_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    F = e_clip.shape[0]
    axis = np.eye(3) * AXIS_LEN_UNITS  # 3 axis unit vectors scaled

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f >= F:
            # safety: if video has more frames than metadata window
            break
        # For each of two end-effectors
        for ee in range(2):
            pos_w = end_pos_clip[f, ee, :]  # (3,)
            quat = end_quat_clip[f, ee, :]  # (4,)
            R_ee = quat_to_rot(quat)        # (3,3)
            # endpoints in world for axes
            pts_axes_w = pos_w[None, :] + (R_ee @ axis).T  # (3, 3)
            # also the effector point itself
            pts_w = np.vstack([pos_w[None, :], pts_axes_w])  # (4, 3)

            Xc = world_to_cam_points(e_clip[f], pts_w)  # (4,3)
            uv = cam_to_pixels(K, Xc[None, ...])[0]     # (4,2)

            # draw: point (white), axes (R,G,B)
            # Note: OpenCV expects BGR
            u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
            cv2.circle(frame, (u0, v0), POINT_RADIUS, (255, 255, 255), -1)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=Red, Y=Green, Z=Blue in RGB
            for k in range(3):
                uk, vk = int(round(uv[k+1, 0])), int(round(uv[k+1, 1]))
                cv2.line(frame, (u0, v0), (uk, vk), colors[k], AXIS_THICKNESS, cv2.LINE_AA)

        writer.write(frame)
        f += 1

    writer.release()
    cap.release()
    
def draw_overlay_on_black(clip_path: Path, out_path: Path,
                         K: np.ndarray,
                         e_clip: np.ndarray,          # (F, 4, 4)
                         end_pos_clip: np.ndarray,    # (F, 2, 3)
                         end_quat_clip: np.ndarray,    # (F, 2, 4)
                         ) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {clip_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    F = e_clip.shape[0]
    axis = np.eye(3) * AXIS_LEN_UNITS  # 3 axis unit vectors scaled

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f >= F:
            # safety: if video has more frames than metadata window
            break
        # For each of two end-effectors
        for ee in range(2):
            frame = np.zeros_like(frame)
            pos_w = end_pos_clip[f, ee, :]  # (3,)
            quat = end_quat_clip[f, ee, :]  # (4,)
            R_ee = quat_to_rot(quat)        # (3,3)
            # endpoints in world for axes
            pts_axes_w = pos_w[None, :] + (R_ee @ axis).T  # (3, 3)
            # also the effector point itself
            pts_w = np.vstack([pos_w[None, :], pts_axes_w])  # (4, 3)

            Xc = world_to_cam_points(e_clip[f], pts_w)  # (4,3)
            uv = cam_to_pixels(K, Xc[None, ...])[0]     # (4,2)

            # draw: point (white), axes (R,G,B)
            # Note: OpenCV expects BGR
            u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
            cv2.circle(frame, (u0, v0), POINT_RADIUS, (255, 255, 255), -1)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=Red, Y=Green, Z=Blue in RGB
            for k in range(3):
                uk, vk = int(round(uv[k+1, 0])), int(round(uv[k+1, 1]))
                cv2.line(frame, (u0, v0), (uk, vk), colors[k], AXIS_THICKNESS, cv2.LINE_AA)

        writer.write(frame)
        f += 1

    writer.release()
    cap.release()
    

def process_video(video_path: Path, out_root: Path, fps: int, clip_len: int, stride: int,
                  crf: int = 20, preset: str = "veryfast",
                  action: np.ndarray = None,
                  obs_state: np.ndarray = None,
                  ) -> Tuple[Path, List[Tuple[Path, int]]]:
    stem = make_video_stem(Path(video_path))
    resampled_path = out_root / "resampled" / f"{stem}_resampled.mp4"
    ensure_dir(resampled_path.parent)

    # get original fps + frame count
    orig_fps = ffprobe_fps(video_path)
    # resample video
    if clip_len == -1:
        resampled_path = video_path
    else:
        if not resampled_path.exists():
            resample_to_fps(video_path, resampled_path, fps=fps, crf=crf, preset=preset)

    # after resampling, estimate total frames
    duration = ffprobe_duration(resampled_path)
    if duration <= 0:
        print(f"[WARN] ffprobe failed or zero duration: {resampled_path}")
    total_frames_est = max(0, int(round(duration * fps)))

    # map metadata to resampled timeline
    N = action.shape[0]
    idx_map = indices_for_resampled(total_frames_est, orig_fps=orig_fps if orig_fps>0 else fps, new_fps=fps, n_orig=N)
    action_res = action[idx_map]
    obs_state_res = obs_state[idx_map]

    # cut into clips + save metadata + overlays
    clip_dir = out_root / "clips"
    meta_dir = out_root / "metadata"
    overlay_dir = out_root / "overlays"
    black_dir = out_root / "blacks"
    ensure_dir(clip_dir); ensure_dir(meta_dir); ensure_dir(overlay_dir)

    products: List[Tuple[Path, int]] = []  # (clip_path, start_frame)
    f0 = 0
    i = 0
    last_processed_f0 = -1
    if clip_len == -1:
        clip_len = total_frames_est
    while f0 + clip_len <= total_frames_est:
        clip_path = clip_dir / f"{stem}_clip_{i:05d}.mp4"
        if not clip_path.exists():
            cut_clip_by_frames(resampled_path, start_frame=f0, num_frames=clip_len, fps=fps,
                           output_path=clip_path, crf=crf, preset=preset)
        # slice metadata
        action_clip = action_res[f0:f0+clip_len]
        obs_state_clip = obs_state_res[f0:f0+clip_len]
        # save sidecar npz
        npz_path = meta_dir / f"{stem}_clip_{i:05d}.npz"
        np.savez_compressed(npz_path,
                            action=action_clip,
                            obs_state=obs_state_clip,
                            start_frame=f0,
                            clip_len=clip_len,
                            fps=fps)
        # # create overlay visualization
        # overlay_path = overlay_dir / f"{stem}_clip_{i:05d}_overlay.mp4"
        # black_path = black_dir / f"{stem}_clip_{i:05d}_black.mp4"
        # try:
        #     if not overlay_path.exists():
        #         draw_overlay_on_clip(clip_path, overlay_path, K=k_params, e_clip=e_clip,
        #                             end_pos_clip=pos_clip, end_quat_clip=quat_clip)
        #     if not black_path.exists():
        #         draw_overlay_on_black(clip_path, black_path, K=k_params, e_clip=e_clip,
        #                             end_pos_clip=pos_clip, end_quat_clip=quat_clip)
        # except Exception as viz_err:
        #     print(f"[WARN] overlay failed for {clip_path}: {viz_err}")
        products.append((clip_path, f0))
        last_processed_f0 = f0
        f0 += stride
        i += 1
        
    if total_frames_est > clip_len:
        final_f0 = total_frames_est - clip_len
        # Only add this clip if it wasn't the same as the last one processed in the loop
        # if final_f0 > last_processed_f0:
        if True:
            clip_path = clip_dir / f"{stem}_clip_{i:05d}.mp4"
            if not clip_path.exists():
                cut_clip_by_frames(resampled_path, start_frame=final_f0, num_frames=clip_len, fps=fps, output_path=clip_path, crf=crf, preset=preset)

            e_clip = e_res[final_f0:final_f0+clip_len]
            action_clip = action_res[f0:f0+clip_len]
            obs_state_clip = obs_state_res[f0:f0+clip_len]
            npz_path = meta_dir / f"{stem}_clip_{i:05d}.npz"
            np.savez_compressed(npz_path,
                                extrinsics=e_clip,
                                intrinsics=k_params,
                                action=action_clip,
                                obs_state=obs_state_clip,
                                start_frame=f0,
                                clip_len=clip_len,
                                fps=fps)
            # overlay_path = overlay_dir / f"{stem}_clip_{i:05d}_overlay.mp4"
            # try:
            #     if not overlay_path.exists():
            #         draw_overlay_on_clip(clip_path, overlay_path, K=k_params, e_clip=e_clip, end_pos_clip=pos_clip, end_quat_clip=quat_clip)
            # except Exception as viz_err:
            #     print(f"[WARN] overlay failed for {clip_path}: {viz_err}")

            products.append((clip_path, final_f0))

    return resampled_path, products


def process_video_task(task: Dict[str, Any]) -> Dict[str, Any]:
    vp: Path = task["video_path"]
    try:
        _, clip_infos = process_video(
            video_path=vp,
            out_root=task["out_root"],
            fps=task["fps"],
            clip_len=task["clip_len"],
            stride=task["stride"],
            crf=task["crf"],
            preset=task["preset"],
            action=task["action"],
            obs_state=task["obs_state"]
        )
        items = [{"caption": task["description"], "media_path": str(p)} for (p, _) in clip_infos]
        return {"ok": True, "items": items, "video": str(vp)}
    except Exception as e:
        return {"ok": False, "err": f"{vp}: {e}", "video": str(vp)}


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """确保帧为 (T,H,W,3) uint8 RGB。"""
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"frames shape should be (T,H,W,3), got {frames.shape}"
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames

def _write_temp_video_rgb(frames_rgb: np.ndarray, out_path: Path, fps: int = 10) -> None:
    """
    将 RGB 帧写成 mp4v 视频。若未提供 fps，则使用默认 10。
    frames_rgb: (T, H, W, 3) 的 uint8 或可转为 uint8 的数组（RGB 顺序）
    """
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"frames_rgb must be (T,H,W,3), got {frames_rgb.shape}")

    # 确保目录存在
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保证类型与内存布局
    if frames_rgb.dtype != np.uint8:
        frames_rgb = frames_rgb.astype(np.uint8)
    frames_rgb = np.ascontiguousarray(frames_rgb)

    T, H, W, _ = frames_rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    # OpenCV 需要 BGR
    for t in range(T):
        vw.write(cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR))

    vw.release()
    

def extract_rlds_data_and_create_video(args, rlds_batch: Dict[str, Any], temp_video_dir: str) -> List[Dict[str, Any]]:
    """
    从单个 RLDS batch 中：
      1) 抽取每路相机的帧串，写出临时 mp4（位于 temp_video_dir）
      2) 组装与后续处理契约一致的 task 字典列表（每路相机一个 task）
         返回的 task 字段：
           - video_path: 临时 mp4 路径
           - e_params:   (F, 4, 4) 外参（占位/或真实）
           - k_params:   (3, 3)    内参（占位/或真实）
           - action:     (F, 2, A) 动作（第二通道零填）
           - obs_state:  (F, 2, S) 观测状态（如无则零填；S 默认 8，可用 args.state_dim 覆盖）
    说明：
      - description/out_root/fps/clip_len/stride/crf/preset 由上层 ta.update(...) 统一补齐（保持与你 LIBERO 代码一致）。
    """
    # 识别本 episode 的基本信息
    dataset_name = rlds_batch["dataset_name"][0].decode()
    language = rlds_batch["task"]["language_instruction"][0].decode().lower()  # 由上层 update() 填入 description

    obs_dict = rlds_batch["observation"]
    # 选择要导出的相机键：优先用 args.cams，否则抓取 observation 中以 "image" 开头的字段
    # if hasattr(args, "cams") and args.cams:
    #     cam_keys = [k for k in args.cams if k in obs_dict]
    # else:
    #     cam_keys = [k for k in obs_dict.keys() if isinstance(k, str) and k.startswith("image")]
    cam_keys = ['image_primary']

    if len(cam_keys) == 0:
        raise ValueError("No camera/image keys found in RLDS batch. "
                         "Provide args.cams or ensure observation has image_* fields.")

    # 抽取动作与状态
    action = rlds_batch["action"][:, 0] # 期望 (F, A)
    if action.ndim != 2:
        raise ValueError(f"Expect action shape (F, A), got {action.shape}")
    action = action.astype(np.float32)
    F = action.shape[0]
    A = action.shape[1]

    obs_state = obs_dict["proprio"][:,0].astype(np.float32)  # (F, S)
    if obs_state.ndim != 2:
        raise ValueError(f"Expect obs_state shape (F, S), got {obs_state.shape}")
    S = obs_state.shape[1]

    # 扩成 (F, 2, ·)：第二通道零填（与你原管道一致）
    action_2 = np.concatenate([action[:, None, :], np.zeros_like(action[:, None, :])], axis=1)     # (F,2,A)
    state_2  = np.concatenate([obs_state[:, None, :], np.zeros_like(obs_state[:, None, :])], axis=1)  # (F,2,S)

    # 临时视频输出目录
    temp_dir = Path(temp_video_dir)
    ensure_dir(temp_dir)

    # episode 唯一 id（若没有显式 episode_id，这里生成一个短 uuid）
    ep_uid = None
    if "episode_id" in rlds_batch:
        # 有的 RLDS 会带 bytes；也可能是标量数组
        raw = rlds_batch["episode_id"]
        try:
            if isinstance(raw, (bytes, bytearray)):
                ep_uid = raw.decode()
            elif hasattr(raw, "item"):
                ep_uid = str(raw.item())
            else:
                ep_uid = str(raw)
        except Exception:
            ep_uid = None
    if not ep_uid:
        ep_uid = uuid.uuid4().hex[:8]

    tasks: List[Dict[str, Any]] = []

    # 针对每路相机，写临时视频 + 组装 task
    for cam_key in cam_keys:
        frames = obs_dict[cam_key][:,0]    # 期望 (F, H, W, 3)
        frames = _as_uint8_rgb(frames)
        if frames.shape[0] != F:
            raise ValueError(f"Frame length mismatch between action ({F}) and {cam_key} ({frames.shape[0]}).")

        # 生成临时 mp4（直接用 target fps 写出，后续 resample_to_fps 会是恒等）
        stem = f"{dataset_name}_ep{ep_uid}_{cam_key}"
        temp_mp4 = temp_dir / f"{stem}.mp4"
        if not temp_mp4.exists():
            _write_temp_video_rgb(frames, temp_mp4)

        ta = {
            "video_path": str(temp_mp4),
            "action":   action_2,   # (F,2,A)
            "obs_state": state_2,   # (F,2,S)
        }
        tasks.append(ta)

    return tasks

# 覆盖写
# python utils/bridge/bridge_to_json.py > run-bridge.log 2>&1
# 追加写
# python utils/openx/openx_to_json.py >> run-openx.log 2>&1
# python utils/openx/openx_to_json.py 2>&1 | tee -a run-openx.log
def main():
    parser = argparse.ArgumentParser(description="Process dataset videos with synchronized metadata + 2D projections.")
    parser.add_argument("--dataset_dirs", type=Path, default='/inspire/ssd/project/robotsimulation/public/data/bridge', help="the dataset root")
    parser.add_argument("--output_dir", type=Path, default='/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/bridge_orig', help="processed output path")
    parser.add_argument("--data_mix", type=str, default='bridge_orig')
    parser.add_argument("--fps", type=int, default=10, help="output fps for initial sampling")
    parser.add_argument("--clip_len", type=int, default=-1, help="frames per clip")
    parser.add_argument("--stride", type=int, default=60, help="frame stride between clips")
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF (lower = better quality, larger size)")
    parser.add_argument("--preset", type=str, default="veryfast", help="x264 preset")
    parser.add_argument("--debug", type=bool, default=False, help="x264 preset")
    parser.add_argument("--jobs", type=int, default=128, help="parallel workers (number of concurrent videos)")
    args = parser.parse_args()

    import sys
    sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/VLA-Adapter")
    from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform, EpisodicRLDSDataset

    dataset = EpisodicRLDSDataset(
        str(args.dataset_dirs),
        args.data_mix,
        batch_transform=None,
        resize_resolution=(256,256),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=False,
    )
    out_root = Path(args.output_dir) / args.data_mix
    ensure_dir(out_root)
    temp_video_dir = Path(args.output_dir) / "temp_videos"
    ensure_dir(temp_video_dir)
    dataset_length = dataset.dataset_length

    meta_items: List[Dict[str, str]] = []
    ep_idx = 0
    tasks = []
    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing episodes"):
        if ep_idx >= dataset_length:
            break
        ep_idx += 1
        
        
        try:
            # 这里的函数名/签名请用你项目里真实实现的那个
            extracted_tasks = extract_rlds_data_and_create_video(args, rlds_batch, temp_video_dir)
        except Exception as e:
            print(f"[WARN] RLDS extract failed at episode {ep_idx}: {e}")
            continue

        # 与你给的 LIBERO 代码完全一致的 update 方式与键集合
        for ta in extracted_tasks:
            ta.update({
                "description": rlds_batch["task"]["language_instruction"][0].decode().lower(),
                "out_root": args.output_dir,
                "fps": args.fps,
                "clip_len": args.clip_len,
                "stride": args.stride,
                "crf": args.crf,
                "preset": args.preset,
            })
        tasks.extend(extracted_tasks)

        if args.debug and len(tasks) >= 50:
            break
    # process_video_task(tasks[0])  # debug
    
    data: List[Dict[str, str]] = []
    if len(tasks) == 0:
        print("No valid videos found.")
    else:
        print(f"Start processing {len(tasks)} videos with {max(1, args.jobs)} workers...")
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            future2task = {ex.submit(process_video_task, t): t for t in tasks}
            done_cnt = 0
            for fut in as_completed(future2task):
                result = fut.result()
                done_cnt += 1
                if result["ok"]:
                    data.extend(result["items"])
                    print(f"[{done_cnt}/{len(tasks)}] OK: {result['video']} -> {len(result['items'])} clips")
                else:
                    print(f"[{done_cnt}/{len(tasks)}] FAIL: {result['err']}")

        meta_path = out_root / 'dataset.json'
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Done. Meta saved to {meta_path}")
        # raise


if __name__ == "__main__":
    main()
