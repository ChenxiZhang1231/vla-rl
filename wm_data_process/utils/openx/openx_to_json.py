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
                  e_params_orig: np.ndarray = None,    # (N, 4, 4)
                  k_params: np.ndarray = None,         # (3, 3)
                  action: np.ndarray = None,
                  obs_state: np.ndarray = None,
                  ) -> Tuple[Path, List[Tuple[Path, int]]]:
    stem = make_video_stem(video_path)
    resampled_path = out_root / "resampled" / f"{stem}_resampled.mp4"
    ensure_dir(resampled_path.parent)

    # get original fps + frame count
    orig_fps = ffprobe_fps(video_path)
    # resample video
    if not resampled_path.exists():
        resample_to_fps(video_path, resampled_path, fps=fps, crf=crf, preset=preset)

    # after resampling, estimate total frames
    duration = ffprobe_duration(resampled_path)
    if duration <= 0:
        print(f"[WARN] ffprobe failed or zero duration: {resampled_path}")
    total_frames_est = max(0, int(round(duration * fps)))

    # map metadata to resampled timeline
    N = e_params_orig.shape[0]
    idx_map = indices_for_resampled(total_frames_est, orig_fps=orig_fps if orig_fps>0 else fps, new_fps=fps, n_orig=N)
    e_res = e_params_orig[idx_map]                    # (F,4,4)
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
    while f0 + clip_len <= total_frames_est:
        clip_path = clip_dir / f"{stem}_clip_{i:05d}.mp4"
        if not clip_path.exists():
            cut_clip_by_frames(resampled_path, start_frame=f0, num_frames=clip_len, fps=fps,
                           output_path=clip_path, crf=crf, preset=preset)
        # slice metadata
        e_clip = e_res[f0:f0+clip_len]
        action_clip = action_res[f0:f0+clip_len]
        obs_state_clip = obs_state_res[f0:f0+clip_len]
        # save sidecar npz
        npz_path = meta_dir / f"{stem}_clip_{i:05d}.npz"
        np.savez_compressed(npz_path,
                            extrinsics=e_clip,
                            intrinsics=k_params,
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
        
    if total_frames_est >= clip_len:
        final_f0 = total_frames_est - clip_len
        # Only add this clip if it wasn't the same as the last one processed in the loop
        if final_f0 > last_processed_f0:
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
            e_params_orig=task["e_params"],
            k_params=task["k_params"],
            action=task["action"],
            obs_state=task["obs_state"]
        )
        items = [{"caption": task["description"], "media_path": str(p)} for (p, _) in clip_infos]
        return {"ok": True, "items": items, "video": str(vp)}
    except Exception as e:
        return {"ok": False, "err": f"{vp}: {e}", "video": str(vp)}

 
# 覆盖写
# python utils/openx/openx_to_json.py > run-openx.log 2>&1
# 追加写
# python utils/openx/openx_to_json.py >> run-openx.log 2>&1
# python utils/openx/openx_to_json.py 2>&1 | tee -a run-openx.log


def main():
    parser = argparse.ArgumentParser(description="Process dataset videos with synchronized metadata + 2D projections.")
    parser.add_argument("--dataset_dirs", type=Path, default='/inspire/hdd/global_public/public_datas/openx-embodiment-lerobot/OXE_lerobot', help="the dataset root")
    parser.add_argument("--output_dir", type=Path, default='/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/openx', help="processed output path")
    parser.add_argument("--fps", type=int, default=10, help="output fps for initial sampling")
    parser.add_argument("--clip_len", type=int, default=121, help="frames per clip")
    parser.add_argument("--stride", type=int, default=60, help="frame stride between clips")
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF (lower = better quality, larger size)")
    parser.add_argument("--preset", type=str, default="veryfast", help="x264 preset")
    parser.add_argument("--jobs", type=int, default=128, help="parallel workers (number of concurrent videos)")
    args = parser.parse_args()

    dataset_item_list = os.listdir(args.dataset_dirs)
    for dataset_item in dataset_item_list:
        if dataset_item in ['.git']:
            continue
        print(f"Start to process {dataset_item}")
        out_root: Path = args.output_dir / dataset_item
        ensure_dir(out_root)
        dataset_dir = args.dataset_dirs / dataset_item
        
        meta = LRMeta(dataset_dir)
        # breakpoint()
        ep_list = list(range(meta.total_episodes))
        video_keys = meta.video_keys()
        print(f"{dataset_item} has {len(ep_list)} episodes.")
        tasks = []
        for ep_idx in tqdm(ep_list, desc="Export episodes"):
            start, end = meta.episode_frame_range(ep_idx)
            N = end - start
            if N <= 0:
                continue
            
            first = meta.hf[start]
            if isinstance(first["task_index"], int):
                task_index = first["task_index"]
            else:
                task_index = int(first["task_index"].item())
            task_name = meta.tasks[task_index]
            
            action_list = []
            obs_state_list = []
            for step in range(start, end):
                step_item = meta.hf[step]
                action = step_item['action']
                obs_state = step_item['observation.state']
                if isinstance(action, list):
                    action = torch.tensor(action)
                if isinstance(obs_state, list):
                    obs_state = torch.tensor(obs_state)
                action_list.append(action)
                obs_state_list.append(obs_state)
            action_single = torch.stack(action_list).unsqueeze(1)
            action_pad = torch.zeros_like(action_single)
            action = torch.cat([action_single, action_pad], dim=1).numpy()  # B, 2, 7
            obs_state_single = torch.stack(obs_state_list).unsqueeze(1)
            obs_state_pad = torch.zeros_like(obs_state_single)
            obs_state = torch.cat([obs_state_single, obs_state_pad], dim=1).numpy()  # B, 2, 8
            
            e_params = np.zeros([N, 4, 4])
            k_params = np.zeros([3, 3])
            
            
            
            for vid_key in video_keys:
                video_path = meta.get_video_file_path(ep_idx, vid_key)
                tasks.append(dict(
                    video_path=video_path,
                    out_root=out_root,
                    fps=args.fps,
                    clip_len=args.clip_len,
                    stride=args.stride,
                    crf=args.crf,
                    preset=args.preset,
                    description=task_name,
                    e_params=e_params,
                    k_params=k_params,
                    action=action,
                    obs_state=obs_state,
                ))
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
