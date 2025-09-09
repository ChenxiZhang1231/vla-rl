#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# =================================================================================
# PART 1: ALL THE CORE PROCESSING AND UTILITY FUNCTIONS FROM YOUR FIRST SCRIPT
# (No modifications needed here, just copy-pasting them in)
# =================================================================================

# ---------------- Configuration switches ----------------
# IMPORTANT: Adjust these based on LIBERO dataset specifics
EXTRINSIC_IS_WORLD_TO_CAMERA = False  # LIBERO extrinsics are camera->world, so set to False
QUAT_IS_W_FIRST = True               # LIBERO quaternions are (w,x,y,z), so set to True
AXIS_LEN_UNITS = 0.05                # length of drawn XYZ axes in world units (tweak as needed)
AXIS_THICKNESS = 2                   # px for axes
POINT_RADIUS = 5                     # px for effector point

# ---------------- Utilities ----------------

def run_cmd(cmd: List[str]):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return res.stdout

def ffprobe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0: return 0.0
    try: return float(res.stdout.strip())
    except: return 0.0

def ffprobe_fps(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0: return 0.0
    fr = res.stdout.strip()
    try:
        if "/" in fr:
            num, den = fr.split("/"); return float(num) / float(den) if float(den) != 0 else 0.0
        return float(fr)
    except: return 0.0

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resample_to_fps(input_path: Path, output_path: Path, fps: int, crf: int = 20, preset: str = "veryfast"):
    ensure_dir(output_path.parent)
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-r", str(fps), "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run_cmd(cmd)

def cut_clip_by_frames(input_path: Path, start_frame: int, num_frames: int, fps: int, output_path: Path, crf: int = 20, preset: str = "veryfast"):
    start_time = start_frame / fps
    ensure_dir(output_path.parent)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_time:.6f}", "-i", str(input_path), "-frames:v", str(num_frames), "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run_cmd(cmd)

def make_video_stem(video_path: Path) -> str:
    # Custom stem generation for LIBERO data
    return video_path.stem

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    if QUAT_IS_W_FIRST:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm = np.sqrt(w*w + x*x + y*y + z*z) + 1e-8
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2*(y*y + z*z); R[..., 0, 1] = 2*(x*y - z*w); R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w); R[..., 1, 1] = 1 - 2*(x*x + z*z); R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w); R[..., 2, 1] = 2*(y*z + x*w); R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R

def world_to_cam_points(T_cw: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    T = T_cw
    if not EXTRINSIC_IS_WORLD_TO_CAMERA:
        T = np.linalg.inv(T_cw)
    R, t = T[..., :3, :3], T[..., :3, 3]
    Xw_h = np.concatenate([Xw, np.ones_like(Xw[..., :1])], axis=-1)
    Xc_h = (T @ Xw_h[..., None]).squeeze(-1)
    return Xc_h[..., :3]


def cam_to_pixels(K: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = np.clip(Xc[..., 2], 1e-6, None)
    u = fx * (Xc[..., 0] / Z) + cx
    v = fy * (Xc[..., 1] / Z) + cy
    return np.stack([u, v], axis=-1)

def indices_for_resampled(num_resampled: int, orig_fps: float, new_fps: float, n_orig: int) -> np.ndarray:
    t_res = np.arange(num_resampled, dtype=np.float64) / max(new_fps, 1e-6)
    idx = np.round(t_res * orig_fps).astype(int)
    return np.clip(idx, 0, n_orig - 1)

# ---------------- Core processing and Visualization Functions ----------------
# (These are copied directly from your first script)
def draw_overlay_on_clip(clip_path: Path, out_path: Path, K: np.ndarray, e_clip: np.ndarray, end_pos_clip: np.ndarray, end_quat_clip: np.ndarray) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened(): raise RuntimeError(f"Failed to open {clip_path}")
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    F = e_clip.shape[0]
    axis = np.eye(3) * AXIS_LEN_UNITS
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret or f >= F: break
        for ee in range(end_pos_clip.shape[1]): # Iterate over available arms
            pos_w, quat = end_pos_clip[f, ee, :], end_quat_clip[f, ee, :]
            R_ee = quat_to_rot(quat)
            pts_axes_w = pos_w[None, :] + (R_ee @ axis).T
            pts_w = np.vstack([pos_w[None, :], pts_axes_w])
            Xc = world_to_cam_points(e_clip[f], pts_w)
            uv = cam_to_pixels(K, Xc)
            u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
            cv2.circle(frame, (u0, v0), POINT_RADIUS, (255, 255, 255), -1)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for k in range(3):
                uk, vk = int(round(uv[k+1, 0])), int(round(uv[k+1, 1]))
                cv2.line(frame, (u0, v0), (uk, vk), colors[k], AXIS_THICKNESS, cv2.LINE_AA)
        writer.write(frame)
        f += 1
    writer.release()
    cap.release()

def process_video(video_path: Path, out_root: Path, fps: int, clip_len: int, stride: int, crf: int, preset: str,
                  e_params_orig: np.ndarray, k_params: np.ndarray, end_pos_orig: np.ndarray,
                  end_quat_orig: np.ndarray, eff_pos_orig: np.ndarray) -> Tuple[Path, List[Tuple[Path, int]]]:
    stem = make_video_stem(video_path)
    resampled_path = out_root / "resampled" / f"{stem}_resampled.mp4"
    ensure_dir(resampled_path.parent)
    orig_fps = ffprobe_fps(video_path)
    if not resampled_path.exists():
        resample_to_fps(video_path, resampled_path, fps=fps, crf=crf, preset=preset)
    duration = ffprobe_duration(resampled_path)
    if duration <= 0: print(f"[WARN] ffprobe failed or zero duration: {resampled_path}")
    total_frames_est = max(0, int(round(duration * fps)))
    N = e_params_orig.shape[0]
    idx_map = indices_for_resampled(total_frames_est, orig_fps=orig_fps if orig_fps > 0 else fps, new_fps=fps, n_orig=N)
    e_res, pos_res, quat_res, eff_res = e_params_orig[idx_map], end_pos_orig[idx_map], end_quat_orig[idx_map], eff_pos_orig[idx_map]
    clip_dir, meta_dir, overlay_dir = out_root / "clips", out_root / "metadata", out_root / "overlays"
    ensure_dir(clip_dir); ensure_dir(meta_dir); ensure_dir(overlay_dir)
    products, f0, i = [], 0, 0
    while f0 + clip_len <= total_frames_est:
        clip_path = clip_dir / f"{stem}_clip_{i:05d}.mp4"
        if not clip_path.exists():
            cut_clip_by_frames(resampled_path, start_frame=f0, num_frames=clip_len, fps=fps, output_path=clip_path, crf=crf, preset=preset)
        e_clip, pos_clip, quat_clip, eff_clip = e_res[f0:f0+clip_len], pos_res[f0:f0+clip_len], quat_res[f0:f0+clip_len], eff_res[f0:f0+clip_len]
        npz_path = meta_dir / f"{stem}_clip_{i:05d}.npz"
        np.savez_compressed(npz_path, extrinsics=e_clip, intrinsics=k_params, end_position=pos_clip, end_orientation=quat_clip, effector_position=eff_clip, start_frame=f0, clip_len=clip_len, fps=fps)
        overlay_path = overlay_dir / f"{stem}_clip_{i:05d}_overlay.mp4"
        try:
            if not overlay_path.exists():
                draw_overlay_on_clip(clip_path, overlay_path, K=k_params, e_clip=e_clip, end_pos_clip=pos_clip, end_quat_clip=quat_clip)
        except Exception as viz_err:
            print(f"[WARN] overlay failed for {clip_path}: {viz_err}")
        products.append((clip_path, f0))
        f0 += stride
        i += 1
    return resampled_path, products

def process_video_task(task: Dict[str, Any]) -> Dict[str, Any]:
    vp: Path = task["video_path"]
    try:
        _, clip_infos = process_video(video_path=vp, out_root=task["out_root"], fps=task["fps"], clip_len=task["clip_len"], stride=task["stride"], crf=task["crf"], preset=task["preset"], e_params_orig=task["e_params"], k_params=task["k_params"], end_pos_orig=task["end_position"], end_quat_orig=task["end_orientation"], eff_pos_orig=task["effector_position"])
        items = [{"caption": task["description"], "media_path": str(p)} for (p, _) in clip_infos]
        return {"ok": True, "items": items, "video": str(vp)}
    except Exception as e:
        return {"ok": False, "err": f"{vp}: {e}", "video": str(vp)}


# ==============================================================================
# PART 2: NEW LOGIC FOR READING LIBERO HDF5 AND CREATING TASKS
# ==============================================================================

def extract_libero_data_and_create_video(
    hdf5_path: Path,
    temp_video_dir: Path,
    camera_names: List[str] = ["agentview", "eye_in_hand"],
) -> List[Dict[str, Any]]:
    """
    Extracts data from a LIBERO HDF5 file, creates temporary MP4 videos,
    and returns a list of task dictionaries ready for processing.
    """
    try:
        h5_file = h5py.File(hdf5_path, "r")
    except Exception as e:
        print(f"[WARN] Could not open HDF5 file {hdf5_path}: {e}")
        return []

    demo_key = next(iter(h5_file["data"].keys()))
    episode_data = h5_file["data"][demo_key]
    
    num_frames = episode_data["actions"].shape[0]
    
    # --- Extract all metadata into lists first ---
    ee_pos_list, ee_quat_list, gripper_qpos_list = [], [], []
    
    cam_data = {name: {'e_params_list': [], 'k_params_list': [], 'images': []} for name in camera_names}

    for f in range(num_frames):
        # Robot state
        ee_pos_list.append(episode_data["obs"][f"ee_pos"][()])
        ee_quat_list.append(episode_data["obs"][f"ee_quat"][()])
        gripper_qpos_list.append(episode_data["obs"][f"gripper_qpos"][()])

        # Camera data
        for cam_name in camera_names:
            if f"{cam_name}_rgb" in episode_data["obs"]:
                cam_data[cam_name]['images'].append(episode_data["obs"][f"{cam_name}_rgb"][()])
                cam_data[cam_name]['e_params_list'].append(episode_data["obs"][f"{cam_name}_extrinsics"][()])
                cam_data[cam_name]['k_params_list'].append(episode_data["obs"][f"{cam_name}_intrinsics"][()])

    # --- Convert lists to correctly shaped NumPy arrays ---
    # Stack and create a dummy second arm with zeros/identity quaternion
    end_position = np.stack([np.stack(ee_pos_list), np.zeros_like(np.stack(ee_pos_list))], axis=1) # (N, 2, 3)
    
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0]) # w,x,y,z
    dummy_quats = np.tile(identity_quat, (num_frames, 1))
    end_orientation = np.stack([np.stack(ee_quat_list), dummy_quats], axis=1) # (N, 2, 4)

    gripper_qpos = np.stack(gripper_qpos_list)
    effector_position = np.stack([gripper_qpos, np.zeros_like(gripper_qpos)], axis=1) # (N, 2, 2)
    
    # --- Create tasks for each camera view ---
    tasks = []
    for cam_name in camera_names:
        if not cam_data[cam_name]['images']:
            continue
            
        # Create video from image sequence
        height, width, _ = cam_data[cam_name]['images'][0].shape
        video_stem = f"{hdf5_path.stem}_{cam_name}"
        video_path = temp_video_dir / f"{video_stem}.mp4"
        
        # Original FPS for LIBERO is 30, but we create video at 30 to match data length
        video_fps = 30 
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (width, height))
        for img in cam_data[cam_name]['images']:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        
        # Finalize camera params
        e_params = np.stack(cam_data[cam_name]['e_params_list'])
        k_params = cam_data[cam_name]['k_params_list'][0] # Intrinsics are constant
        
        task_description = hdf5_path.parts[-2] # Use the suite name as description
        
        tasks.append({
            "video_path": video_path,
            "description": task_description,
            "e_params": e_params,
            "k_params": k_params,
            "end_position": end_position,
            "end_orientation": end_orientation,
            "effector_position": effector_position,
        })
        
    h5_file.close()
    return tasks

# ==============================================================================
# PART 3: THE MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process LIBERO dataset videos into clips with synchronized metadata.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the root of the LIBERO dataset (e.g., '.../libero_90')")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the processed output")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS for resampled clips")
    parser.add_argument("--clip_len", type=int, default=121, help="Frames per clip")
    parser.add_argument("--stride", type=int, default=60, help="Frame stride between clips")
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF (lower = better quality)")
    parser.add_argument("--preset", type=str, default="veryfast", help="x264 preset")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Number of parallel workers")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    temp_video_dir = args.output_dir / "temp_videos"
    ensure_dir(temp_video_dir)

    print("Step 1: Finding HDF5 files and extracting data...")
    all_hdf5_files = sorted(list(args.dataset_dir.glob("*/*_demo.hdf5")))
    
    tasks = []
    for hdf5_path in tqdm(all_hdf5_files, desc="Extracting HDF5 data"):
        # This function creates temp videos and extracts all numpy arrays
        extracted_tasks = extract_libero_data_and_create_video(hdf5_path, temp_video_dir)
        
        # Add other processing params to each task
        for task in extracted_tasks:
            task.update({
                "out_root": args.output_dir,
                "fps": args.fps,
                "clip_len": args.clip_len,
                "stride": args.stride,
                "crf": args.crf,
                "preset": args.preset,
            })
        tasks.extend(extracted_tasks)

    if not tasks:
        print("No valid tasks created. Exiting.")
        shutil.rmtree(temp_video_dir)
        return

    print(f"\nStep 2: Start processing {len(tasks)} videos with {max(1, args.jobs)} workers...")
    data = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        future2task = {ex.submit(process_video_task, t): t for t in tasks}
        for fut in tqdm(as_completed(future2task), total=len(tasks), desc="Processing Videos"):
            result = fut.result()
            if result["ok"]:
                data.extend(result["items"])
            else:
                print(f"FAIL: {result['err']}")
    
    print("\nStep 3: Saving dataset metadata...")
    meta_path = args.output_dir / 'dataset.json'
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\nStep 4: Cleaning up temporary video files...")
    shutil.rmtree(temp_video_dir)
    
    print(f"\nDone. Processed data saved to {args.output_dir}")
    print(f"Dataset index file saved to {meta_path}")

if __name__ == "__main__":
    main()