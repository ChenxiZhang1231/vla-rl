from pathlib import Path

import numpy as np
from h5py import File
import math


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm

def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)  # noqa: E741
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # (-pi , pi]
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    while y > np.pi:
        y -= 2 * np.pi
    while y <= -np.pi:
        y += 2 * np.pi
    while z > np.pi:
        z -= 2 * np.pi
    while z <= -np.pi:
        z += 2 * np.pi
    return np.array([x, y, z])



def load_local_episodes(input_h5: Path, use_delta_action=False):
    with File(input_h5, "r") as f:
        for demo in f["data"].values():
            demo_len = len(demo["obs/agentview_rgb"])
            # (-1: open, 1: close) -> (0: close, 1: open)
 
            state = np.concatenate(
                [
                    np.array(demo["obs/ee_states"]),
                    np.array(demo["obs/gripper_states"]),
                ],
                axis=1,
            )

            if use_delta_action:
                end_position = state[:, 0:3]
                end_rotation = state[:, 3:6]
                end_gripper = state[:, 6:7]
                action = np.zeros((state.shape[0]-1, 7))
                for k in range(1, state.shape[0]):
                    prev_xyz = end_position[k-1]
                    prev_rpy = end_rotation[k-1]
                    prev_rotm = euler2rotm(prev_rpy)
                
                    curr_xyz = end_position[k]
                    curr_rpy = end_rotation[k]
                    curr_rotm = euler2rotm(curr_rpy)
                    curr_gripper = end_gripper[k]
                
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k-1, 0:3] = rel_xyz
                    action[k-1, 3:6] = rel_rpy
                    action[k-1, 6] = curr_gripper
                action = np.concatenate([action, np.array([[0,0,0,0,0,0,action[-1,-1]]])])
            else:
                
                action = np.array(demo["actions"])
                action = np.concatenate(
                    [
                        action[:, :6],
                        (1 - np.clip(action[:, -1], 0, 1))[:, None],
                    ],
                    axis=1,
                )
                
            episode = {
                "observation.images.image": np.array(demo["obs/agentview_rgb"]),
                "observation.images.wrist_image": np.array(demo["obs/eye_in_hand_rgb"]),
                "observation.state": np.array(state, dtype=np.float32),
                "observation.states.ee_state": np.array(demo["obs/ee_states"], dtype=np.float32),
                "observation.states.joint_state": np.array(demo["obs/joint_states"], dtype=np.float32),
                "observation.states.gripper_state": np.array(demo["obs/gripper_states"], dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]
