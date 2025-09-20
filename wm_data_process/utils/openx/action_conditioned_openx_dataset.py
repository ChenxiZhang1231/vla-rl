import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/world_model/ActionWorldModel")
from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)

def quats_to_euler(quats, order="xyz", degrees=False): 
    q_flat = quats.reshape(-1, 4)
    r = R.from_quat(q_flat)
    euler_flat = r.as_euler(order, degrees=degrees)
    return euler_flat.reshape(quats.shape[0], quats.shape[1], 3)

class ActionConditionedOpenxDataset(Dataset):
    def __init__(
        self,
        data_path='',
        sequence_interval=1,
        num_frames=13,
        cam_ids=[0],
        accumulate_action=False,
        video_size=[128, 128],
        val_start_frame_interval=1,
        debug=False,
        normalize=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        mode="train",
    ):
        """Dataset class for loading 3D robot action-conditioned data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.

        The dataset loads robot trajectories and computes:
        - RGB video frames from specified camera views
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.start_frame_interval = 10
        elif mode == "val":
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.start_frame_interval = val_start_frame_interval
        self.data_path = data_path
        self.video_path = os.path.join(self.data_path, 'clips')
        self.ann_path = os.path.join(self.data_path, 'metadata')
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.sequence_length = num_frames
        self.normalize = normalize
        self.load_action = load_action
        self.pre_encode = False

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action
        self.load_t5_embeddings = load_t5_embeddings

        self.action_dim = 14  # ee xyz (3) + ee euler (3) + gripper(1) * 2 for 2 arms
        self.c_act_scaler = [20, 20, 20, 20, 20, 20, 1e-5, 20, 20, 20, 20, 20, 20, 1e-5]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.ann_path)

        print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".npz") and 'agentview' in f)]
        total_num = len(ann_files)
        if self.mode == 'train':
            return ann_files[:total_num * 14 // 16]
        elif self.mode == 'test':
            return ann_files[total_num * 14 // 16: total_num * 15 // 16]
        elif self.mode == 'val':
            return ann_files[total_num * 15 // 16: ]
            # return ann_files[-100:]

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples
    
    def _load_and_process_ann_file(self, ann_file):
        samples = []
        ann = np.load(ann_file)

        n_frames = ann['end_position'].shape[0]
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        from decord import VideoReader, cpu  # Importing here due to malloc errors on ARM when importing on top level

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def _get_frames(self, video_path, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=10)  # fps
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, video_path, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(video_path, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, temp_cam_id
    
    def _get_robot_states(self, label, frame_ids):
        extrinsics = np.array(label['extrinsics']) # (N, 4, 4)
        intrinsics = np.array(label['intrinsics']) # (3, 3)
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position'])[..., 0] # (N, 2)
        effector_position = (effector_position > 0.01) * 1.0
        
        
        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states  #[:, :, 0]

    def _get_all_robot_states(self, label, frame_ids):
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position']) # (N, 2)
        
        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states
    
    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            for i in range(arm_states.shape[1]):
                first_xyz = arm_states[0, i, 0:3]
                first_rpy = arm_states[0, i, 3:6]
                first_rotm = euler2rotm(first_rpy)
                for k in range(1, self.sequence_length):
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                    rel_rotm = first_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        else:
            for i in range(arm_states.shape[1]):
                for k in range(1, self.sequence_length):
                    prev_xyz = arm_states[k - 1, i, 0:3]
                    prev_rpy = arm_states[k - 1, i, 3:6]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def __getitem__(self, index, cam_id=None, return_video=False):
        try:
            if self.mode != "train":
                np.random.seed(index)
                random.seed(index)

            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            label = np.load(ann_file)
            video_path = ann_file.replace('metadata', 'clips').replace('.npz', '.mp4')
            black_path = ann_file.replace('metadata', 'blacks').replace('.npz', '_black.mp4')
            actions = np.array(label['actions'])[frame_ids][:-1].reshape(self.sequence_length - 1, self.action_dim)
            actions = torch.from_numpy(actions)
            # arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            # actions = np.concatenate([arm_states, gripper_states[..., None]], axis=-1)[:-1].reshape(self.sequence_length - 1, self.action_dim)
            # actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            # actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, cam_id = self._get_obs(video_path, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video.to(dtype=torch.uint8)
                data['first_frame'] = video[:, 0, ...]
                if False and os.path.exists(black_path):
                    black, _ = self._get_obs(black_path, frame_ids, cam_id, pre_encode=False)
                    black = black.permute(1, 0, 2, 3)
                    data["blacks"] = black.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file
            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = video_path

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings_path = ann_file.replace('metadata', '.precomputed/conditions/clips').replace('.npz', '.pt')
                t5_embeddings_dict = torch.load(t5_embeddings_path)
                t5_embeddings = t5_embeddings_dict['prompt_embeds'][:, :1024] # (256, 4096)
                data["t5_text_embeddings"] = t5_embeddings.cuda() 
                data["t5_text_mask"] = t5_embeddings_dict['prompt_attention_mask'].to(torch.int64).cuda() # (256, )
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
                data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 10
            data["image_size"] = 256 * torch.ones(4).cuda()  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
        
if __name__ == '__main__':
    dataset = ActionConditionedOpenxDataset(
        data_path='',
        mode='val'
    )
    data = dataset[1]
    # import pdb; pdb.set_trace()