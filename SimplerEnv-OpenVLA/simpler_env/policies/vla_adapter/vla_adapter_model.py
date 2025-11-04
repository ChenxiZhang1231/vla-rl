from typing import Optional, Sequence, List
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModel, AutoProcessor
from collections import deque
from PIL import Image
import torch
import cv2 as cv

from simpler_env.utils.action.action_ensemble import ActionEnsembler
import sys 
sys.path.append("/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev")
from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.action_heads import FlowMatchingActionHead
from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.projectors import NoisyActionProjector
from verl_vla.utils.vla_utils.vla_adapter.openvla_utils import update_auto_map, check_model_logic_mismatch, _load_dataset_stats, find_checkpoint_file, load_component_state_dict


class VLAAdapterInference:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 5,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
        load_ckpt_path: str = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            # unnorm_key = "/inspire/ssd/project/robotsimulation/public/data/bridge/bridge_orig" if unnorm_key is None else unnorm_key
            action_ensemble = True
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = (
                "fractal20220817_data/0.1.0" if unnorm_key is None else unnorm_key
            )
            action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained(
            saved_model_path, trust_remote_code=True
        )
        # self.vla = (
        #     AutoModel.from_pretrained(
        #         saved_model_path,
        #         torch_dtype=torch.bfloat16,
        #         trust_remote_code=True,
        #     )
        #     .eval()
        #     .cuda()
        # )
        config = OpenVLAConfig.from_pretrained(saved_model_path, trust_remote_code=True)
        vla = OpenVLAForActionPrediction.from_pretrained(pretrained_model_name_or_path=saved_model_path,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation='flash_attention_2',
                                            low_cpu_mem_usage=False,
                                            config=config,  
                                            trust_remote_code=True)
        vla.vision_backbone.set_num_images_in_input(1)
        vla.set_version('v1')
        _load_dataset_stats(vla, saved_model_path)
        self.raw_state_dice = vla.state_dict()
        
        ACTION_DIM = 7
        NUM_FLOW_MATCHING_STEPS = 10
        NUM_ACTIONS_CHUNK = 5

        llm_dim = vla.llm_dim
        noisy_action_projector = NoisyActionProjector(
            llm_dim=llm_dim).to(dtype=torch.bfloat16)
        noisy_action_projector_path = find_checkpoint_file(saved_model_path, "noisy_action_projector")
        noisy_action_projector_state_dict = load_component_state_dict(noisy_action_projector_path)
        noisy_action_projector.load_state_dict(noisy_action_projector_state_dict)
        
        vla.noisy_action_projector = noisy_action_projector
        self.noisy_action_projector = vla.noisy_action_projector

        action_head = FlowMatchingActionHead(
                input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_flow_steps=NUM_FLOW_MATCHING_STEPS, num_actions=NUM_ACTIONS_CHUNK,
            ).to(dtype=torch.bfloat16)
        action_head_path = find_checkpoint_file(saved_model_path, "action_head")
        action_head_state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(action_head_state_dict)
        
        vla.action_head = action_head
        self.action_head = vla.action_head
        
        if load_ckpt_path is not None:
            model_state = torch.load(load_ckpt_path, map_location="cpu")
            vla.load_state_dict(model_state, strict=True)
            print(load_ckpt_path)
        self.vla = (
            vla
            .eval()
            .cuda()
        )
        
        
        self.image_size = image_size
        self.action_scale = action_scale
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 20
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon
        self.action_queue = deque(maxlen=self.exec_horizon)

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_queue.clear()

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if len(self.action_queue) == 0:
            if task_description is not None:
                if task_description != self.task_description:
                    self.reset(task_description)

            assert image.dtype == np.uint8
            image = self._resize_image(image)
            image = Image.fromarray(image).convert("RGB")
            
            # prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_description.lower()}?<|im_end|>\n<|im_start|>assistant\n'
            inputs  = self.processor(prompt, image)

            with torch.no_grad():
                raw_actions, return_dict = self.vla.predict_action(
                    input_ids=inputs['input_ids'].to(self.vla.device),
                    pixel_values=inputs['pixel_values'].to(torch.bfloat16).to(self.vla.device),
                    attention_mask=inputs['attention_mask'].to(self.vla.device),
                    unnorm_key=self.unnorm_key,
                    do_sample=False,
                    proprio=None,
                    proprio_projector=None,
                    noisy_action_projector=self.noisy_action_projector,
                    action_head=self.action_head,
                    use_film=False,
                    use_sde=False,
                    a_shape=(5,7)
                    # use_sde=True,
                )
                raw_actions = raw_actions[0]  # ck, 7
            for ac in raw_actions:
                 self.action_queue.append(ac)
        
        raw_actions = self.action_queue.popleft()
            
        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
