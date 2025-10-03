#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
import os
import re
import json
import inspect
import copy
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from functools import partial, wraps
import tempfile
import shutil
from torch.distributions import Normal
import numpy as np

import safetensors
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel
# from transformers.configuration_utils import PretrainedConfig
# from transformers.integrations import is_deepspeed_zero3_enabled
# from transformers.utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
# from transformers.quantizers import AutoHfQuantizer, HfQuantizer
# from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
# from transformers.utils import (CONFIG_NAME,
#                                 is_torch_greater_or_equal,
#                                 is_accelerate_available,
#                                 is_safetensors_available,
#                                 cached_file,
#                                 extract_commit_hash,
#                                 is_peft_available,
#                                 find_adapter_config_file,
#                                 is_offline_mode,
#                                 logging,
#                                 ContextManagers)
# from transformers.modeling_utils import (_get_resolved_checkpoint_files,
#                                          _find_mismatched_keys,
#                                          _find_missing_and_unexpected_keys,
#                                          _get_device_map,
#                                          load_state_dict,
#                                          expand_device_map,
#                                          get_disk_only_shard_files,
#                                          caching_allocator_warmup,
#                                          is_fsdp_enabled,
#                                          is_local_dist_rank_0,
#                                          _get_torch_dtype,
#                                          _is_ds_init_called,
#                                          _load_state_dict_into_meta_model)
# from accelerate.utils import (
#     save_offload_index,
# )
# from accelerate import dispatch_model, infer_auto_device_map
# from safetensors import safe_open
# from safetensors.torch import load_file as safe_load_file
# from safetensors.torch import save_file as safe_save_file
# from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from pathlib import Path
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.utils import get_safe_dtype
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
# logger = logging.get_logger(__name__)

def restore_default_torch_dtype(func):
    """
    Decorator to restore the default torch dtype
    at the end of the function. Serves
    as a backup in case calling the function raises
    an error after the function has changed the default dtype but before it could restore it.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        old_dtype = torch.get_default_dtype()
        try:
            return func(*args, **kwargs)
        finally:
            torch.set_default_dtype(old_dtype)

    return _wrapper

def canonicalise(k: str) -> str:
    """
    Remove dataset-variant markers like '.so100-blue_' or '.so100_' from a
    normalisation-buffer key.
    """
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: dict[str, torch.Tensor], ref_keys: set[str], *, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    • Re-keys `checkpoint ` so that every entry matches the *reference* key set.
    • If several variant keys collapse to the same canonical name we keep the
      first one and log the collision.
    • Returns the new dict + a list of entries that could not be matched.
    """
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:  # duplicate after collapsing
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    if verbose:
        for canon, variants in collisions.items():
            print(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            print(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: dict, rename_str: str):
    """
    Renames keys in a checkpoint dictionary based on the given rename string.

    Args:
        checkpoint (dict): The checkpoint dictionary.
        rename_str (str): A string specifying key mappings in the format "old1//new1,old2//new2".

    Returns:
        dict: The modified checkpoint with renamed keys.
    """

    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def load_smolvla(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    state_dict = safetensors.torch.load_file(filename, device=device)

    # Optional user-supplied renames (e.g. "model._orig_mod.//model.")
    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))

    # HACK(aliberts): to not overwrite normalization parameters as they should come from the dataset
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if not all(key.startswith(norm_keys) for key in missing) or unexpected:
        raise RuntimeError(
            "SmolVLA %d missing / %d unexpected keys",
            len(missing),
            len(unexpected),
        )

    return model

def _repeat_past_kv_batch(past_kv, repeats: int):
    """
    把 past_key_values 的 batch 维复制 repeats 次。
    兼容 tuple/list 的递归结构：每个张量都沿 dim=0 repeat_interleave。
    """
    if torch.is_tensor(past_kv):
        return past_kv.repeat_interleave(repeats, dim=0)
    elif isinstance(past_kv, (list, tuple)):
        return type(past_kv)(_repeat_past_kv_batch(x, repeats) for x in past_kv)
    elif isinstance(past_kv, dict):
        return {k: _repeat_past_kv_batch(v, repeats) for k, v in past_kv.items()}
    else:
        raise TypeError(f"Unsupported past_kv type: {type(past_kv)}")
    
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


# def resize_with_pad(img, width, height, pad_value=-1):
#     # assume no-op when width height fits already
#     if img.ndim != 4:
#         raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

#     cur_height, cur_width = img.shape[2:]

#     ratio = max(cur_width / width, cur_height / height)
#     resized_height = int(cur_height / ratio)
#     resized_width = int(cur_width / ratio)
#     resized_img = F.interpolate(
#         img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
#     )

#     pad_height = max(0, int(height - resized_height))
#     pad_width = max(0, int(width - resized_width))

#     # pad on left and top of image
#     padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
#     return padded_img

def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim < 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[-2:]
    *lead, C, cur_height, cur_width = img.shape
    img_4d = img.reshape(-1, C, cur_height, cur_width)
    
    ratio = max(cur_width / float(width), cur_height / float(height))
    resized_height = int(cur_height / ratio)
    resized_width  = int(cur_width  / ratio)
    
    resized_img = F.interpolate(
        img_4d, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    out = padded_img.reshape(*lead, C, height, width)

    return out


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


# class SmolVLAPolicy(PreTrainedPolicy):
class SmolVLAPolicy(PreTrainedModel):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"
    # _no_split_modules = ['LlamaDecoderLayer']

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        # self._tp_plan = {}
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    # HACK(aliberts, danaaubakirova): we overwrite this classmethod here to fix smolVLA-specific issues
    @classmethod
    def _load_as_safetensor(
        cls,
        model: "SmolVLAPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ):
        safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return load_smolvla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False,
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
                
        if batch.get("lang_tokens", None) is None:
            lang_tokens, lang_masks = self.prepare_language(batch)
            is_train = True
        else:
            lang_tokens, lang_masks = batch["lang_tokens"], batch["lang_masks"]
            # is_train = False
            is_train = True
            
        images, img_masks = self.prepare_images(batch, is_train=is_train)
        # state = self.prepare_state(batch)
        state = None

        if recompute_log_prob:
            x_t = batch['x_t']
            t = batch['t']
            x_next = batch['x_next']
            finish_step = batch['finish_step']
            images = images[0]
            img_masks = img_masks[0]
            B, S, _, H, W = images.shape
            img_masks = img_masks.unsqueeze(-1).unsqueeze(-1).repeat(1, S, 1)
            (logp_action,
             logp_step, 
             logp_outer, 
             logp_joint, 
             logp_elem,
             ent_step, 
             ent_outer, 
             ent_joint,
             mean,
             std_dev_t,
             out_metric,
             ) = self.model.recompute_logprob(images, img_masks, lang_tokens, lang_masks, state, x_t, t, x_next, finish_step)
            return_dict = {
                "logp_action": logp_action,
                "logp_step": logp_step, 
                "logp_outer": logp_outer, 
                "logp_joint": logp_joint,
                "logp_elem": logp_elem,
                "ent_step": ent_step, 
                "ent_outer": ent_outer, 
                "ent_joint": ent_joint,
                "mean": mean,
                "std": std_dev_t,
                "out_metric": out_metric,
                }
            return None, lang_tokens, lang_masks, return_dict
        
        if use_sde:
            return_dict = self.model.sample_actions_sde(images, img_masks, lang_tokens, lang_masks, state, noise=noise, return_logprob=return_logprob)
            if return_logprob:
                return_dict, log_probs = return_dict
            actions = return_dict['x_next'][-1]
        else:
            return_dict = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            actions = return_dict['x_next'][-1]

        # Unpad actions
        # original_action_dim = self.config.action_feature.shape[0]
        original_action_dim = 7
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return (actions, lang_tokens, lang_masks, return_dict, log_probs) if return_logprob else actions, lang_tokens, lang_masks, return_dict

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions_pred, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
            batch,
            noise,
            use_sde=use_sde,
            return_logprob=return_logprob,
            recompute_log_prob=recompute_log_prob
        )
        if actions_pred is not None:
            actions = actions_pred[:, :self.config.n_action_steps]
        else:
            actions = None
        return actions, lang_tokens, lang_masks, return_dict
    
    def predict_action_chunk_update(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False
    ) -> Tensor:

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions_pred, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
            batch,
            noise,
            use_sde=use_sde,
            return_logprob=return_logprob,
            recompute_log_prob=recompute_log_prob
        )
        if actions_pred is not None:
            actions = actions_pred[:, :self.config.n_action_steps]
        else:
            actions = None
        return actions, lang_tokens, lang_masks, return_dict
    
    def compute_values(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False
    ) -> Tensor:

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        return_dict = self._get_compute_values(
            batch,
            noise,
            use_sde=use_sde,
            return_logprob=return_logprob,
            recompute_log_prob=recompute_log_prob
        )
        return return_dict


    def _get_compute_values(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False,
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
                
        if batch.get("lang_tokens", None) is None:
            lang_tokens, lang_masks = self.prepare_language(batch)
            is_train = True
        else:
            lang_tokens, lang_masks = batch["lang_tokens"], batch["lang_masks"]
            # is_train = False
            is_train = True
            
        images, img_masks = self.prepare_images(batch, is_train=is_train)
        # state = self.prepare_state(batch)
        state = None
        x_t = batch['x_t']
        t = batch['t']
        x_next = batch['x_next']
        finish_step = batch['finish_step']
        images = images[0]
        img_masks = img_masks[0]
        B, S, _, H, W = images.shape
        img_masks = img_masks.unsqueeze(-1).unsqueeze(-1).repeat(1, S, 1)
        values = self.model.compute_values(images, img_masks, lang_tokens, lang_masks, state, x_t, t, x_next, finish_step)
        return {"values": values}
        
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    # def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
    def forward(self, 
                batch: dict[str, Tensor], 
                noise=None, 
                time=None
    ) -> dict[str, Tensor]:
        
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch, is_train=False):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            if is_train:
                img = batch[key]
            else:
                img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        
        # for num_empty_cameras in range(len(missing_img_keys)):
        #     if num_empty_cameras >= self.config.empty_cameras:
        #         break
        #     img = torch.ones_like(img) * -1
        #     mask = torch.zeros_like(mask)
        #     images.append(img)
        #     img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch['observation.images.image'].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch['observation.images.image'].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        if batch[OBS_STATE].ndim == 4:
            state = batch[OBS_STATE]
        else:
            state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        
        # self.set_requires_grad_all_true()
        
    def set_requires_grad_all_true(self):
        for params in self.parameters():
            params.requires_grad = True


    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            # dtype=torch.bfloat16,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        # time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.bfloat16)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        if state is not None:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        # suffix_out = suffix_out.to(dtype=torch.bfloat16)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_next = x_t + dt * v_t
            
            x_t_all.append(x_t)
            t_all.append(expanded_time.detach().cpu())
            x_next_all.append(x_next)
            
            x_t = x_next
            time += dt
            
        return_dict = {
            "x_t": x_t_all,
            "t": t_all,
            "x_next": x_next_all,
        }
        return return_dict

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
    
    def sample_actions_sde(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, return_logprob=False) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        sqrt_abs_dt = torch.sqrt(-dt)

        sde_sigma_max = 0.07
        sde_sigma_power = 1.5
        x_t = noise
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while t >= -dt / 2:
            t_b = t.expand(bsize)
            # t_safe = t_b
            # t_safe = t_b.clamp(1e-4, 1 - 1e-4).to(dtype=torch.float32)
            v_t = self.denoise_step_sde(
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_b,
            )  # bs, num_step, action_dim
            # sigma_t = self.config.noise_level * torch.sqrt(t_safe / (1 - t_safe + 1e-6))
            # sigma_t  = (sde_sigma_max * t.pow(sde_sigma_power))  # bs, 1, 1
            sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t.device, dtype=v_t.dtype)
            index = (self.config.num_steps * (1 - t)).to(torch.long)
            sigma = sigmas[index]
            sigma_max = sigmas[1]
            noise_level = 0.7
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
            
            # sde_sigma_max = 0.07
            # sde_sigma_power = 1.5
            # sigma_t = (sde_sigma_max * t_safe.pow(sde_sigma_power))[..., None, None]  # [B,S,K,1,1]
            
            drift = v_t + (std_dev_t ** 2 / (2 * t + 1e-6)) * (x_t + (1 - t) * v_t)
            mean = x_t + drift * dt
            std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)
            eps = torch.randn_like(x_t)
            x_next = mean + std * eps
            
            
            if return_logprob:
                lp = Normal(mean, std).log_prob(x_next.detach())
                lp = lp.sum(dim=tuple(range(1, lp.ndim)))
                log_probs.append(lp)

            x_t_all.append(x_t)
            t_all.append(t_b.detach().cpu())
            x_next_all.append(x_next)
            x_t  = x_next
            t = t + dt

        return_dict = {
            "x_t": x_t_all,
            "t": t_all,
            "x_next": x_next_all,
        }
        return (return_dict, log_probs) if return_logprob else return_dict

    def denoise_step_sde(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def recompute_logprob_raw(
        self,
        images,             # [B,S,3,H,W]
        img_masks,          # [B,S,1]
        lang_tokens,        # [B,S,L]
        lang_masks,         # [B,S,L]
        state,              # [B,S,1,Ds]
        x_t,                # [B,S,K,50,Da]
        t,                  # [B,S,K]
        x_next,             # [B,S,K,50,Da]
        finish_step,        # [B]
    ):
        """
        返回:
        logp_step  : [B,S,K]   —— 每个外层步、每个流步的 log-prob（已mask）
        logp_outer : [B,S]     —— 每个外层步联合（对K求和，已mask）
        logp_joint : [B]       —— 整条有效轨迹联合（对S再求和，已mask）
        """
        # ---------- 基本维度 ----------
        device = lang_tokens.device
        B, S, K, CH, D = x_t.shape  # CH=50, D=动作维
        assert t.shape == (B, S, K)
        assert x_next.shape == (B, S, K, CH, D)
        dt = -1.0 / float(K)                       # 按你的设定：K个等步，t: 1→0
        sqrt_abs_dt = math.sqrt(-dt)
        dtype = torch.float32                      # 计算用 fp32 更稳

        # ---------- 构建 prefix KV（按 [B*S] 批量） ----------
        BS = B * S
        # 展平外层时间 S 维
        def _flat(x, keep_tail=0):
            # 把 [B,S,...] → [B*S,...]
            if keep_tail == 0:
                return x.reshape(BS, *x.shape[2:])
            elif keep_tail == 1:
                return x.reshape(BS, x.shape[-1])
            else:
                raise ValueError

        images_flat    = _flat(images)
        img_masks_flat = _flat(img_masks)
        lang_tokens_f  = _flat(lang_tokens)
        lang_masks_f   = _flat(lang_masks)
        state_flat     = _flat(state) if state is not None else None

        # prefix embeds & KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            [images_flat], [img_masks_flat.squeeze(-1)], lang_tokens_f, lang_masks_f, state=state_flat
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        # 把 KV 从 [BS,...] 复制成 [BS*K,...]
        prefix_pad_masks_rep = prefix_pad_masks.repeat_interleave(K, dim=0)   # [BS*K, Lp]
        past_kv_rep = _repeat_past_kv_batch(past_key_values, repeats=K)       # 每层KV沿batch复制K次

        # ---------- 展平到 [BS*K]，一次性前向得到 v_t ----------
        # x_t / t / x_next 搬到 device + fp32
        x_t_f    = x_t.to(device=device, dtype=dtype)       # [B,S,K,CH,D]
        x_next_f = x_next.to(device=device, dtype=dtype)
        t_f      = t.to(device=device, dtype=dtype)

        # [B,S,K,...] → [BS*K,...]
        x_t_flat    = x_t_f.reshape(BS * K, CH, D)          # [BSK,CH,D]
        x_next_flat = x_next_f.reshape(BS * K, CH, D)       # [BSK,CH,D]
        t_flat      = t_f.reshape(BS * K)                   # [BSK]

        # 调 denoise_step_sde（一次处理所有 step）
        # 需要把 prefix mask/KV 扩到 [BS*K]（上面已处理）
        # denoise_step_sde 内部会 embed_suffix(x_t_flat, t_flat)
        # outputs_embeds, _ = self.vlm_with_expert.forward(
        #     attention_mask=torch.cat([
        #         prefix_pad_masks_rep[:, None, :].expand(BS * K, self.embed_suffix(x_t_flat, t_flat)[1].shape[1], prefix_pad_masks_rep.shape[1]),
        #         make_att_2d_masks(*self.embed_suffix(x_t_flat, t_flat)[1:])  # (suffix_pad_masks, suffix_att_masks)
        #     ], dim=2),
        #     position_ids=(torch.sum(prefix_pad_masks_rep, dim=-1)[:, None] + torch.cumsum(self.embed_suffix(x_t_flat, t_flat)[1], dim=1) - 1),
        #     past_key_values=past_kv_rep,
        #     inputs_embeds=[None, self.embed_suffix(x_t_flat, t_flat)[0]],
        #     use_cache=self.config.use_cache,
        #     fill_kv_cache=False,
        # )
        
        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t_flat, t_flat)   # [BSK, Ls, H], [BSK, Ls], [BSK, Ls]

        prefix_pad2d = prefix_pad_masks_rep[:, None, :].expand(BS*K, suffix_pad.shape[1], prefix_pad_masks_rep.shape[1])
        suffix_att2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att2d   = torch.cat([prefix_pad2d, suffix_att2d], dim=2)
        pos_ids      = (torch.sum(prefix_pad_masks_rep, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1)
        
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att2d,
            position_ids=pos_ids,
            past_key_values=past_kv_rep,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        # breakpoint()
        # 上面为了避免多次计算，可拆开：先一次 self.embed_suffix(...)，再复用结果
        # 这里做个小优化：实际写法请参考下面“更高效版本”注释
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]      # [BSK, CH, H]
        v_t_flat   = self.action_out_proj(suffix_out).to(dtype)           # [BSK, CH, D]
        v_t_all    = v_t_flat.view(B, S, K, CH, D)                        # [B,S,K,CH,D]

        # ---------- 构造 sigma(t)、漂移、mean/std ----------
        t_safe = t_f.clamp(1e-4, 1 - 1e-4).view(B, S, K)                  # [B,S,K]
        t3     = t_safe[..., None, None]                                   # [B,S,K,1,1]
        # breakpoint()
        
        sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t_flat.device, dtype=v_t_flat.dtype)
        index = (K * (1 - t)).to(torch.long)
        sigma = sigmas[index]
        sigma_max = sigmas[1]
        noise_level = 0.7
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
        std_dev_t = std_dev_t[..., None, None]
        
        # sde_sigma_max = 0.07
        # sde_sigma_power = 1.5
        # sigma_t = (sde_sigma_max * t_safe.pow(sde_sigma_power))[..., None, None]  # [B,S,K,1,1]
        
        drift = v_t_all + (std_dev_t**2 / (2 * t3 + 1e-6)) * (x_t_f + (1 - t3) * v_t_all)

        mean = x_t_f + dt * drift                                          # [B,S,K,CH,D]
        std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)                     # [B,S,K,1,1]

        # ---------- finish_step → 掩码 [B,S,CH] ----------
        # 有效动作：扁平索引 0..finish_step[b]（含）为1，其它0
        # 映射到外层步 s 和 chunk 内索引 c
        #   s_finish = finish_step // CH
        #   c_finish = finish_step %  CH
        CH_idx = torch.arange(CH, device=device)[None, None, :]            # [1,1,CH]
        S_idx  = torch.arange(S,  device=device)[None, :, None]            # [1,S,1]
        s_fin  = (finish_step.to(device) // CH).view(B, 1, 1)              # [B,1,1]
        c_fin  = (finish_step.to(device) %  CH).view(B, 1, 1)              # [B,1,1]

        mask_before = (S_idx <  s_fin).float()                             # [B,S,1]（广播到CH）
        mask_equal  = (S_idx == s_fin).float() * (CH_idx <= c_fin).float() # [B,S,CH]
        mask_actions = mask_before.expand(B, S, CH) + mask_equal           # [B,S,CH] ∈ {0,1}
        # [B,S,K,CH,1]
        mask_elem = mask_actions[:, :, None, :, None]                      # [B,S,K,CH,1]

        # ---------- 逐元素 log-prob（各向同性对角高斯） ----------
        # 注意 detach x_next，避免噪声路径入图
        # log_prob = (
        #     -((x_next_f.detach() - mean) ** 2) / (2 * ((std)**2))
        #     - torch.log(std)
        #     - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        # )
        lp_elem = Normal(mean, std).log_prob(x_next_f.detach())            # [B,S,K,CH,D]
        lp_elem = lp_elem * mask_elem   
        original_action_dim = 7
        lp_elem = lp_elem[..., :original_action_dim]
        # logp_action = lp_elem.mean(dim=(-3))     # [B,S,50,7]
        logp_action = lp_elem.sum(dim=(-3)) 
        # 聚合：先动作维 (CH,D)，得到每步的 logp
        logp_step = lp_elem.sum(dim=(-1, -2))                               # [B,S,K]
        # 外层步联合（对K求和）
        logp_outer = logp_step.sum(dim=2)                                   # [B,S]
        # 全轨迹联合（对S求和；后续也可只取到 s_fin 的和）
        logp_joint = logp_outer.sum(dim=1)                                  # [B]
        
        
        # ---------- 对应的熵（entropy） ----------
        # 单维熵: 0.5*(1 + log(2π)) + log(std)
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        # std: [B,S,K,1,1]  ->  [B,S,K]
        h_per_dim = c0 + torch.log(std).squeeze(-1).squeeze(-1)            # [B,S,K]

        # 有效维度个数 = 有效动作个数(沿 CH 求和) * D
        # mask_actions: [B,S,CH] ∈ {0,1}
        num_valid_actions = mask_actions.sum(dim=-1)                        # [B,S]
        num_valid_dims    = (num_valid_actions * D).unsqueeze(-1)           # [B,S,1]
        num_valid_dims    = num_valid_dims.to(h_per_dim.dtype)

        # 每步熵 = 单维熵 × 有效维度数
        ent_step  = h_per_dim * num_valid_dims                              # [B,S,K]
        ent_outer = ent_step.sum(dim=2)                                     # [B,S]
        ent_joint = ent_outer.sum(dim=1)                                    # [B]
        return logp_action, logp_step, logp_outer, logp_joint, ent_step, ent_outer, ent_joint
    
    def recompute_logprob(
        self,
        images,             # [B,S,3,H,W]
        img_masks,          # [B,S,1]
        lang_tokens,        # [B,S,L]
        lang_masks,         # [B,S,L]
        state,              # [B,S,1,Ds]
        x_t,                # [B,S,K,50,Da]
        t,                  # [B,S,K]
        x_next,             # [B,S,K,50,Da]
        finish_step,        # [B]
    ):
        """
        返回:
        logp_step  : [B,S,K]   —— 每个外层步、每个流步的 log-prob（已mask）
        logp_outer : [B,S]     —— 每个外层步联合（对K求和，已mask）
        logp_joint : [B]       —— 整条有效轨迹联合（对S再求和，已mask）
        """
        # ---------- 基本维度 ----------
        device = lang_tokens.device
        B, S, K, CH, D = x_t.shape  # CH=50, D=动作维
        assert t.shape == (B, S, K)
        assert x_next.shape == (B, S, K, CH, D)
        dt = -1.0 / float(K)                       # 按你的设定：K个等步，t: 1→0
        sqrt_abs_dt = math.sqrt(-dt)
        dtype = torch.float32                      # 计算用 fp32 更稳

        # ---------- 构建 prefix KV（按 [B*S] 批量） ----------
        BS = B * S
        # 展平外层时间 S 维
        def _flat(x, keep_tail=0):
            # 把 [B,S,...] → [B*S,...]
            if keep_tail == 0:
                return x.reshape(BS, *x.shape[2:])
            elif keep_tail == 1:
                return x.reshape(BS, x.shape[-1])
            else:
                raise ValueError

        images_flat    = _flat(images)
        img_masks_flat = _flat(img_masks)
        lang_tokens_f  = _flat(lang_tokens)
        lang_masks_f   = _flat(lang_masks)
        state_flat     = _flat(state) if state is not None else None

        x_t_flat    = x_t.reshape(BS * K, CH, D).to(device=device, dtype=dtype)
        x_next_f    = x_next.to(device=device, dtype=dtype) # 保持原始形状，后面用
        t_flat      = t.reshape(BS * K).to(device=device, dtype=dtype)
        # breakpoint()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            [images_flat], [img_masks_flat.squeeze(-1)], lang_tokens_f, lang_masks_f, state=state_flat
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t_flat, t_flat)
        prefix_embs_rep = prefix_embs.repeat_interleave(K, dim=0)
        prefix_pad_masks_rep = prefix_pad_masks.repeat_interleave(K, dim=0)
        prefix_att_masks_rep = prefix_att_masks.repeat_interleave(K, dim=0)

        pad_masks = torch.cat([prefix_pad_masks_rep, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_rep, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_rep, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t_flat = self.action_out_proj(suffix_out)
        v_t_all    = v_t_flat.view(B, S, K, CH, D)
        
        x_t_f = x_t.to(device=device, dtype=dtype)
        t_f = t.to(device=device, dtype=dtype)

        t_safe = t_f.clamp(1e-4, 1 - 1e-4).view(B, S, K)
        t3     = t_safe[..., None, None]
        
        sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t_flat.device, dtype=v_t_flat.dtype)
        index = (K * (1 - t)).to(torch.long)
        sigma = sigmas[index]
        sigma_max = sigmas[1]
        noise_level = 0.7
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
        std_dev_t = std_dev_t[..., None, None]
        
        drift = v_t_all + (std_dev_t**2 / (2 * t3 + 1e-6)) * (x_t_f + (1 - t3) * v_t_all)

        mean = x_t_f + dt * drift
        std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)

        CH = self.config.n_action_steps
        CH_idx = torch.arange(CH, device=device)[None, None, :]
        S_idx  = torch.arange(S,  device=device)[None, :, None]
        s_fin  = (finish_step.to(device) // CH).view(B, 1, 1)
        c_fin  = (finish_step.to(device) %  CH).view(B, 1, 1)

        mask_before = (S_idx <  s_fin).float()
        # mask_equal  = (S_idx == s_fin).float() * (CH_idx <= c_fin).float()
        mask_equal  = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
        mask_actions = mask_before.expand(B, S, CH) + mask_equal
        mask_elem = mask_actions[:, :, None, :, None]
        # breakpoint()
        # lp_elem = Normal(mean, std).log_prob(x_next_f.detach())
        lp_elem = (
            -((x_next_f.detach() - mean) ** 2) / (2 * ((std)**2))
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        lp_elem = lp_elem[..., :self.config.n_action_steps, :]            # [B,S,K,CH,D]
        lp_elem = lp_elem * mask_elem
        original_action_dim = 7
        lp_elem = lp_elem[..., :original_action_dim]
        logp_action = lp_elem.sum(dim=(-3))
        
        logp_step = lp_elem.sum(dim=(-1, -2))
        logp_outer = logp_step.sum(dim=2)
        logp_joint = logp_outer.sum(dim=1)
        
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        h_per_dim = c0 + torch.log(std).squeeze(-1).squeeze(-1)
        num_valid_actions = mask_actions.sum(dim=-1)
        num_valid_dims    = (num_valid_actions * D).unsqueeze(-1).to(h_per_dim.dtype)
        
        ent_step  = h_per_dim * num_valid_dims
        ent_outer = ent_step.sum(dim=2)
        ent_joint = ent_outer.sum(dim=1)
        
        # breakpoint()
        # logp_action = logp_action / K
        # original_action_dim = 7
        # valid_CH = mask_actions.sum(dim=-1)                 # [B,S]
        # num_elems = (valid_CH * original_action_dim * K).clamp_min(1)
        # logp_outer = logp_outer / num_elems

        mean = mean[..., :original_action_dim]
        std_dev_t = std_dev_t[..., :original_action_dim]
        std = std[..., :original_action_dim]
        out_metric = self.summarize_logprob_metrics(
            logp_step, logp_outer, logp_joint,
            mean, std, x_next_f, mask_actions, mask_elem,
            t, sigmas, finish_step, original_action_dim=7
        )
        return (
            logp_action, 
            logp_step, 
            logp_outer, 
            logp_joint, 
            lp_elem,
            ent_step, 
            ent_outer, 
            ent_joint, 
            mean, 
            # std_dev_t,
            std,
            out_metric,
        )
    
    def summarize_logprob_metrics(
        self,
        logp_step, logp_outer, logp_joint,
        mean, std, x_next_f, mask_actions, mask_elem,
        t, sigmas, finish_step, original_action_dim=7
    ):
        out = {}
        # 1) NLL / BPD
        B = logp_joint.shape[0]
        num_valid_actions = mask_actions.sum(dim=-1)         # [B,S]
        valid_dims = (num_valid_actions * original_action_dim).sum(dim=-1)  # [B]
        bpd_joint = -logp_joint / (valid_dims.clamp_min(1) * math.log(2.0))
        out["bpd_joint_mean"] = bpd_joint
        out["logp_joint_mean"] = logp_joint

        # 2) Entropy
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        std_use = std[..., :original_action_dim]
        H_elem = (c0 + torch.log(std_use)) * mask_elem[..., :original_action_dim]
        H_joint = H_elem.sum(dim=(-1,-2,-3,-4))  # [B]
        out["H_joint_mean"] = H_joint

        # 3) Calibration
        err = (x_next_f[..., :7] - mean[..., :7]) / (std[..., :7] + 1e-12)
        err = err[..., :self.config.n_action_steps, :]
        with torch.autocast('cuda', enabled=False):
            err = err.to(torch.float64)
            m = mask_elem[..., :7].expand_as(err).to(torch.float64)   # 显式扩成与 err 同形状

            denom = m.sum(dim=(1,2,3,4)).clamp_min(1)
            z2 = (err.pow(2) * m).sum(dim=(1,2,3,4)) / denom
            z4 = (err.pow(4) * m).sum(dim=(1,2,3,4)) / denom

        out["z_m2"], out["z_m4"] = z2, z4

        # 4) K-decomp
        out["logp_k_first"] = torch.nanmean(logp_step[..., 0], dim=1)
        out["logp_k_last"]  = torch.nanmean(logp_step[..., -1], dim=1)

        # 5) Masks / stability
        out["frac_valid_actions"] = mask_actions.float().mean(dim=(1,2))

        return out


    def compute_values(
        self,
        images,             # [B,S,3,H,W]
        img_masks,          # [B,S,1]
        lang_tokens,        # [B,S,L]
        lang_masks,         # [B,S,L]
        state,              # [B,S,1,Ds]
        x_t,                # [B,S,K,50,Da]
        t,                  # [B,S,K]
        x_next,             # [B,S,K,50,Da]
        finish_step,        # [B]
    ):
        """
        返回:
        logp_step  : [B,S,K]   —— 每个外层步、每个流步的 log-prob（已mask）
        logp_outer : [B,S]     —— 每个外层步联合（对K求和，已mask）
        logp_joint : [B]       —— 整条有效轨迹联合（对S再求和，已mask）
        """
        # ---------- 基本维度 ----------
        device = lang_tokens.device
        B, S, K, CH, D = x_t.shape  # CH=50, D=动作维
        assert t.shape == (B, S, K)
        assert x_next.shape == (B, S, K, CH, D)
        dt = -1.0 / float(K)                       # 按你的设定：K个等步，t: 1→0
        sqrt_abs_dt = math.sqrt(-dt)
        dtype = torch.float32                      # 计算用 fp32 更稳

        # ---------- 构建 prefix KV（按 [B*S] 批量） ----------
        BS = B * S
        # 展平外层时间 S 维
        def _flat(x, keep_tail=0):
            # 把 [B,S,...] → [B*S,...]
            if keep_tail == 0:
                return x.reshape(BS, *x.shape[2:])
            elif keep_tail == 1:
                return x.reshape(BS, x.shape[-1])
            else:
                raise ValueError

        images_flat    = _flat(images)
        img_masks_flat = _flat(img_masks)
        lang_tokens_f  = _flat(lang_tokens)
        lang_masks_f   = _flat(lang_masks)
        state_flat     = _flat(state) if state is not None else None

        x_t_flat    = x_t.reshape(BS * K, CH, D).to(device=device, dtype=dtype)
        x_next_f    = x_next.to(device=device, dtype=dtype) # 保持原始形状，后面用
        t_flat      = t.reshape(BS * K).to(device=device, dtype=dtype)
        # breakpoint()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            [images_flat], [img_masks_flat.squeeze(-1)], lang_tokens_f, lang_masks_f, state=state_flat
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        outputs_embeds, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        state_vector = outputs_embeds[0][:, -1, :]
        value = self.value_head(state_vector).squeeze(-1)
        return value