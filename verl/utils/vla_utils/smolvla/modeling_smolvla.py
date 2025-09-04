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
    _no_split_modules = ['LlamaDecoderLayer']

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
        self._tp_plan = {}
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
    
    # @classmethod
    # @restore_default_torch_dtype
    # def from_pretrained(
    #     cls: Type[SpecificPreTrainedModelType],
    #     pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    #     *model_args,
    #     config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    #     cache_dir: Optional[Union[str, os.PathLike]] = None,
    #     ignore_mismatched_sizes: bool = False,
    #     force_download: bool = False,
    #     local_files_only: bool = False,
    #     token: Optional[Union[str, bool]] = None,
    #     revision: str = "main",
    #     use_safetensors: Optional[bool] = None,
    #     weights_only: bool = True,
    #     **kwargs,
    # ) -> SpecificPreTrainedModelType:
    #     r"""
    #     Copy from transformers/modeling_utils.py
    #     """
    #     state_dict = kwargs.pop("state_dict", None)
    #     from_tf = kwargs.pop("from_tf", False)
    #     from_flax = kwargs.pop("from_flax", False)
    #     proxies = kwargs.pop("proxies", None)
    #     output_loading_info = kwargs.pop("output_loading_info", False)
    #     use_auth_token = kwargs.pop("use_auth_token", None)
    #     from_pipeline = kwargs.pop("_from_pipeline", None)
    #     from_auto_class = kwargs.pop("_from_auto", False)
    #     torch_dtype = kwargs.pop("torch_dtype", None)
    #     device_map = kwargs.pop("device_map", None)
    #     max_memory = kwargs.pop("max_memory", None)
    #     offload_folder = kwargs.pop("offload_folder", None)
    #     offload_state_dict = kwargs.pop("offload_state_dict", False)
    #     offload_buffers = kwargs.pop("offload_buffers", False)
    #     load_in_8bit = kwargs.pop("load_in_8bit", False)
    #     load_in_4bit = kwargs.pop("load_in_4bit", False)
    #     quantization_config = kwargs.pop("quantization_config", None)
    #     subfolder = kwargs.pop("subfolder", "")
    #     commit_hash = kwargs.pop("_commit_hash", None)
    #     variant = kwargs.pop("variant", None)
    #     adapter_kwargs = kwargs.pop("adapter_kwargs", {})
    #     adapter_name = kwargs.pop("adapter_name", "default")
    #     use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
    #     generation_config = kwargs.pop("generation_config", None)
    #     gguf_file = kwargs.pop("gguf_file", None)
    #     tp_plan = kwargs.pop("tp_plan", None)
    #     key_mapping = kwargs.pop("key_mapping", None)
    #     # Not used anymore -- remove them from the kwargs
    #     _ = kwargs.pop("resume_download", None)
    #     _ = kwargs.pop("trust_remote_code", None)
    #     _ = kwargs.pop("mirror", None)
    #     _ = kwargs.pop("_fast_init", True)
    #     _ = kwargs.pop("low_cpu_mem_usage", None)

    #     if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
    #         raise ValueError(
    #             "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
    #         )

    #     if tp_plan is not None and tp_plan != "auto":
    #         # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
    #         raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
    #     if tp_plan is not None and device_map is not None:
    #         raise ValueError(
    #             "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
    #         )

    #     # If torchrun was used, make sure to TP by default. This way people don't need to change tp or device map
    #     if device_map == "auto" and tp_plan is None and int(os.environ.get("WORLD_SIZE", 0)):
    #         tp_plan = "auto"  # device_map = "auto" in torchrun equivalent to TP plan = AUTO!
    #         device_map = None

    #     # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
    #     # `device_map` pointing to the correct device
    #     device_mesh = None
    #     if tp_plan is not None:
    #         if not is_torch_greater_or_equal("2.5"):
    #             raise EnvironmentError("tensor parallel is only supported for `torch>=2.5`.")

    #         # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
    #         device_type = torch._C._get_accelerator().type

    #         if not torch.distributed.is_initialized():
    #             try:
    #                 rank = int(os.environ["RANK"])
    #                 world_size = int(os.environ["WORLD_SIZE"])
    #                 if device_type == "cuda":
    #                     torch.distributed.init_process_group(
    #                         "nccl", rank=rank, world_size=world_size, init_method="env://"
    #                     )
    #                     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    #                 elif device_type == "cpu":
    #                     cpu_backend = "ccl" if int(os.environ.get("CCL_WORKER_COUNT", 0)) else "gloo"
    #                     torch.distributed.init_process_group(cpu_backend, rank=rank, world_size=world_size)

    #             except Exception as e:
    #                 raise EnvironmentError(
    #                     "We tried to initialize torch.distributed for you, but it failed, make"
    #                     "sure you init torch distributed in your script to use `tp_plan='auto'`"
    #                 ) from e

    #         # Get device with index assuming equal number of devices per host
    #         index = None if device_type == "cpu" else torch.cuda.current_device()
    #         tp_device = torch.device(device_type, index)

    #         if index is not None and index > 0:
    #             import sys

    #             sys.stdout = open(os.devnull, "w")
    #             sys.stderr = open(os.devnull, "w")
    #         # This is the easiest way to dispatch to the current process device
    #         device_map = tp_device
    #         # Assuming sharding the model onto the world
    #         world_size = torch.distributed.get_world_size()
    #         device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))

    #     if use_auth_token is not None:
    #         warnings.warn(
    #             "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
    #             FutureWarning,
    #         )
    #         if token is not None:
    #             raise ValueError(
    #                 "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
    #             )
    #         token = use_auth_token

    #     if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
    #         adapter_kwargs["token"] = token

    #     if use_safetensors is None and not is_safetensors_available():
    #         use_safetensors = False

    #     if gguf_file is not None and not is_accelerate_available():
    #         raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

    #     if commit_hash is None:
    #         if not isinstance(config, PretrainedConfig):
    #             # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
    #             resolved_config_file = cached_file(
    #                 pretrained_model_name_or_path,
    #                 CONFIG_NAME,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 token=token,
    #                 revision=revision,
    #                 subfolder=subfolder,
    #                 _raise_exceptions_for_gated_repo=False,
    #                 _raise_exceptions_for_missing_entries=False,
    #                 _raise_exceptions_for_connection_errors=False,
    #             )
    #             commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
    #         else:
    #             commit_hash = getattr(config, "_commit_hash", None)

    #     if is_peft_available():
    #         _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

    #         if _adapter_model_path is None:
    #             _adapter_model_path = find_adapter_config_file(
    #                 pretrained_model_name_or_path,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 _commit_hash=commit_hash,
    #                 **adapter_kwargs,
    #             )
    #         if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
    #             with open(_adapter_model_path, "r", encoding="utf-8") as f:
    #                 _adapter_model_path = pretrained_model_name_or_path
    #                 pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
    #     else:
    #         _adapter_model_path = None

    #     # change device_map into a map if we passed an int, a str or a torch.device
    #     if isinstance(device_map, torch.device):
    #         device_map = {"": device_map}
    #     elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
    #         try:
    #             device_map = {"": torch.device(device_map)}
    #         except RuntimeError:
    #             raise ValueError(
    #                 "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
    #                 f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
    #             )
    #     elif isinstance(device_map, int):
    #         if device_map < 0:
    #             raise ValueError(
    #                 "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
    #             )
    #         else:
    #             device_map = {"": device_map}

    #     if device_map is not None:
    #         if is_deepspeed_zero3_enabled():
    #             raise ValueError("DeepSpeed Zero-3 is not compatible with passing a `device_map`.")
    #         if not is_accelerate_available():
    #             raise ValueError(
    #                 "Using a `device_map` or `tp_plan` requires `accelerate`. You can install it with `pip install accelerate`"
    #             )

    #     # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
    #     if load_in_4bit or load_in_8bit:
    #         if quantization_config is not None:
    #             raise ValueError(
    #                 "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
    #                 "`quantization_config` argument at the same time."
    #             )

    #         # preparing BitsAndBytesConfig from kwargs
    #         config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
    #         config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
    #         quantization_config, kwargs = BitsAndBytesConfig.from_dict(
    #             config_dict=config_dict, return_unused_kwargs=True, **kwargs
    #         )
    #         logger.warning(
    #             "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
    #             "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
    #         )

    #     from_pt = not (from_tf | from_flax)

    #     user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    #     if from_pipeline is not None:
    #         user_agent["using_pipeline"] = from_pipeline

    #     if is_offline_mode() and not local_files_only:
    #         logger.info("Offline mode: forcing local_files_only=True")
    #         local_files_only = True

    #     # Load config if we don't provide a configuration
    #     if not isinstance(config, PretrainedConfig):
    #         config_path = config if config is not None else pretrained_model_name_or_path
    #         config, model_kwargs = cls.config_class.from_pretrained(
    #             config_path,
    #             cache_dir=cache_dir,
    #             return_unused_kwargs=True,
    #             force_download=force_download,
    #             proxies=proxies,
    #             local_files_only=local_files_only,
    #             token=token,
    #             revision=revision,
    #             subfolder=subfolder,
    #             gguf_file=gguf_file,
    #             _from_auto=from_auto_class,
    #             _from_pipeline=from_pipeline,
    #             **kwargs,
    #         )
    #         if "gguf_file" in model_kwargs:
    #             model_kwargs.pop("gguf_file")
    #     else:
    #         # In case one passes a config to `from_pretrained` + "attn_implementation"
    #         # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
    #         # Please see: https://github.com/huggingface/transformers/issues/28038

    #         # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
    #         # we pop attn_implementation from the kwargs but this handles the case where users
    #         # passes manually the config to `from_pretrained`.
    #         config = copy.deepcopy(config)

    #         kwarg_attn_imp = kwargs.pop("attn_implementation", None)
    #         if kwarg_attn_imp is not None:
    #             config._attn_implementation = kwarg_attn_imp

    #         model_kwargs = kwargs

    #     pre_quantized = hasattr(config, "quantization_config")
    #     if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
    #         pre_quantized = False

    #     if pre_quantized or quantization_config is not None:
    #         if pre_quantized:
    #             config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
    #                 config.quantization_config, quantization_config
    #             )
    #         else:
    #             config.quantization_config = quantization_config

    #         hf_quantizer = AutoHfQuantizer.from_config(
    #             config.quantization_config,
    #             pre_quantized=pre_quantized,
    #         )
    #     else:
    #         hf_quantizer = None

    #     if hf_quantizer is not None:
    #         hf_quantizer.validate_environment(
    #             torch_dtype=torch_dtype,
    #             from_tf=from_tf,
    #             from_flax=from_flax,
    #             device_map=device_map,
    #             weights_only=weights_only,
    #         )
    #         torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
    #         device_map = hf_quantizer.update_device_map(device_map)
    #         config = hf_quantizer.update_tp_plan(config)

    #         # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
    #         if hasattr(hf_quantizer.quantization_config.quant_method, "value"):
    #             user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
    #         else:
    #             user_agent["quant"] = hf_quantizer.quantization_config.quant_method

    #     if gguf_file is not None and hf_quantizer is not None:
    #         raise ValueError(
    #             "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
    #         )

    #     if (
    #         gguf_file
    #         and device_map is not None
    #         and ((isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map)
    #     ):
    #         raise RuntimeError(
    #             "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
    #             "loaded from GGUF files."
    #         )

    #     checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
    #         pretrained_model_name_or_path=pretrained_model_name_or_path,
    #         subfolder=subfolder,
    #         variant=variant,
    #         gguf_file=gguf_file,
    #         from_tf=from_tf,
    #         from_flax=from_flax,
    #         use_safetensors=use_safetensors,
    #         cache_dir=cache_dir,
    #         force_download=force_download,
    #         proxies=proxies,
    #         local_files_only=local_files_only,
    #         token=token,
    #         user_agent=user_agent,
    #         revision=revision,
    #         commit_hash=commit_hash,
    #     )

    #     is_sharded = sharded_metadata is not None
    #     is_quantized = hf_quantizer is not None
    #     is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

    #     if (
    #         is_safetensors_available()
    #         and is_from_file
    #         and not is_sharded
    #         and checkpoint_files[0].endswith(".safetensors")
    #     ):
    #         with safe_open(checkpoint_files[0], framework="pt") as f:
    #             metadata = f.metadata()

    #         if metadata is None:
    #             # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
    #             pass
    #         elif metadata.get("format") == "pt":
    #             pass
    #         elif metadata.get("format") == "tf":
    #             from_tf = True
    #             logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
    #         elif metadata.get("format") == "flax":
    #             from_flax = True
    #             logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
    #         elif metadata.get("format") == "mlx":
    #             # This is a mlx file, we assume weights are compatible with pt
    #             pass
    #         else:
    #             raise ValueError(
    #                 f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
    #             )

    #     from_pt = not (from_tf | from_flax)

    #     if from_pt:
    #         if gguf_file:
    #             from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

    #             # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
    #             # passed directly as a kwarg from now on
    #             with torch.device("meta"):
    #                 dummy_model = cls(config)
    #             state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
    #                 "tensors"
    #             ]

    #         # Find the correct dtype based on current state
    #         config, torch_dtype, dtype_orig = _get_torch_dtype(
    #             cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
    #         )

    #     config.name_or_path = pretrained_model_name_or_path

    #     # Instantiate model.
    #     model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)

    #     config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
    #     if not getattr(config, "_attn_implementation_autoset", False):
    #         config = cls._autoset_attn_implementation(
    #             config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
    #         )

    #     with ContextManagers(model_init_context):
    #         # Let's make sure we don't run the init function of buffer modules
    #         model = cls(config, *model_args, **model_kwargs)

    #     # Make sure to tie the weights correctly
    #     model.tie_weights()

    #     # Last check for tp
    #     if device_mesh is not None and not model.supports_tp_plan:
    #         if config.base_model_tp_plan is None and config.get_text_config().base_model_tp_plan is None:
    #             raise NotImplementedError("This model does not have a tensor parallel plan.")

    #     # make sure we use the model's config since the __init__ call might have copied it
    #     config = model.config

    #     # Find fp32 modules if needed
    #     keep_in_fp32_regex = None
    #     # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
    #     # in case of force loading a model that should stay bf16 in fp16 (which includes a few quantizers as this is a pre-processing
    #     # step for e.g. bitsandbytes). See https://github.com/huggingface/transformers/issues/20287 for details.
    #     if model._keep_in_fp32_modules is not None and (
    #         torch_dtype == torch.float16 or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
    #     ):
    #         # We need to match exact layers, so we add either `.` on each side, or start/end of string
    #         keep_in_fp32_regex = re.compile(
    #             "|".join([rf"((^|\.){module}($|\.))" for module in model._keep_in_fp32_modules])
    #         )

    #     if hf_quantizer is not None:
    #         hf_quantizer.preprocess_model(
    #             model=model, device_map=device_map, keep_in_fp32_modules=model._keep_in_fp32_modules, config=config
    #         )
    #         # We store the original dtype for quantized models as we cannot easily retrieve it
    #         # once the weights have been quantized
    #         # Note that once you have loaded a quantized model, you can't change its dtype so this will
    #         # remain a single source of truth
    #         config._pre_quantization_dtype = torch_dtype if torch_dtype is not None else torch.get_default_dtype()

    #     # Prepare the full device map
    #     if device_map is not None:
    #         device_map = _get_device_map(model, device_map, max_memory, hf_quantizer, torch_dtype, keep_in_fp32_regex)

    #     # Finalize model weight initialization
    #     if from_tf:
    #         model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
    #     elif from_flax:
    #         model = cls._load_from_flax(model, checkpoint_files)
    #     elif from_pt:
    #         # restore default dtype
    #         if dtype_orig is not None:
    #             torch.set_default_dtype(dtype_orig)

    #         (
    #             model,
    #             missing_keys,
    #             unexpected_keys,
    #             mismatched_keys,
    #             offload_index,
    #             error_msgs,
    #         ) = cls._load_pretrained_model(
    #             model,
    #             state_dict,
    #             checkpoint_files,
    #             pretrained_model_name_or_path,
    #             ignore_mismatched_sizes=ignore_mismatched_sizes,
    #             sharded_metadata=sharded_metadata,
    #             device_map=device_map,
    #             disk_offload_folder=offload_folder,
    #             offload_state_dict=offload_state_dict,
    #             dtype=torch_dtype,
    #             hf_quantizer=hf_quantizer,
    #             keep_in_fp32_regex=keep_in_fp32_regex,
    #             device_mesh=device_mesh,
    #             key_mapping=key_mapping,
    #             weights_only=weights_only,
    #         )

    #     # make sure token embedding weights are still tied if needed
    #     model.tie_weights()

    #     # Set model in evaluation mode to deactivate DropOut modules by default
    #     model.eval()

    #     # If it is a model with generation capabilities, attempt to load the generation config
    #     if model.can_generate() and generation_config is not None:
    #         logger.info("The user-defined `generation_config` will be used to override the default generation config.")
    #         model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
    #     elif model.can_generate() and pretrained_model_name_or_path is not None:
    #         try:
    #             model.generation_config = GenerationConfig.from_pretrained(
    #                 pretrained_model_name_or_path,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 token=token,
    #                 revision=revision,
    #                 subfolder=subfolder,
    #                 _from_auto=from_auto_class,
    #                 _from_pipeline=from_pipeline,
    #                 **kwargs,
    #             )
    #         except OSError:
    #             logger.info(
    #                 "Generation config file not found, using a generation config created from the model config."
    #             )
    #             pass

    #     # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
    #     # harm performances)
    #     if device_map is not None and device_mesh is None:
    #         device_map_kwargs = {
    #             "device_map": device_map,
    #             "offload_dir": offload_folder,
    #             "offload_index": offload_index,
    #             "offload_buffers": offload_buffers,
    #         }
    #         if "skip_keys" in inspect.signature(dispatch_model).parameters:
    #             device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
    #         # For HQQ method we force-set the hooks for single GPU envs
    #         if (
    #             "force_hooks" in inspect.signature(dispatch_model).parameters
    #             and hf_quantizer is not None
    #             and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
    #         ):
    #             device_map_kwargs["force_hooks"] = True
    #         if (
    #             hf_quantizer is not None
    #             and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
    #             and isinstance(device_map, dict)
    #             and ("cpu" in device_map.values() or "disk" in device_map.values())
    #         ):
    #             device_map_kwargs["offload_buffers"] = True

    #         if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
    #             dispatch_model(model, **device_map_kwargs)

    #     if hf_quantizer is not None:
    #         hf_quantizer.postprocess_model(model, config=config)
    #         model.hf_quantizer = hf_quantizer

    #     if _adapter_model_path is not None:
    #         model.load_adapter(
    #             _adapter_model_path,
    #             adapter_name=adapter_name,
    #             token=token,
    #             adapter_kwargs=adapter_kwargs,
    #         )

    #     if output_loading_info:
    #         if from_pt:
    #             loading_info = {
    #                 "missing_keys": missing_keys,
    #                 "unexpected_keys": unexpected_keys,
    #                 "mismatched_keys": mismatched_keys,
    #                 "error_msgs": error_msgs,
    #             }
    #         elif from_flax:
    #             loading_info = None
    #         return model, loading_info

    #     return model
    
    # @classmethod
    # def _load_pretrained_model(
    #     cls,
    #     model: "PreTrainedModel",
    #     state_dict: Optional[Dict],
    #     checkpoint_files: Optional[List[str]],
    #     pretrained_model_name_or_path: Optional[str],
    #     ignore_mismatched_sizes: bool = False,
    #     sharded_metadata: Optional[Dict] = None,
    #     device_map: Optional[Dict] = None,
    #     disk_offload_folder: Optional[str] = None,
    #     offload_state_dict: Optional[bool] = None,
    #     dtype: Optional[torch.dtype] = None,
    #     hf_quantizer: Optional[HfQuantizer] = None,
    #     keep_in_fp32_regex: Optional[re.Pattern] = None,
    #     device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    #     key_mapping: Optional[Dict[str, str]] = None,
    #     weights_only: bool = True,
    # ):
    #     # Useful flags
    #     is_quantized = hf_quantizer is not None
    #     is_hqq = is_quantized and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
    #     is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in [
    #         QuantizationMethod.HQQ,
    #         QuantizationMethod.BITS_AND_BYTES,
    #     ]

    #     # Get all the keys of the state dicts that we have to initialize the model
    #     if sharded_metadata is not None:
    #         original_checkpoint_keys = sharded_metadata["all_checkpoint_keys"]
    #     elif state_dict is not None:
    #         original_checkpoint_keys = list(state_dict.keys())
    #     else:
    #         original_checkpoint_keys = list(
    #             load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
    #         )
    #         norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    #         original_checkpoint_keys = [k for k in original_checkpoint_keys if not k.startswith(norm_keys)]
    #         original_checkpoint_keys_copy = original_checkpoint_keys.copy()
    #         new_original_checkpoint_keys = []
    #         for key in original_checkpoint_keys:
    #             new_name = key.replace('model._orig_mod.', 'model.')
    #             new_original_checkpoint_keys.append(new_name)
    #         original_checkpoint_keys = new_original_checkpoint_keys

       

    #     # Check if we are in a special state, i.e. loading from a state dict coming from a different architecture
    #     prefix = model.base_model_prefix
    #     _prefix = f"{prefix}."
    #     has_prefix_module = any(s.startswith(prefix) for s in original_checkpoint_keys) if len(prefix) > 0 else False
    #     expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
    #     loading_task_model_from_base_state_dict = not has_prefix_module and expects_prefix_module
    #     loading_base_model_from_task_state_dict = has_prefix_module and not expects_prefix_module

    #     # Find the key names that the model expects from the serialized keys
    #     key_renaming_mapping = model._get_key_renaming_mapping(
    #         original_checkpoint_keys_copy,
    #         key_mapping,
    #         loading_base_model_from_task_state_dict,
    #         loading_task_model_from_base_state_dict,
    #     )
    #     key_renaming_mapping = {k: v.replace('model._orig_mod.', 'model.') for k, v in key_renaming_mapping.items()}
    #     checkpoint_keys = list(key_renaming_mapping.values())

    #     # Find missing and unexpected keys from the state dict
    #     missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(
    #         cls,
    #         model,
    #         original_checkpoint_keys,
    #         checkpoint_keys,
    #         loading_base_model_from_task_state_dict,
    #         hf_quantizer,
    #         device_map,
    #     )
    #     # Find all the keys with shape mismatch (if we ignore the mismatch, the weights need to be newly initialized the
    #     # same way as missing keys)
    #     mismatched_keys, mismatched_shapes = _find_mismatched_keys(
    #         model,
    #         state_dict,
    #         checkpoint_files,
    #         ignore_mismatched_sizes,
    #         key_renaming_mapping,
    #         is_quantized,
    #         weights_only,
    #     )

    #     # We need to update both the mapping and the list of checkpoint keys to remove the mismatched ones
    #     key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys}
    #     checkpoint_keys = list(key_renaming_mapping.values())

    #     # Move missing (and potentially mismatched) keys back to cpu from meta device (because they won't be moved when
    #     # loading the weights as they are not in the loaded state dict)
    #     model._move_missing_keys_from_meta_to_cpu(missing_keys + mismatched_keys, unexpected_keys, dtype, hf_quantizer)

    #     # correctly initialize the missing (and potentially mismatched) keys
    #     model._initialize_missing_keys(checkpoint_keys, ignore_mismatched_sizes, is_quantized)

    #     # Set some modules to fp32 if needed
    #     if keep_in_fp32_regex is not None:
    #         for name, param in model.named_parameters():
    #             if keep_in_fp32_regex.search(name):
    #                 # param = param.to(torch.float32) does not work here as only in the local scope.
    #                 param.data = param.data.to(torch.float32)

    #     # Make sure we are able to load base models as well as derived models (specific task models, with heads)
    #     model_to_load = model
    #     # In this case, we load a ForTaskModel with keys from a BaseModel -> only load keys to the BaseModel
    #     if loading_task_model_from_base_state_dict:
    #         model_to_load = getattr(model, prefix)
    #         # Here we need to remove the prefix we added to correctly find missing/unexpected keys, as we will load
    #         # in the submodule
    #         key_renaming_mapping = {k: v[len(_prefix) :] for k, v in key_renaming_mapping.items()}
    #         checkpoint_keys = list(key_renaming_mapping.values())
    #         # We need to update the device map as well
    #         if device_map is not None:
    #             device_map = {k[len(_prefix) :] if k.startswith(_prefix) else k: v for k, v in device_map.items()}
    #         # small sanity check: the base model should not contain task-specific head keys
    #         task_specific_expected_keys = [s for s in model.state_dict().keys() if not s.startswith(_prefix)]
    #         base_model_expected_keys = list(model_to_load.state_dict().keys())
    #         if any(
    #             key in task_specific_expected_keys and key not in base_model_expected_keys for key in checkpoint_keys
    #         ):
    #             raise ValueError(
    #                 "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
    #                 "properly saved?"
    #             )

    #     # Get reverse key mapping
    #     reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

    #     is_offloaded_safetensors = False
    #     # This offload index if for params explicitly on the "disk" in the device_map
    #     disk_offload_index = None
    #     disk_only_shard_files = []
    #     # Prepare parameters offloading if needed
    #     if device_map is not None and "disk" in device_map.values():
    #         if offload_state_dict is None:
    #             offload_state_dict = True
    #         if disk_offload_folder is not None:
    #             os.makedirs(disk_offload_folder, exist_ok=True)
    #         is_offloaded_safetensors = checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors")
    #         if disk_offload_folder is None and not is_offloaded_safetensors:
    #             raise ValueError(
    #                 "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
    #                 " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
    #                 " offers the weights in this format."
    #             )
    #         if is_offloaded_safetensors:
    #             param_device_map = expand_device_map(device_map, checkpoint_keys)
    #             str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
    #             if sharded_metadata is None:
    #                 weight_map = dict.fromkeys(checkpoint_keys, checkpoint_files[0])
    #             else:
    #                 folder = os.path.sep.join(checkpoint_files[0].split(os.path.sep)[:-1])
    #                 # Fix the weight map keys according to the key mapping
    #                 weight_map = {
    #                     key_renaming_mapping[k]: v
    #                     for k, v in sharded_metadata["weight_map"].items()
    #                     if k in key_renaming_mapping
    #                 }
    #                 weight_map = {k: os.path.join(folder, v) for k, v in weight_map.items()}
    #                 # Find potential checkpoints containing only offloaded weights
    #                 disk_only_shard_files = get_disk_only_shard_files(device_map, weight_map)
    #             disk_offload_index = {
    #                 name: {
    #                     "safetensors_file": file,
    #                     "weight_name": reverse_key_renaming_mapping[name],
    #                     "dtype": str_dtype,
    #                 }
    #                 for name, file in weight_map.items()
    #                 if param_device_map[name] == "disk"
    #             }
    #         else:
    #             disk_offload_index = {}

    #     # This offload index if for params that are supposed to be on the "cpu", either with or without a device_map
    #     # It allows to load parameters one-by-one from the state dict, avoiding a memory peak of 2 x state_dict_size,
    #     # i.e. 1x to load it, and 1x to copy it to model
    #     cpu_offload_folder = None
    #     cpu_offload_index = None
    #     if offload_state_dict:
    #         cpu_offload_folder = tempfile.mkdtemp()
    #         cpu_offload_index = {}

    #     # For nice tqdm bars
    #     if checkpoint_files is not None and len(checkpoint_files) > 1:
    #         checkpoint_files = logging.tqdm(checkpoint_files, desc="Loading checkpoint shards")
    #     # To be able to iterate, even if we don't use it if the state_dict is already provided
    #     elif state_dict is not None:
    #         checkpoint_files = [""]

    #     # Compute expected model keys
    #     expected_keys = list(model_to_load.state_dict().keys())
    #     if hf_quantizer is not None:
    #         expected_keys = hf_quantizer.update_expected_keys(model_to_load, expected_keys, checkpoint_keys)

    #     # Warmup cuda to load the weights much faster on devices
    #     if device_map is not None and not is_hqq:
    #         expanded_device_map = expand_device_map(device_map, expected_keys)
    #         caching_allocator_warmup(model_to_load, expanded_device_map, factor=2 if hf_quantizer is None else 4)

    #     error_msgs = []
    #     # Iterate on all the shards to load the weights
    #     for shard_file in checkpoint_files:
    #         # Skip the load for shards that only contain disk-offloaded weights
    #         if shard_file in disk_only_shard_files:
    #             continue

    #         map_location = "cpu"
    #         if (
    #             shard_file.endswith(".safetensors")
    #             and not is_hqq_or_bnb
    #             and not (is_deepspeed_zero3_enabled() and not is_quantized)
    #         ):
    #             map_location = "meta"
    #         elif (
    #             device_map is not None
    #             and hf_quantizer is not None
    #             and hf_quantizer.quantization_config.quant_method == QuantizationMethod.TORCHAO
    #             and (
    #                 hf_quantizer.quantization_config.quant_type in ["int4_weight_only", "autoquant"]
    #                 or isinstance(hf_quantizer.quantization_config.quant_type, Int4WeightOnlyConfig)
    #             )
    #         ):
    #             map_location = torch.device([d for d in device_map.values() if d not in ["cpu", "disk"]][0])

    #         # If shard_file is "", we use the existing state_dict instead of loading it
    #         if shard_file != "":
    #             state_dict = load_state_dict(
    #                 shard_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only
    #             )
    #             # checkpoint_keys_mapping = "model._orig_mod.//model."
    #             # state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)
    #             # norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    #             # state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    #         # Fix the key names
    #         state_dict = {key_renaming_mapping[k]: v for k, v in state_dict.items() if k in key_renaming_mapping}

    #         if is_deepspeed_zero3_enabled() and not is_quantized:
    #             error_msgs += _load_state_dict_into_zero3_model(model_to_load, state_dict)
    #         # Skip it with fsdp on ranks other than 0
    #         elif not (is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized):
    #             disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
    #                 model_to_load,
    #                 state_dict,
    #                 shard_file,
    #                 expected_keys,
    #                 reverse_key_renaming_mapping,
    #                 device_map=device_map,
    #                 disk_offload_folder=disk_offload_folder,
    #                 disk_offload_index=disk_offload_index,
    #                 cpu_offload_folder=cpu_offload_folder,
    #                 cpu_offload_index=cpu_offload_index,
    #                 hf_quantizer=hf_quantizer,
    #                 is_safetensors=is_offloaded_safetensors,
    #                 keep_in_fp32_regex=keep_in_fp32_regex,
    #                 unexpected_keys=unexpected_keys,
    #                 device_mesh=device_mesh,
    #             )

    #         # force memory release if loading multiple shards, to avoid having 2 state dicts in memory in next loop
    #         del state_dict

    #     # Adjust offloaded weights name and save if needed
    #     if disk_offload_index is not None and len(disk_offload_index) > 0:
    #         if loading_task_model_from_base_state_dict:
    #             # We need to add the prefix of the base model
    #             prefix = cls.base_model_prefix
    #             if not is_offloaded_safetensors:
    #                 for weight_name in disk_offload_index:
    #                     shutil.move(
    #                         os.path.join(disk_offload_folder, f"{weight_name}.dat"),
    #                         os.path.join(disk_offload_folder, f"{prefix}.{weight_name}.dat"),
    #                     )
    #             disk_offload_index = {f"{prefix}.{key}": value for key, value in disk_offload_index.items()}
    #         if not is_offloaded_safetensors:
    #             save_offload_index(disk_offload_index, disk_offload_folder)
    #             disk_offload_index = None

    #     # one-at-a-time param loading for the cpu offloaded params
    #     if offload_state_dict:
    #         # Load back temporarily offloaded state dict
    #         load_offloaded_weights(model_to_load, cpu_offload_index, cpu_offload_folder)
    #         shutil.rmtree(cpu_offload_folder)

    #     if hf_quantizer is not None:
    #         missing_keys = hf_quantizer.update_missing_keys_after_loading(model_to_load, missing_keys, prefix)

    #     # Post-processing for tensor parallelism
    #     if device_mesh is not None:
    #         # When using TP, the device map is a single device for all parameters
    #         tp_device = list(device_map.values())[0]
    #         # This is needed for the RotaryEmbedding, which was not initialized on the correct device as it is
    #         # not part of the state_dict (persistent=False)
    #         for buffer in model.buffers():
    #             if buffer.device != tp_device:
    #                 buffer.data = buffer.to(tp_device)

    #         # In this case, the top-most task module weights were not moved to device and parallelized as they
    #         # were not part of the loaded weights: do it now
    #         if loading_task_model_from_base_state_dict:
    #             parameters_to_initialize = {
    #                 name: param for name, param in model.named_parameters() if not name.startswith(prefix)
    #             }
    #             for name, param in parameters_to_initialize.items():
    #                 # First move data to correct
    #                 to_contiguous, casting_dtype = _infer_parameter_dtype(model, name, param, keep_in_fp32_regex)
    #                 shard_and_distribute_module(
    #                     model,
    #                     param.to(tp_device),
    #                     param,
    #                     name,
    #                     casting_dtype,
    #                     to_contiguous,
    #                     os.environ["RANK"],
    #                     device_mesh,
    #                 )

    #     # All potential warnings/infos
    #     if len(error_msgs) > 0:
    #         error_msg = "\n\t".join(error_msgs)
    #         if "size mismatch" in error_msg:
    #             error_msg += (
    #                 "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
    #             )
    #         raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
    #     if len(unexpected_keys) > 0:
    #         archs = [] if model.config.architectures is None else model.config.architectures
    #         warner = logger.warning if model.__class__.__name__ in archs else logger.info
    #         warner(
    #             f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
    #             f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
    #             f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
    #             " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
    #             " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
    #             f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
    #             " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
    #         )
    #     else:
    #         logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    #     if len(missing_keys) > 0:
    #         logger.warning(
    #             f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
    #             f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
    #             " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
    #         )
    #     elif len(mismatched_keys) == 0:
    #         logger.info(
    #             f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
    #             f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
    #             f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
    #             " training."
    #         )
    #     if len(mismatched_keys) > 0:
    #         mismatched_warning = "\n".join(
    #             [
    #                 f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
    #                 for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes)
    #             ]
    #         )
    #         logger.warning(
    #             f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
    #             f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
    #             f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
    #             " to use it for predictions and inference."
    #         )

    #     return model, missing_keys, unexpected_keys, mismatched_keys, disk_offload_index, error_msgs
    
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
             ent_step, 
             ent_outer, 
             ent_joint) = self.model.recompute_logprob(images, img_masks, lang_tokens, lang_masks, state, x_t, t, x_next, finish_step)
            return_dict = {
                "logp_action": logp_action,
                "logp_step": logp_step, 
                "logp_outer": logp_outer, 
                "logp_joint": logp_joint,
                "ent_step": ent_step, 
                "ent_outer": ent_outer, 
                "ent_joint": ent_joint,
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

        actions, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
            batch,
            noise,
            use_sde=use_sde,
            return_logprob=return_logprob,
            recompute_log_prob=recompute_log_prob
        )
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

        actions, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
            batch,
            noise,
            use_sde=use_sde,
            return_logprob=return_logprob,
            recompute_log_prob=recompute_log_prob
        )
        return actions, lang_tokens, lang_masks, return_dict

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
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

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
        breakpoint()
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

        CH_idx = torch.arange(CH, device=device)[None, None, :]
        S_idx  = torch.arange(S,  device=device)[None, :, None]
        s_fin  = (finish_step.to(device) // CH).view(B, 1, 1)
        c_fin  = (finish_step.to(device) %  CH).view(B, 1, 1)

        mask_before = (S_idx <  s_fin).float()
        mask_equal  = (S_idx == s_fin).float() * (CH_idx <= c_fin).float()
        mask_actions = mask_before.expand(B, S, CH) + mask_equal
        mask_elem = mask_actions[:, :, None, :, None]

        # lp_elem = Normal(mean, std).log_prob(x_next_f.detach())
        lp_elem = (
            -((x_next_f.detach() - mean) ** 2) / (2 * ((std)**2))
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
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

        return logp_action, logp_step, logp_outer, logp_joint, ent_step, ent_outer, ent_joint