# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl_vla import DataProto
from verl_vla.trainer.ppo import core_algos
from verl_vla.workers.critic import BasePPOCritic
from verl_vla.utils.py_functional import append_to_dict
from verl_vla.utils.torch_functional import masked_mean

__all__ = ['DataParallelPPOCritic']


class RobDataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.critic_module(input_ids=micro_batch['input_ids'],
                                        attention_mask=micro_batch['attention_mask'],
                                        position_ids=micro_batch['position_ids'],
                                        use_cache=False)  # prevent model thinks we are generating
            values = output.logits
            values = values[:, -response_length - 1:-1]
            return values

    def _forward_micro_batch_smolvla(self, micro_batch):
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['action_tensor'].size(0)
        traj_len = micro_batch['action_tensor'].size(1)
        
        # breakpoint()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return_dict = self.critic_module.compute_values(micro_batch)
            return return_dict["values"]
        
    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        # select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        select_keys = ['observation.images.image', 'observation.images.image_is_pad',
                    #    'observation.images.wrist_image', 'observation.images.wrist_image_is_pad', 
                        'observation.state', 'observation.state_is_pad', 'action_tensor', 
                        "x_t", "t", "x_next",
                        "lang_tokens", "lang_masks", "finish_step",
                        "values", "returns"]
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        if self.config.model.vla == 'smolvla':
            select_keys = ['observation.images.image', 'observation.images.image_is_pad',
                        #    'observation.images.wrist_image', 'observation.images.wrist_image_is_pad', 
                           'observation.state', 'observation.state_is_pad', 'action_tensor', 
                           "x_t", "t", "x_next",
                           "lang_tokens", "lang_masks", "finish_step"]
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
            
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)
        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                if self.config.model.vla == 'smolvla':
                    values = self._forward_micro_batch_smolvla(micro_batch)
                else:
                    raise
            values_lst.append(values)
        values = torch.stack(values_lst, dim=0)
        # responses = data.batch['responses']
        # attention_mask = data.batch['attention_mask']
        # response_length = responses.size(1)
        # values = values * attention_mask[:, -response_length - 1:-1]
        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        dataloader = self._make_minibatch_iterator(data)

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            micro_batches = data.batch.split(self.config.ppo_micro_batch_size)
            self.critic_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # critic device is cpu when using offload

                returns = data['returns']
                values = data['values']

                B, S, K, CH, D = data['x_t'].shape
                response_length = S * CH * 7 
                finish_step = data['finish_step'] * 7
                steps = torch.arange(response_length, device=data['x_t'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['x_t'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)
                response_mask = response_mask.reshape(B, S, CH, 7).sum(dim=(-1,-2))
                eos_mask = (response_mask > 0).float()
                
                vpreds = self._forward_micro_batch_smolvla(data)
                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'critic/vf_loss': vf_loss.detach().item(),
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                }

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'critic/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
