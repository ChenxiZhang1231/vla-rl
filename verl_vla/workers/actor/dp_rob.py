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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import math
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl_vla import DataProto
from verl_vla.trainer.ppo import core_algos
from verl_vla.workers.actor import BasePPOActor
from verl_vla.utils.py_functional import append_to_dict
from verl_vla.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl_vla.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl_vla.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['RobDataParallelPPOActor', 'RobDataParallelPPOActorWM']



class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

        self.use_world_model = False
        
        
    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len
    
    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask
    
    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """
        mask_expanded = mask.unsqueeze(-1)  

        log_probs_masked = torch.where(
            mask_expanded,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )

        entropy_masked = torch.where(
            mask_expanded,
            entropy,
            torch.zeros_like(entropy, requires_grad=False)   
        )

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        # breakpoint()
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
            
            if self.config.vla == "openvla-oft":
                logits = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        )  # prevent model thinks we are generating
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                #assert (0<=responses<=255).all()
            
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
                assert len(log_probs.shape)==2 and len(entropy.shape)==2 
                log_probs = log_probs.reshape((batch_size, traj_len*8,7) )
                entropy = entropy.reshape((batch_size, traj_len*8,7) )

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                    attention_mask=attention_mask_unpad,
                                    pixel_values=pixel_values,
                                    use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD
                
                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                
                

            return entropy, log_probs
    
    def _forward_micro_batch_smolvla(self, micro_batch, return_logprob=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
            actions, lang_tokens, lang_masks, return_dict = self.actor_module.predict_action_chunk(micro_batch, recompute_log_prob=True)
            logp_action, logp_step, logp_outer, logp_joint, logp_elem = return_dict["logp_action"], return_dict["logp_step"], return_dict["logp_outer"], return_dict["logp_joint"], return_dict["logp_elem"]
            ent_step, ent_outer, ent_joint = return_dict["ent_step"], return_dict["ent_outer"], return_dict["ent_joint"]
            mean, std = return_dict["mean"], return_dict["std"]
            out_metric = return_dict["out_metric"]
            out_metric['logp_outer'] = logp_outer
            out_metric['logp_elem'] = logp_elem
            
            # if self.config.vla == "smolvla":
            #       # prevent model thinks we are generating
                
            #     assert self.actor_module.vocab_size == 32000
            #     start_index = self.actor_module.vocab_size - 256 
            #     logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            #     responses = responses - start_index
            #     #assert (0<=responses<=255).all()
            
            #     logits = logits.div(temperature) 
                
            #     log_probs = logprobs_from_logits(logits, responses)
            #     entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
            #     assert len(log_probs.shape)==2 and len(entropy.shape)==2 
            #     log_probs = log_probs.reshape((batch_size, traj_len*8,7) )
            #     entropy = entropy.reshape((batch_size, traj_len*8,7) )

            #     mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
            #     log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
            #     log_probs = log_probs.reshape((batch_size, traj_len*response_length))
            #     entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                

            return ent_outer, logp_action, mean, std, out_metric
        
    
    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
       
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
            if self.config.vla == "openvla-oft":
                
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

                
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                )  
                
                assert logits.requires_grad 
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))
                
                return entropy, log_probs
            
            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
                

    def _forward_micro_batch_update_smolvla(self, micro_batch, return_logprob=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        # breakpoint()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
            # breakpoint()
            actions, lang_tokens, lang_masks, return_dict = self.actor_module.predict_action_chunk_update(micro_batch, recompute_log_prob=True)
            logp_action, logp_step, logp_outer, logp_joint, logp_elem = return_dict["logp_action"], return_dict["logp_step"], return_dict["logp_outer"], return_dict["logp_joint"], return_dict["logp_elem"]
            ent_step, ent_outer, ent_joint = return_dict["ent_step"], return_dict["ent_outer"], return_dict["ent_joint"]
            mean, std = return_dict["mean"], return_dict["std"]
            
            
            return ent_outer, logp_action, mean, std, logp_outer, logp_elem
        
    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
 
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
            
        response_length = micro_batch['responses'].size(-1)
        #assert response_length == 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            #batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if  self.config.vla == "openvla-oft":
            
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                ) 
            
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            
                logits = logits.div(temperature) 
            
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape)==2 
                entropy = entropy.reshape((batch_size, traj_len*8,7) )
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        
        self.actor_module.eval()
        if 'step_images' in data.batch:
            del data.batch['step_images']

        micro_batch_size = data.meta_info['micro_batch_size'] #256
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz'] #trues
        self.pad_token_id = data.meta_info['pad_token_id']
        
        if self.config.vla == 'smolvla':
            select_keys = ['observation.images.image', 'observation.images.image_is_pad',
                        #    'observation.images.wrist_image', 'observation.images.wrist_image_is_pad', 
                        #    'observation.state', 'observation.state_is_pad', 
                           'action_tensor', 
                           "x_t", "t", "x_next",
                           "lang_tokens", "lang_masks", "finish_step"]
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        batch = data.select(batch_keys=select_keys).batch


        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        mean_lst = []
        std_lst = []
        out_metric_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # if isinstance(self.actor_module, FSDP):
                #     # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                #     param_ctx = FSDP.summon_full_params(self.actor_module, writeback=False, recurse=False)
                # with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # with torch.autocast(device_type='cuda', dtype=torch.float32):
                    if self.config.vla == 'smolvla':
                        ent_outer, logp_action, mean, std, out_metric = self._forward_micro_batch_smolvla(micro_batch, return_logprob=True)
                        log_probs = logp_action
                    else:
                        _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
                        mean, std = None, None
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                #     ent_outer, logp_action = self._forward_micro_batch_smolvla(micro_batch, return_logprob=True)
                #     log_probs = logp_action
            log_probs_lst.append(log_probs)
            mean_lst.append(mean)
            std_lst.append(std)
            out_metric_lst.append(out_metric)
            
        log_probs = torch.concat(log_probs_lst, dim=0)
        if mean is not None:
            mean = torch.concat(mean_lst, dim=0)
            std = torch.concat(std_lst, dim=0)
        else:
            mean, std = None, None
        out_metric_dict = {}
        for item in out_metric_lst:
            for key, values in item.items():
                if key not in out_metric_dict:
                    out_metric_dict[key] = [values]
                else:
                    out_metric_dict[key].append(values)
        
        for key, values in out_metric_dict.items():
            out_metric[key] = torch.cat(values)
        
                
        # if use_dynamic_bsz:
        if False:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, mean, std, out_metric

    def update_policy_smolvla(self, data: DataProto):
        self.actor_module.train()
        if 'step_images' in data.batch:
            del data.batch['step_images']
        
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        # temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        if self.config.vla == 'smolvla':
            select_keys = ['observation.images.image', 'observation.images.image_is_pad',
                        #    'observation.images.wrist_image', 'observation.images.wrist_image_is_pad', 
                        #    'observation.state', 'observation.state_is_pad', 
                           'action_tensor', 
                           "x_t", "t", "x_next",
                           "lang_tokens", "lang_masks", "finish_step",
                           'old_log_probs', 'advantages',]
            if self.config.use_kl_loss:
                select_keys.append('old_mean')
                select_keys.append('old_std')
                select_keys.append('ref_log_prob')
                select_keys.append('ref_mean')
                select_keys.append('ref_std')
                if self.config.kl_loss_type in ['outer_kl', 'hkb_lite', 'kl_seg', 'kl_kwise', 'kl_ffp']:
                    select_keys.append('old_logp_outer')
                    select_keys.append('ref_logp_outer')
                # if self.config.kl_loss_type in ['dwc_pg']:
                    select_keys.append('old_logp_elem')
                    select_keys.append('ref_logp_elem')
                    
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages', "finish_step"]
        batch = data.select(batch_keys=select_keys).batch
        
        # assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        # torch.autograd.set_detect_anomaly(True)
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload

                B, S, K, CH, D = data['x_t'].shape
                finish_step = data['finish_step'] * self.config.action_token_len
                response_length = S * self.config.action_chunks_len * self.config.action_token_len
                steps = torch.arange(response_length, device=data['x_t'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['x_t'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                
                advantages = data['advantages']  # .reshape(B, S, CH)
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff
                
                action_tensor = data['action_tensor']
                batch_size = action_tensor.size(0)
                traj_len = action_tensor.size(1)
             
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size ==0
                traj_split_num = int(traj_len/self.config.traj_mini_batch_size)
                assert traj_split_num == 1    

                entropy, log_prob, mean, std, logp_outer, logp_elem = self._forward_micro_batch_update_smolvla(data)
                print("[chk] log_prob.requires_grad:", log_prob.requires_grad)
                print("[chk] loss.requires_grad(before):", (log_prob.reshape(1,-1).mean()).requires_grad)
                # breakpoint()
   
                old_log_prob_tmp = old_log_prob.reshape(B, -1)
                advantages_tmp = advantages.reshape(B, -1)
                response_mask_tmp = response_mask
                log_prob = log_prob.reshape(B, -1)
                log_prob_action = log_prob
                ref_log_prob = data["ref_log_prob"].reshape(B, -1)
                if self.config.kl_loss_type in ['outer_kl', 'hkb_lite']:
                    log_prob = logp_outer
                    old_log_prob_tmp = data['old_logp_outer']
                    log_prob_action = logp_outer
                elif self.config.kl_loss_type in ['dwc_pg']:

                    device = log_prob.device
                    D = 7
                    CH_idx = torch.arange(CH, device=device)[None, None, :]
                    S_idx  = torch.arange(S,  device=device)[None, :, None]
                    s_fin  = (data['finish_step'] // CH).view(B, 1, 1)
                    c_fin  = (data['finish_step'] %  CH).view(B, 1, 1)

                    mask_before   = (S_idx < s_fin).float()
                    mask_equal    = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
                    mask_actions  = mask_before.expand(B, S, CH) + mask_equal          # [B,S,CH]
                    valid_CH      = mask_actions.sum(dim=-1).clamp_min(0.0)            # [B,S]
                    elem_count    = (valid_CH * K * D).clamp_min(1.0)                  # [B,S]
                    seg_weight    = valid_CH    

                    eps = 1e-6
                    std  = std.float().clamp_min(eps)
                    ref_std = data['ref_std'].float().clamp_min(eps)
                    ref_mean = data['ref_mean']
                    kl_elem = (torch.log(ref_std / std)
                                + (std ** 2 + (mean - ref_mean) ** 2) / (2 * ref_std ** 2)
                                - 0.5)   

                    kl_elem = torch.nan_to_num(kl_elem, 0.0, 0.0, 0.0)
                    kl_elem = (kl_elem * mask_actions[..., None].unsqueeze(2)).sum(dim=(-1, -2))  # [B,S,K]
                    kl_elem = torch.nan_to_num(kl_elem, 0.0, 0.0, 0.0)
                    KL_k_perdim = kl_elem / (valid_CH[..., None] * D).clamp_min(1.0)
                    KL_k_perdim = torch.nan_to_num(KL_k_perdim, 0.0, 0.0, 0.0)
                    
                    # breakpoint()
                    delta = KL_k_perdim
                    delta = (delta - delta.mean(dim=2, keepdim=True)) / (delta.std(dim=2, keepdim=True) + 1e-6)
                    with torch.no_grad():
                        K_eff  = logp_elem.size(2)
                        gamma  = 0.8
                        x      = (gamma * delta).float().clamp(-50, 50)
                        w      = torch.softmax(x, dim=2)
                        w      = 0.7 * w + 0.3 * (1.0 / K_eff)
                        
                        c_max = 2.0                                      # 别设成 1；建议 1.5~3.0
                        c_k   = K_eff * w                                # [B,S,K]

                        for _ in range(2):                               # 2~3 次足够
                            over   = c_k > c_max
                            excess = (c_k - c_max).clamp_min(0).sum(-1, keepdim=True)     # 多出来的质量
                            c_k    = torch.where(over, c_max, c_k)                         # 截到上界
                            free   = ~over
                            denom  = free.sum(-1, keepdim=True).clamp_min(1)
                            # 把“多出来的质量”均匀回流到尚未触顶的分量
                            c_k    = c_k + free.float() * (excess / denom)
                        # 保险：总和轻微偏差时按比例微调（不会破坏上界）
                        sum_k = c_k.sum(-1, keepdim=True).clamp_min(1e-6)
                        c_k   = c_k * (K_eff / sum_k).clamp_min(0.0)
                        c     = c_k[..., None, None].expand(-1, -1, -1, CH, D)
                        
                    log_prob = (c * logp_elem).sum(dim=2).reshape(B, -1)
                    old_log_prob_tmp = (c * data['old_logp_elem']).sum(dim=2).reshape(B, -1)
                    log_prob_action = logp_elem.sum(dim=2).reshape(B, -1)
                
                if self.config.kl_loss_type in ['kl_kwise']:
                    # breakpoint()
                    eps = 1e-6
                    B, S, K, CH, D = logp_elem.shape
                    device = log_prob.device
                    D = 7
                    CH_idx = torch.arange(CH, device=device)[None, None, :]
                    S_idx  = torch.arange(S,  device=device)[None, :, None]
                    s_fin  = (data['finish_step'] // CH).view(B, 1, 1)
                    c_fin  = (data['finish_step'] %  CH).view(B, 1, 1)

                    mask_before   = (S_idx < s_fin).float()
                    mask_equal    = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
                    mask_actions  = mask_before.expand(B, S, CH) + mask_equal          # [B,S,CH]
                    # 有效元素掩码，扩到 [B,S,K,CH,D]
                    mask_elem = mask_actions[:, :, None, :, None].expand(B, S, K, CH, D).float()

                    # 逐 k 的 “噪声强度” 代理：E_{CH,D}[ 1/std^2 ]，只看有效 CH、维度
                    std32  = std.to(torch.float32).clamp_min(eps)
                    inv_var = 1.0 / (std32 ** 2)                        # [B,S,K,CH,D]
                    g_raw = (inv_var * mask_elem).sum(dim=(-1, -2))                    # [B,S,K]
                    valid_CH = mask_actions.sum(dim=-1).clamp_min(1.0)               # [B,S]
                    g_raw = g_raw / (valid_CH[..., None] * D).clamp_min(1.0)             # [B,S,K]
                    g_raw = torch.nan_to_num(g_raw, 0.0, 0.0, 0.0)
                    g_z   = (g_raw - g_raw.mean(dim=2, keepdim=True)) / (g_raw.std(dim=2, keepdim=True) + eps)
                    gamma = 0.5
                    g_k   = torch.tanh(gamma * g_z)    

                    # 基线：b_k = η_b * g_k  （η_b ∈ {0.1, 0.3, 1.0} 试三档）
                    eta_b = self.config.k_baseline_eta   # 0.1 / 0.3 / 1.0
                    b_k = (eta_b * g_k).detach()  
                    b_k = b_k[..., None].expand(-1, -1, -1, CH)
                    b_k = b_k.reshape(B, -1)
                    
                    log_prob = logp_elem.sum(dim=-1).reshape(B, -1)   #  [B, S*K*CH]
                    old_log_prob_tmp = data['old_logp_elem'].sum(dim=-1).reshape(B, -1)
                    
                    response_mask_tmp = mask_actions[:, :, None, :].expand(B,S,K,CH).reshape(B, -1).bool()
                    advantages_tmp = advantages_tmp.reshape(B, S, CH, D).sum(dim=-1)
                    advantages_tmp = advantages_tmp[:, :, None, :].expand(B,S,K,CH).reshape(B, -1)
                    advantages_tmp = advantages_tmp - b_k
                else:
                    b_k = None

                if self.config.kl_loss_type in ['kl_ffp']:
                    breakpoint()
                    eps = 1e-6
                    B, S, K, CH, D = logp_elem.shape
                    device = log_prob.device
                    D = 7
                    CH_idx = torch.arange(CH, device=device)[None, None, :]
                    S_idx  = torch.arange(S,  device=device)[None, :, None]
                    s_fin  = (data['finish_step'] // CH).view(B, 1, 1)
                    c_fin  = (data['finish_step'] %  CH).view(B, 1, 1)

                    mask_before   = (S_idx < s_fin).float()
                    mask_equal    = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
                    mask_actions  = mask_before.expand(B, S, CH) + mask_equal          # [B,S,CH]
                    # 有效元素掩码，扩到 [B,S,K,CH,D]
                    mask_elem = mask_actions[:, :, None, :, None].expand(B, S, K, CH, D).float()

                    # 逐 k 的 “噪声强度” 代理：E_{CH,D}[ 1/std^2 ]，只看有效 CH、维度
                    p      = 0.5         # 开始先用 sqrt
                    alpha  = 0.3         # 30% 均匀混合，防塌缩
                    w_min, w_max = 0.3, 2.0

                    sigma2 = (std.to(torch.float32).squeeze(-1, -1) ** 2)         # [B,S,K]
                    w_k    = (sigma2 + eps) ** p                                  # [B,S,K]
                    w_k    = w_k / (w_k.mean(dim=2, keepdim=True) + eps)          # 段内均值=1
                    w_k    = alpha * 1.0 + (1 - alpha) * w_k                      # 混均匀
                    w_k    = torch.clamp(w_k, w_min, w_max).detach()              # stop-grad

                    w_kch  = w_k[..., None].expand(B, S, K, CH).reshape(B, -1)     # [B, S*K*CH]

                    log_prob = logp_elem.sum(dim=-1).reshape(B, -1)   #  [B, S*K*CH]
                    old_log_prob_tmp = data['old_logp_elem'].sum(dim=-1).reshape(B, -1)
                    
                    response_mask_tmp = mask_actions[:, :, None, :].expand(B,S,K,CH).reshape(B, -1).bool()
                    advantages_tmp = advantages_tmp.reshape(B, S, CH, D).sum(dim=-1)
                    advantages_tmp = advantages_tmp[:, :, None, :].expand(B,S,K,CH).reshape(B, -1)
                    advantages_tmp = advantages_tmp * w_kch
                       
                print("[dbg] T_lp=", log_prob.shape[1], "T_mask=", response_mask_tmp.shape[1], "mask.sum=", response_mask_tmp.sum().item())
                # assert log_prob.shape[1] == response_mask_tmp.shape[1], f"length mismatch: logp={log_prob.shape}, mask={response_mask_tmp.shape}"
                # assert response_mask_tmp.sum().item() > 0, "mask 全 0：这一轮没有任何有效 token"
                if self.config.algo == "grpo":
                    pass
                elif self.config.algo == "ppo":
                    advantages_tmp = advantages_tmp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config.action_chunks_len, self.config.action_token_len)
                    advantages_tmp = advantages_tmp.reshape(B, -1)
                else:
                    raise
                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                                log_prob=log_prob,
                                                                                advantages=advantages_tmp,
                                                                                eos_mask=response_mask_tmp,
                                                                                clip_ratio_high=clip_ratio_high,
                                                                                clip_ratio_low=clip_ratio_low,
                                                                                dlogp_clamp=self.config.dlogp_clamp,
                                                                                dlogp_clamp_max=self.config.dlogp_clamp_max,
                                                                                dlogp_clamp_min=self.config.dlogp_clamp_min,
                                                                                loss_type=self.config.kl_loss_type,
                                                                                data_shape=(B, S, K, CH, 7))
                    
                response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                # breakpoint()
                device = log_prob.device
                D = 7
                policy_loss = pg_loss
                if self.config.use_kl_loss:
                    kl_type = self.config.kl_loss_type
                    # ---- 覆盖率（只在 outer/hkb 下需要）----
                    if kl_type in ['outer_kl', 'hkb_lite', 'kl_seg']:
                        CH_idx = torch.arange(CH, device=device)[None, None, :]
                        S_idx  = torch.arange(S,  device=device)[None, :, None]
                        s_fin  = (data['finish_step'] // CH).view(B, 1, 1)
                        c_fin  = (data['finish_step'] %  CH).view(B, 1, 1)

                        mask_before   = (S_idx < s_fin).float()
                        mask_equal    = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
                        mask_actions  = mask_before.expand(B, S, CH) + mask_equal          # [B,S,CH]
                        valid_CH      = mask_actions.sum(dim=-1).clamp_min(0.0)            # [B,S]
                        elem_count    = (valid_CH * K * D).clamp_min(1.0)                  # [B,S]
                        seg_weight    = valid_CH                                           # [B,S] 用于 HKB 段级聚合
                    else:
                        mask_actions = None
                        valid_CH     = None
                        elem_count   = 1

                    # ---- 计算 KL（外层 + 可选 HKB 内层）----
                    # core_algos.kl_penalty 约定返回：
                    # - outer_kl / hkb_lite:  KL_seg_perdim: [B,S], 以及（hkb_lite 时）KL_k_elem: [B,S,K,CH,D]
                    # - kl:                  元素/Token级 KL（你原有的分支，用 response_mask）
                    KL_out = core_algos.kl_penalty(
                        logprob=log_prob_action.reshape(B, -1),
                        ref_logprob=ref_log_prob,
                        kl_penalty=kl_type,
                        mean=mean, std=std,                               # std 必须是 σ_t * sqrt(|Δt|)
                        ref_mean=data.get("ref_mean", None),
                        ref_std=data.get("ref_std", None),
                        logp_outer=logp_outer,                            # [B,S]
                        ref_logp_outer=data.get("ref_logp_outer", None),  # [B,S]
                        num_elems=elem_count,                             # [B,S] 仅供 outer/hkb 使用
                    )

                    if kl_type == 'hkb_lite':
                        KL_seg_perdim, KL_k_elem = KL_out                 # shapes: [B,S], [B,S,K,CH,D]
                    elif kl_type == 'kl_seg':
                        KL_action_perdim, KL_seg_perdim = KL_out
                    else:
                        KL_action_perdim = KL_out
                        KL_k_elem = None

                    # ---- 段级 outer-KL 的覆盖率加权聚合 ----
                    if kl_type in ['outer_kl', 'hkb_lite']:
                        # KL_seg_perdim 是 [B,S] 的“每维”outer-KL；按元素数加权：
                        L_kl_seg = (KL_seg_perdim * elem_count).sum() / elem_count.sum().clamp_min(1.0)
                    elif kl_type == 'kl_seg':
                        L_kl_action = verl_F.masked_mean(KL_action_perdim.reshape(B, -1), response_mask)
                        L_kl_seg = (KL_seg_perdim * elem_count).sum() / elem_count.sum().clamp_min(1.0)
                    else:
                        # 你原来的 token-level 分支
                        L_kl_action = verl_F.masked_mean(KL_action_perdim.reshape(B, -1), response_mask)
                        L_kl_seg = None

                    # ---- HKB-Lite：逐 k 的方向塑形（小权重）----
                    if kl_type == 'hkb_lite':
                        # 逐步每维 KL：对 CH,D 做平均，mask CH
                        KL_k_perdim = (KL_k_elem * mask_actions[..., None].unsqueeze(2)).sum(dim=(-1, -2))  # [B,S,K]
                        KL_k_perdim = torch.nan_to_num(KL_k_perdim, 0.0, 0.0, 0.0)
                        KL_k_perdim = KL_k_perdim / (valid_CH[..., None] * D).clamp_min(1.0)
                        KL_k_perdim = torch.nan_to_num(KL_k_perdim, 0.0, 0.0, 0.0)

                        with torch.no_grad():
                            gamma = getattr(self.config, 'hkb_gamma', 1.0)
                            x = (gamma * KL_k_perdim).to(torch.float32)
                            x = torch.clamp(x, -50.0, 50.0)
                            x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
                            w_k = torch.softmax(x, dim=2)
                            # w_k = torch.softmax(gamma * KL_k_perdim, dim=2)                 # [B,S,K]
                            # 可选：混一点均匀避免塌缩
                            # alpha = 0.7
                            # w_k = (1 - alpha) * (1.0 / K) + alpha * w_k

                        # 段内加权 → 段级值；再按覆盖率聚合到 batch
                        L_hkb_seg = (w_k * KL_k_perdim).sum(dim=2)                           # [B,S]
                        L_hkb     = (L_hkb_seg * seg_weight).sum() / seg_weight.sum().clamp_min(1.0)

                    # ---- KL 系数（outer 自适应可选；inner 固定为 outer 的 5~10%）
                    target_kl = self.config.get('kl_loss_target', 0.0)
                    beta       = self.config.kl_loss_coef
                    beta_inner = self.config.kl_loss_coef_inner
                    beta_seg = self.config.kl_loss_coef_seg

                    if kl_type in ['outer_kl', 'hkb_lite']:
                        L_kl_main = L_kl_seg
                        L_kl_second = L_hkb
                    else:
                        L_kl_main = L_kl_action
                        L_kl_second = L_kl_seg
                        
                    if target_kl > 0.0:  # 简单自适应 outer
                        kl_now = (L_kl_main.detach().item()) / self.gradient_accumulation
                        hi, lo = 1.5, 0.5
                        if   kl_now > hi * target_kl: beta *= 1.5
                        elif kl_now < lo * target_kl: beta /= 1.5
                        beta = float(max(1e-8, min(beta, 1e4)))

                    # ---- 汇总到 policy loss ----
                    policy_loss = policy_loss + beta * L_kl_main
                    if kl_type == 'hkb_lite':
                        policy_loss = policy_loss + beta_inner * L_kl_second
                    elif kl_type == 'kl_seg':
                        policy_loss = policy_loss + beta_seg * L_kl_second

                    # ---- 日志 ----
                    loss_info["actor/kl_loss"] = (L_kl_main.detach().item() / self.gradient_accumulation)
                    loss_info["actor/kl_coef"] = beta
                    if kl_type == 'hkb_lite':  # HACK: / self.gradient_accumulation is a bug here,
                        loss_info["actor/kl_loss_inner"] = (L_kl_second.detach().item() / self.gradient_accumulation)
                        loss_info["actor/kl_coef_inner"] = beta_inner
                    if kl_type == 'kl_seg':
                        loss_info["actor/kl_loss_seg"] = (L_kl_second.detach().item() / self.gradient_accumulation)
                        loss_info["actor/kl_coef_seg"] = beta_seg
                    if kl_type == 'dwc_pg':
                        loss_info["actor/dwc_pg_c"] = c.mean().item()

                    
                loss = policy_loss / self.gradient_accumulation
                loss.backward()
                    
                n_has_grad_after = sum((p.grad is not None) for p in self.actor_module.parameters())
                print(f"[chk] after backward: with_grad={n_has_grad_after}")
                if n_has_grad_after == 0:
                    # breakpoint()
                    assert n_has_grad_after != 0, "n_has_grad_after == 0"
                
                loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()

                append_to_dict(metrics, loss_info)
            
            grad_norm = self._optimizer_step()
            # breakpoint()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            grad_norm_val = data['actor/grad_norm']
            if math.isnan(grad_norm_val):
                print("!!! grad_norm is NaN! Breaking...")
                breakpoint()
            if grad_norm_val == 0:
                print("!!! grad_norm is zero! Breaking...")
                breakpoint()
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
            # breakpoint()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                
                
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
                
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size ==0
                traj_split_num = int(traj_len/self.config.traj_mini_batch_size)
                
                
    

                for i in range(0, traj_len, int(traj_len/traj_split_num)):
                   
                    entropy, log_prob = self._forward_micro_batch_update(input_ids=input_ids[i:i+int(traj_len/traj_split_num)], attention_mask=attention_mask[i:i+int(traj_len/traj_split_num)], pixel_values=pixel_values[i:i+int(traj_len/traj_split_num)], responses=responses[i:i+int(traj_len/traj_split_num)], temperature=temperature)
                    
                    slice_id = i*self.config.action_token_len*self.config.action_chunks_len
                    next_slice_id = (i+int(traj_len/traj_split_num))*self.config.action_token_len*self.config.action_chunks_len
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                        
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                            log_prob=log_prob,
                                                                            advantages=advantages_tmp,
                                                                            eos_mask=response_mask_tmp,
                                                                            clip_ratio_high=clip_ratio_high,
                                                                            clip_ratio_low=clip_ratio_low)
                    
                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss* response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac* response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl* response_mask_tmp_sum / response_mask_sum
                    
                    policy_loss = pg_loss / response_mask_sum
                    
                    loss = policy_loss / self.gradient_accumulation
                    
                    loss.backward()
                    n_has_grad_after = sum((p.grad is not None) for p in self.actor_module.parameters())
                    if n_has_grad_after == 0:
                        breakpoint()
                    print(f"[chk] after backward: with_grad={n_has_grad_after}")
                    
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()

                append_to_dict(metrics, loss_info)
            # breakpoint()
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
    
    
    def compute_entropy(self, bacth_data: DataProto):
        
        if bacth_data.meta_info['train_mode'] ==True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        print("dataloader_length:", len(dataloader))
        
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                        
                
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
    
    
    
class RobDataParallelPPOActorWM(BasePPOActor):

    def __init__(
        self,
        config,
        wm_module: nn.Module,
        wm_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.wm_module = wm_module
        self.wm_optimizer = wm_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)

    def compute_log_prob(self, data: DataProto):
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```


        """
        pass

    def update_policy(self, data: DataProto):
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass