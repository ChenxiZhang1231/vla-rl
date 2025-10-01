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
            logp_action, logp_step, logp_outer, logp_joint = return_dict["logp_action"], return_dict["logp_step"], return_dict["logp_outer"], return_dict["logp_joint"]
            ent_step, ent_outer, ent_joint = return_dict["ent_step"], return_dict["ent_outer"], return_dict["ent_joint"]
            mean, std = return_dict["mean"], return_dict["std"]
            out_metric = return_dict["out_metric"]
            out_metric['logp_outer'] = logp_outer
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
            logp_action, logp_step, logp_outer, logp_joint = return_dict["logp_action"], return_dict["logp_step"], return_dict["logp_outer"], return_dict["logp_joint"]
            ent_step, ent_outer, ent_joint = return_dict["ent_step"], return_dict["ent_outer"], return_dict["ent_joint"]
            mean, std = return_dict["mean"], return_dict["std"]
            
            
            return ent_outer, logp_action, mean, std, logp_outer
        
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
                if self.config.kl_loss_type == 'outer_kl':
                    select_keys.append('old_logp_outer')
                    select_keys.append('ref_logp_outer')
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

                entropy, log_prob, mean, std, logp_outer = self._forward_micro_batch_update_smolvla(data)
                print("[chk] log_prob.requires_grad:", log_prob.requires_grad)
                print("[chk] loss.requires_grad(before):", (log_prob.reshape(1,-1).mean()).requires_grad)
                # breakpoint()
   
                old_log_prob_tmp = old_log_prob.reshape(B, -1)
                advantages_tmp = advantages.reshape(B, -1)
                response_mask_tmp = response_mask
                log_prob = log_prob.reshape(B, -1)
                if self.config.kl_loss_type == 'outer_kl':
                    log_prob = logp_outer
                    old_log_prob_tmp = data['old_logp_outer']
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
                policy_loss = pg_loss
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # if self.config.kl_loss_type == 'outer_kl':
                    #     ref_logp_outer = data["ref_logp_outer"]
                    if self.config.kl_loss_type == 'outer_kl':
                        CH_idx = torch.arange(CH)[None, None, :].to('cuda')
                        S_idx  = torch.arange(S)[None, :, None].to('cuda')
                        s_fin  = (data['finish_step'] // CH).view(B, 1, 1)
                        c_fin  = (data['finish_step'] %  CH).view(B, 1, 1)

                        mask_before = (S_idx <  s_fin).float()
                        mask_equal  = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
                        mask_actions = mask_before.expand(B, S, CH) + mask_equal
                        original_action_dim = 7
                        valid_CH = mask_actions.sum(dim=-1)                 # [B,S]
                        num_elems = (valid_CH * original_action_dim * K).clamp_min(1)
                    elif self.config.kl_loss_type == 'kl':
                        num_elems = K
                        
                    ref_log_prob = ref_log_prob.reshape(B, -1)
                    kld = core_algos.kl_penalty(
                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type, 
                        mean=mean, std=std, ref_mean=data.get("ref_mean", None), 
                        logp_outer=logp_outer,
                        ref_logp_outer=data.get("ref_logp_outer", None),
                        num_elems=num_elems,
                    )
                    kld = kld.reshape(B, -1)
                    # if max(abs(kld.max().item()), abs(kld.min().item())) != 0.0:
                    #     breakpoint()
                    if self.config.kl_loss_type == 'outer_kl':
                        kl_loss = kld.mean()
                    else:
                        kl_loss = verl_F.masked_mean(kld, response_mask)
                    target_kl = self.config.get('kl_loss_target', 0.0)
                    if target_kl == 0.0:
                        beta = self.config.kl_loss_coef
                    else:
                        kl_now = kl_loss.detach().item() / self.gradient_accumulation
                        hi, lo = 1.5, 0.5
                        beta = self.config.kl_loss_coef
                        if kl_now > hi * target_kl:
                            beta *= 1.5
                        elif kl_now < lo * target_kl:
                            beta /= 1.5
                        beta = float(max(1e-8, min(beta, 1e4)))
                        
                    policy_loss = policy_loss + kl_loss * beta
                    loss_info["actor/kl_loss"] = kl_loss.detach().item() / self.gradient_accumulation
                    loss_info["actor/kl_coef"] = beta
                    
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