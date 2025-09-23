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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import statistics
from functools import partial

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, countdown, multiply, logic
from verl.trainer.ppo.ray_trainer import RayTrainer

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import special

import multiprocessing
import numpy as np
import random
import re
import csv

class RobRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config

    def verify(self, data):
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        # task_file_name_list = data.batch["task_file_name"]
        def tensor_to_str_list(tensor):
            # 1. 将张量移到 CPU 并转为 numpy
            bytes_array = tensor.cpu().numpy()
            
            # 2. 解码每个字节序列为字符串
            return [bytes(x).decode('utf-8', errors='ignore').rstrip('\0') for x in bytes_array]

        task_file_name_list = tensor_to_str_list(data.batch["task_file_name"])
        format = [1.0 for _ in range(len(completes))]
        print(f"vjepa_embedding shape: {data.batch['vjepa_embedding'].shape}; complete_shape: {data.batch['complete'].shape}; responses shape: {data.batch['responses'].shape}; finish_step shape: {data.batch['finish_step'].shape}; task_file_name_list length: {len(task_file_name_list)}")

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        print("==========score, reward_metrics, format_metrics, reward_format_metrics shape:", len(score), len(reward_metrics), len(format_metrics), len(reward_format_metrics))

        return score, reward_metrics, format_metrics, reward_format_metrics

    def new_reward_cal_ori(self, data):
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]
        print(data.batch['vjepa_embedding'].shape)

        embeddings = data.batch['vjepa_embedding'].cpu().numpy()  # shape: [batch, emb_dim]
        # task_file_names = data.batch['task_file_name']
        def tensor_to_str_list(tensor):
            # 1. 将张量移到 CPU 并转为 numpy
            bytes_array = tensor.cpu().numpy()
            
            # 2. 解码每个字节序列为字符串
            return [bytes(x).decode('utf-8', errors='ignore').rstrip('\0') for x in bytes_array]

        task_file_names = tensor_to_str_list(data.batch["task_file_name"])

        # 1. 解析 task_name（去掉 trial 部分）
        def extract_task_name(task_file_name):
            # 例：libero_10_task_7_trial_26 -> libero_10_task_7
            m = re.match(r"(libero_\w+_task_\d+)_trial_\d+", task_file_name)
            if m:
                return m.group(1)
            else:
                return task_file_name  # fallback

        task_names = [extract_task_name(name) for name in task_file_names]

        # 2. 按 task 分组
        task_to_indices = {}
        for idx, tname in enumerate(task_names):
            task_to_indices.setdefault(tname, []).append(idx)

        # 3. 计算 reward
        reward = [0.0] * batch_size
        for tname, indices in task_to_indices.items():
            # 找到所有成功和失败的索引
            succ_idx = [i for i in indices if completes[i]]
            fail_idx = [i for i in indices if not completes[i]]
            # 检查是否有全零的 embedding
            for i in indices:
                if np.all(embeddings[i] == 0):
                    reward[i] = 0.0  # 全零 embedding 直接给 0
                    continue
            if not succ_idx:
                # 没有成功数据，全部给0
                print("===没有成功数据，全给0===")
                for i in fail_idx:
                    reward[i] = 0.0
                continue
            # 随机选一个成功的 anchor
            anchor_idx = random.choice(succ_idx)
            anchor_emb = embeddings[anchor_idx]
            # 成功数据 reward=1
            for i in succ_idx:
                reward[i] = 1.0
            # 失败数据，计算距离
            if fail_idx:
                dists = [np.linalg.norm(embeddings[i] - anchor_emb) for i in fail_idx]
                max_dist = max(dists)
                min_dist = min(dists)
                for i, dist in zip(fail_idx, dists):
                    if max_dist == min_dist:
                        new_score = 0.0
                    else:
                        new_score = 0.6 * (max_dist - dist) / (max_dist - min_dist)
                    reward[i] = float(new_score)

        data.batch['acc'] = torch.tensor(reward, dtype=torch.float32, device=data.batch['responses'].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch['responses'].device)

        data.batch['new_reward'] = torch.tensor(reward, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['new_reward'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['new_reward'].mean().item()

        return reward, reward_metrics, format_metrics, reward_format_metrics

    def new_reward_cal(self, data):
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]
        
        embeddings = data.batch['vjepa_embedding'].cpu().numpy()
        # task_file_names = data.batch['task_file_name']
        def tensor_to_str_list(tensor):
            # 1. 将张量移到 CPU 并转为 numpy
            bytes_array = tensor.cpu().numpy()
            
            # 2. 解码每个字节序列为字符串
            return [bytes(x).decode('utf-8', errors='ignore').rstrip('\0') for x in bytes_array]

        task_file_names = tensor_to_str_list(data.batch["task_file_name"])

        # 1. 解析 task_name（去掉 trial 部分）
        def extract_task_name(task_file_name):
            m = re.match(r"(libero_\w+_task_\d+)_trial_\d+", task_file_name)
            return m.group(1) if m else task_file_name

        task_names = [extract_task_name(name) for name in task_file_names]

        # 2. 按 task 分组
        task_to_indices = {}
        for idx, tname in enumerate(task_names):
            task_to_indices.setdefault(tname, []).append(idx)

        # 3. 计算 reward
        reward = [0.0] * batch_size
        for tname, indices in task_to_indices.items():
            # 找到所有成功和失败的索引
            succ_idx = [i for i in indices if completes[i]]
            fail_idx = [i for i in indices if not completes[i]]
            
            # 处理全零embedding
            for i in indices:
                if np.all(embeddings[i] == 0):
                    reward[i] = 0.0
                    continue
            
            # 没有成功数据，全部给0
            if not succ_idx:
                for i in fail_idx:
                    reward[i] = 0.0
                continue
            
            # 成功数据直接给1.0
            for i in succ_idx:
                reward[i] = 1.0
            
            # 没有失败数据则跳过后续处理
            if not fail_idx:
                continue
            
            # ============== DBSCAN聚类改进 ==============
            succ_embeddings = embeddings[succ_idx]

            # 标准化数据（DBSCAN对尺度敏感）
            scaler = StandardScaler()
            scaled_succ = scaler.fit_transform(succ_embeddings)

            # 执行DBSCAN聚类（参数可调整）
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_succ)

            # 获取聚类中心
            cluster_centers = []
            unique_labels = set(clustering.labels_)

            for label in unique_labels:
                if label == -1:  # 跳过噪声点
                    continue
                cluster_mask = (clustering.labels_ == label)
                cluster_points = scaled_succ[cluster_mask]
                if len(cluster_points) > 0:  # 安全检查
                    cluster_center = scaler.inverse_transform(
                        cluster_points.mean(axis=0).reshape(1, -1)
                    ).flatten()
                    cluster_centers.append(cluster_center)

            # 如果没有找到有效簇，使用所有成功轨迹的均值作为中心
            if not cluster_centers:
                overall_center = scaler.inverse_transform(
                    scaled_succ.mean(axis=0).reshape(1, -1)
                ).flatten()
                cluster_centers = [overall_center]
            
            # ============== 非线性奖励映射 ==============
            # 计算每个失败轨迹到所有簇中心的最小距离
            min_dists = []
            for i in fail_idx:
                fail_emb = embeddings[i]
                dists = [np.linalg.norm(fail_emb - center) for center in cluster_centers]
                min_dists.append(min(dists))
            
            # 计算距离范围
            max_dist = max(min_dists) if min_dists else 0
            min_dist = min(min_dists) if min_dists else 0
            
            # 非线性映射参数
            sigmoid_steepness = 10.0  # 控制曲线陡峭度
            sigmoid_offset = 0.5      # 控制奖励分布中心
            
            for i, dist in zip(fail_idx, min_dists):
                if max_dist - min_dist < 1e-6:
                    normalized_dist = 0.5  # 避免除零
                else:
                    normalized_dist = (dist - min_dist) / (max_dist - min_dist)
                
                # Sigmoid非线性映射（expit即1/(1 + e^(-x))）
                sigmoid_input = sigmoid_steepness * (sigmoid_offset - normalized_dist)
                reward_val = 0.6 * special.expit(sigmoid_input)  # 固定最大奖励0.6
                
                reward[i] = float(reward_val)
        
        # 保存奖励到batch
        device = data.batch['responses'].device
        data.batch['acc'] = torch.tensor(reward, dtype=torch.float32, device=device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=device)
        data.batch['new_reward'] = torch.tensor(reward, dtype=torch.float32, device=device)
        
        # 计算指标
        reward_metrics = {'all': data.batch['new_reward'].mean().item()}
        format_metrics = {'all': data.batch['format_correctness'].mean().item()}
        reward_format_metrics = {'all': data.batch['new_reward'].mean().item()}
        
        return reward, reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0],-1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0],-1))
        
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len 
       
        if False: #'acc' in data.batch:
            # the separated rewards have been logged; now we add format correctness back for reward shaping
            #verifier_score = data.batch['acc'].cpu().numpy().tolist() + (0.0 * data.batch['format_correctness'].cpu().numpy()).tolist()
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            # verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.new_reward_cal(data)
            print(f"====新的奖励: {len(verifier_score)} 长度 内容: {verifier_score} ====")
            reward_metrics.update(verifier_metrics)

        completes = data.batch['complete'].tolist()
        # task_file_name_list = data.batch["task_file_name"]
        def tensor_to_str_list(tensor):
            # 1. 将张量移到 CPU 并转为 numpy
            bytes_array = tensor.cpu().numpy()
            
            # 2. 解码每个字节序列为字符串
            return [bytes(x).decode('utf-8', errors='ignore').rstrip('\0') for x in bytes_array]

        task_file_name_list = tensor_to_str_list(data.batch["task_file_name"])
        embedding_save = data.batch['vjepa_embedding'].cpu().numpy()

        # csv_file = '/inspire/hdd/project/embodied-multimodality/public/syfei/VLA-RL-verl/rollouts/csv/reward_record.csv'  # TODO：避免 hard code 的地址
        # if not os.path.exists(csv_file):
        #     with open(csv_file, 'w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(['verifier_score', "verifier_metrics", 'completes', "task_file_name_list", "embedding_save"])  # 如果不存在 csv 就建一个，然后输入第一行表头
        # with open(csv_file, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([verifier_score, verifier_metrics, completes, task_file_name_list, embedding_save])

        for i in range(verifier_reward.shape[0]):
            verifier_reward[i,valid_response_length[i]-1] += verifier_score[i]
            
        reward_tensor_dict['gt_scores'] = verifier_reward

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     raise  ValueError
        #     reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
        #     reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
        #     if self.config.reward_model.rm_coef!=0:
        #         reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0:
            
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env, num_cpus=60)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}, num_cpus=60)

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(RobActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef!=0.:
        if config.reward_model.rm_type == 'normal':
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == 'prime':
            from verl.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RobRewardManager( num_examine=0, config=config) # note: verifier is called both inside reward_fn and outside.

    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager( num_examine=1,config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    main()
