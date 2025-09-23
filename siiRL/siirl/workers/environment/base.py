# Copyright (c) 2025, Shanghai Innovation Institute.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     <url id="d0rb3ebacc47j8if2om0" type="url" status="parsed" title="Apache License, Version 2.0" wc="10467">http://www.apache.org/licenses/LICENSE-2.0</url>
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Dict, Optional, Tuple
import yaml
import json
import os
import importlib
import ray
import sys
from loguru import logger


# --- 1. Abstract Base Class for Environment ---


async def get_ddp_world_size_rank(local_world_size, local_rank, local_parallel_size):
    ddp_world_size = local_world_size // local_parallel_size
    ddp_rank = local_rank // local_parallel_size
    return ddp_world_size, ddp_rank


class BaseEnvironment(ABC):
    """
    BaseEnvironment defines functions for users to implement
    """

    def __init__(self, env_id: str, config: Dict[str, Any]):
        self.env_id = env_id
        self.config = config
        self._validate_config()
        logger.info(f"[BaseEnvironment] Environment '{self.env_id}' initialized, configuration: {self.config}")

    def _validate_config(self):
        """(Optional) Validates whether the input configuration meets the environment requirements."""
        pass

    async def _reset(self, dp_rank: int, ddp_world_size: int, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Any:
        obs = await self.reset(dp_rank, ddp_world_size, seed, options)
        return await self.format_observation(obs)

    async def reset(self, dp_rank: int, ddp_world_size: int, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resets the environment to its initial state.
        Args:
            seed: The seed used for environment randomness.
            options: Environment-specific reset options.
        Returns:
            Initial observation (raw observation).
        """
        logger.info("Reset not implemented")
        raise NotImplementedError
        return obs

    async def step(self, action: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Executes an action in the environment. Subclasses can override this method. Alternatively, you can override do_action, get_rewards, and get_obs, but with limited flexibility.
        Args:
            action: Input parameters for executing the action.
        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
                   - observation: The environment's raw observation.
                   - reward: The reward obtained after executing the action.
                   - info: A dictionary containing additional diagnostic information as needed.
        """
        logger.info("Step not implemented")
        raise NotImplementedError
        return next_obs, rewards, info

    async def get_rewards(self, actions: Any) -> Any:
        """Returns the rewards of action"""
        return None

    async def _step(self, action: Any) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Adds env_id to the returned data after executing the action.
        """
        obs, rewards, info = await self.step(action)
        info["sii_env_id"] = self.env_id
        obs = await self.format_observation(obs)
        return obs, rewards, info

    async def format_observation(self, observation: Any) -> Any:
        """
        Formats the environment's raw observation into a format suitable for LLM input.
        The default implementation directly returns the raw observation. Subclasses can override this method.
        Automatically called by _step and _reset.
        Args:
            observation: The raw observation from step() or reset().
        Returns:
            Formatted observation.
        """
        # logger.info(f"[BaseEnvironment {self.env_id}] Formatting observation (default implementation).")
        return observation

    async def get_observation_space(self) -> Any:
        """Returns the definition of the environment's observation space (e.g., a gym.spaces.Space object)."""
        # raise NotImplementedError # Encouraged for subclasses to implement
        return None  # Simplified

    async def get_action_space(self) -> Any:
        """Returns the definition of the environment's action space."""
        # raise NotImplementedError # Encouraged for subclasses to implement
        return None  # Simplified

    async def close(self) -> None:
        """Cleans up environment resources if needed."""
        logger.info(f"[BaseEnvironment {self.env_id}] Environment closed.")
        pass


# --- 2. Ray Actor Wrapping Environment Instances ---
@ray.remote
class RayEnvRunner:
    def __init__(self, env_id: str, env_config: Dict[str, Any], env_class_ref: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        module_path, class_name = env_class_ref.rsplit(".", 1)
        module = importlib.import_module(module_path)
        env_class = getattr(module, class_name)
        if not issubclass(env_class, BaseEnvironment):
            raise TypeError(f"Class '{env_class_ref}' must be a subclass of BaseEnvironment.")
        self.env = env_class(env_id, env_config)

    def _validate_config(self):
        self.env._validate_config()

    async def reset(self, local_world_size: int, local_rank: int, local_parallel_size: int, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Any:
        ddp_world_size, dp_rank = await get_ddp_world_size_rank(local_world_size, local_rank, local_parallel_size)
        futures = await self.env._reset(dp_rank, ddp_world_size, seed, options)
        return futures

    async def step(self, action: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        return await self.env._step(action)

    async def get_rewards(self, action: Any) -> Any:
        """Returns the rewards of action"""
        return await self.env.get_rewards(action)

    async def format_observation(self, observation: Any) -> Any:
        """
        Formats the environment's raw observation into a format suitable for LLM input.
        The default implementation directly returns the raw observation. Subclasses can override this method.
        Automatically called by _step and _reset.
        Args:
            observation: The raw observation from step() or reset().
        Returns:
            Formatted observation.
        """
        # logger.info(f"[BaseEnvironment {self.env_id}] Formatting observation (default implementation).")
        return await self.env.format_observation(observation)

    async def get_observation_space(self) -> Any:
        """Returns the definition of the environment's observation space (e.g., a gym.spaces.Space object)."""
        # raise NotImplementedError # Encouraged for subclasses to implement
        return await self.env.get_observation_space()  # Simplified

    async def get_action_space(self) -> Any:
        """Returns the definition of the environment's action space."""
        # raise NotImplementedError # Encouraged for subclasses to implement
        return await self.env.get_action_space()  # Simplified

    async def close(self) -> None:
        """Cleans up environment resources if needed."""
        return await self.env.close()  # Simplified


"""
this method requires env don't save status
"""


@ray.remote
class ParallelRayEnvRunner:
    """
    Multi Ray Actor used
    """

    def __init__(self, env_id: int, agent_parallel_num: int, env_class_ref: str, env_config: Dict[str, Any]):
        """
        Args:
            env_id: A unique identifier for the environment.
            env_class_ref: A reference string pointing to a subclass of BaseEnvironment (e.g., "my_module.MyCustomEnv").
            env_config: A dictionary of configurations passed to the environment constructor.
        """
        self.env_id = env_id
        self.env_worker = Queue()
        # todo: calculate how many workers are in one environment, now max = 4
        self.worker_num = agent_parallel_num
        try:
            for i in range(self.worker_num):
                self.env_worker.put(RayEnvRunner.remote(env_id=env_id, env_config=env_config, env_class_ref=env_class_ref))
            logger.info(f"[RayEnvActor {self.env_id}] Successfully instantiated environment type: {env_class_ref}")
        except Exception as e:
            logger.info(f"Error: [RayEnvActor {self.env_id}] Failed to initialize environment '{env_class_ref}': {e}")
            # In practical applications, a more robust error handling or retry mechanism may be needed
            raise

    async def reset(self, local_world_size: int, local_rank: int, local_parallel_size: int, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Any:
        # logger.info(f"[RayEnvActor {self.env_id}] Calling reset...")
        # Use all envWorkers to reset
        futures = []
        for i in range(self.worker_num):
            worker = self.env_worker.get()
            futures.append(worker.reset.remote(local_world_size, local_rank, local_parallel_size, seed, options))
            self.env_worker.put(worker)
        return futures

    async def step(self, action: Any) -> Any:
        # logger.info(f"[RayEnvActor {self.env_id}] Calling step, action: {action is not None}")
        worker = self.env_worker.get()
        futures = worker.step.remote(action)
        self.env_worker.put(worker)
        return futures

    async def get_rewards(self, action: Any) -> Any:
        # logger.info(f"[RayEnvActor {self.env_id}] Calling step, action: {action is not None}")
        worker = self.env_worker.get()
        futures = worker.get_rewards.remote(action)
        self.env_worker.put(worker)
        return futures

    async def format_observation(self, observation: Any) -> Any:
        # logger.info(f"[RayEnvActor {self.env_id}] Calling format_observation...")
        worker = self.env_worker.get()
        futures = worker.format_observation.remote(observation)
        self.env_worker.put(worker)
        return futures

    async def get_observation_space(self) -> Any:
        worker = self.env_worker.get()
        futures = worker.get_observation_space.remote()
        self.env_worker.put(worker)
        return futures

    async def get_action_space(self) -> Any:
        worker = self.env_worker.get()
        futures = worker.get_action_space.remote()
        self.env_worker.put(worker)
        return futures

    async def close(self) -> None:
        for i in range(self.worker_num):
            worker = self.env_worker.get()
            await worker.close.remote()

    def get_env_id(self) -> str:
        return self.env_id


# --- 3. Environment Configuration Loader ---
class EnvironmentConfigLoader:
    """
    Loads environment configurations from configuration files (YAML/JSON) and creates RayEnvironmentActor instances.
    """

    def __init__(self):
        pass

    @staticmethod
    def load_environments_from_file(file_path: str) -> Dict[int, "ray.actor.ActorHandle"]:
        """
        Loads all environment definitions from the specified configuration file and starts a RayEnvironmentActor for each definition.
        Args:
            file_path: The path to the YAML or JSON configuration file.
        Returns:
            A dictionary mapping env_id to the corresponding RayEnvironmentActor handle.
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        raw_configs = None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_extension in [".yaml", ".yml"]:
                    raw_configs_list = yaml.safe_load(f)
                elif file_extension == ".json":
                    raw_configs_list = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file extension: {file_extension}")

            if not isinstance(raw_configs_list, list):  # Assume the configuration file is a list of environment configuration objects
                raise ValueError("The environment configuration file should contain a list of environment configuration objects.")

        except Exception as e:
            logger.info(f"Error: Failed to load or parse environment configuration file '{file_path}': {e}")
            raise

        environment_actors: Dict[int, "ray.actor.ActorHandle"] = {}
        for env_conf in raw_configs_list:
            env_id = env_conf.get("environment_id", None)
            env_class_ref = env_conf.get("environment_type_ref", None)
            env_specific_config = env_conf.get("config", {})
            ray_actor_options = env_conf.get("source_options", {})  # E.g., {"num_cpus": 1}

            if env_id is None or env_class_ref is None:
                logger.info(f"Warning: Skipping invalid environment configuration (missing id or type_ref): {env_conf}")
                continue
            try:
                agent_parallel_num = env_conf.get("agent_parallel_num", 1)
                n_agent = env_specific_config.get("num_agents", 1)
                if agent_parallel_num > 1 and n_agent > 1:
                    actor_handle = ParallelRayEnvRunner.options(**ray_actor_options).remote(env_id=env_id, agent_parallel_num=min(agent_parallel_num, n_agent), env_class_ref=env_class_ref, env_config=env_specific_config)
                    environment_actors[env_id] = actor_handle
                    logger.info(f"Successfully started RayEnvironmentActor for env_id='{env_id}' (type: {env_class_ref}).")
                else:
                    actor_handle = RayEnvRunner.options(**ray_actor_options).remote(env_id=env_id, env_class_ref=env_class_ref, env_config=env_specific_config)
                    environment_actors[env_id] = actor_handle
                    logger.info(f"Successfully started RayEnvironmentActor for env_id='{env_id}' (type: {env_class_ref}).")
            except Exception as e:
                logger.info(f"Error: Failed to start RayEnvironmentActor for env_id='{env_id}': {e}")

        return environment_actors
