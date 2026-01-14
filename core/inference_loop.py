"""
推理循环

在环境中执行策略，收集数据
"""
from typing import Optional, Dict, List, Any
from collections import deque
import numpy as np

from .weight_sync import BaseWeightSync
from policy import BasePolicy
from env import BaseEnv
from data import DataHub, Observation, RobotState, Action, Transition, Episode, EnvOutput
from config import InferenceConfig


class InferenceLoop:
    """
    推理循环
    
    负责:
    - 在环境中执行策略
    - 收集 rollout 数据
    - 从训练端同步权重
    - 评估策略性能
    
    Example:
        loop = InferenceLoop(
            policy=policy,
            env=env,
            config=config,
            data_hub=data_hub,
        )
        loop.collect_rollout(num_steps=1000)
    """
    
    def __init__(self,
                 policy: BasePolicy,
                 env: BaseEnv,
                 config: InferenceConfig,
                 data_hub: Optional[DataHub] = None,
                 weight_sync: Optional[BaseWeightSync] = None):
        """
        Args:
            policy: 策略网络
            env: 环境
            config: 推理配置
            data_hub: 数据中心 (用于写入 rollout)
            weight_sync: 权重同步器
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.data_hub = data_hub
        self.weight_sync = weight_sync
        
        self._current_env_output: Optional[EnvOutput] = None
        self._episode_count = 0
    
    def collect_rollout(self, 
                        num_steps: int, 
                        source: str = "rollout",
                        deterministic: bool = False) -> int:
        """
        收集 rollout 数据
        
        Args:
            num_steps: 收集步数
            source: 数据来源标签
            deterministic: 是否使用确定性策略
            
        Returns:
            实际收集的步数
        """
        collected = 0
        
        # 确保环境已重置
        if self._current_env_output is None:
            self._current_env_output = self.env.reset()
        
        for _ in range(num_steps):
            # 尝试同步权重
            self._try_sync_weights()
            
            env_output = self._current_env_output
            
            # 获取动作
            action = self.policy.act(
                env_output.obs,
                env_output.robot_state,
                deterministic=deterministic or self.config.deterministic,
            )
            
            # 执行动作
            next_env_output = self.env.step(action)
            
            # 构造 Transition
            transition = Transition(
                obs=env_output.obs,
                robot_state=env_output.robot_state,
                action=action,
                reward=next_env_output.reward,
                next_obs=next_env_output.obs,
                next_robot_state=next_env_output.robot_state,
                done=next_env_output.done,
                source=source,
            )
            
            # 写入 DataHub
            if self.data_hub:
                self.data_hub.write(transition, source=source)
            
            collected += 1
            
            # 更新状态
            if next_env_output.done:
                self._current_env_output = self.env.reset()
                self._episode_count += 1
            else:
                self._current_env_output = next_env_output
        
        return collected
    
    def evaluate(self, 
                 num_episodes: int = 10,
                 deterministic: bool = True) -> Dict[str, float]:
        """
        评估策略
        
        Args:
            num_episodes: 评估 episode 数
            deterministic: 是否使用确定性策略
            
        Returns:
            评估结果
        """
        rewards = []
        successes = []
        lengths = []
        
        for _ in range(num_episodes):
            env_output = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                action = self.policy.act(
                    env_output.obs,
                    env_output.robot_state,
                    deterministic=deterministic,
                )
                
                env_output = self.env.step(action)
                episode_reward += env_output.reward
                episode_length += 1
                
                if env_output.done:
                    break
            
            rewards.append(episode_reward)
            successes.append(env_output.info.get("success", False))
            lengths.append(episode_length)
        
        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
            "avg_length": np.mean(lengths),
        }
    
    def _try_sync_weights(self):
        """尝试从训练端同步权重"""
        if self.weight_sync is None:
            return
        
        result = self.weight_sync.pull()
        if result is not None:
            state_dict, version = result
            if "policy" in state_dict:
                self.policy.load_state_dict(state_dict["policy"])
    
    @property
    def episode_count(self) -> int:
        return self._episode_count
