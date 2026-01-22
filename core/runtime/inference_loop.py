"""
推理循环

负责在环境中执行策略，收集数据
"""
from typing import Dict, Any, Optional
import numpy as np

from .base_loop import BaseLoop
from ..interfaces import EnvInterface, PolicyInterface, SyncInterface


class InferenceLoop(BaseLoop):
    """
    推理循环
    
    职责:
    - 在环境中执行策略
    - 收集rollout数据
    - 从训练端同步权重
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        env: EnvInterface,
        config: Dict[str, Any],
        data_hub: Optional[Any] = None,
        weight_sync: Optional[SyncInterface] = None,
    ):
        super().__init__()
        
        self.policy = policy
        self.env = env
        self.config = config
        self.data_hub = data_hub
        self.weight_sync = weight_sync
        
        # 状态
        self._current_obs = None
        self._current_info = None
        self._episode_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        
        # 配置
        self.deterministic = config.get("deterministic", False)
        self.sync_freq = config.get("sync_freq", 10)
    
    def step(self) -> Dict[str, Any]:
        """执行单步推理"""
        # 0. 尝试同步权重
        if self.weight_sync and self._step_count % self.sync_freq == 0:
            self._try_sync_weights()
        
        # 1. 确保环境已重置
        if self._current_obs is None:
            self._current_obs, self._current_info = self.env.reset()
        
        # 2. 获取动作
        action = self.policy.act(self._current_obs, deterministic=self.deterministic)
        
        # 3. 执行动作
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # 4. 记录数据
        if self.data_hub is not None:
            transition = {
                "obs": self._current_obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": info,
            }
            self.data_hub.add(transition, source="rollout")
        
        # 5. 更新状态
        self._episode_reward += reward
        self._episode_length += 1
        
        step_info = {
            "reward": reward,
            "done": done,
        }
        
        if done:
            step_info["episode_reward"] = self._episode_reward
            step_info["episode_length"] = self._episode_length
            self._episode_count += 1
            self._episode_reward = 0.0
            self._episode_length = 0
            self._current_obs, self._current_info = self.env.reset()
        else:
            self._current_obs = next_obs
            self._current_info = info
        
        return step_info
    
    def _try_sync_weights(self) -> None:
        """尝试同步权重"""
        result = self.weight_sync.pull(tag="policy_weights")
        if result is not None:
            if "policy" in result:
                self.policy.load_state_dict(result["policy"])
    
    @property
    def episode_count(self) -> int:
        return self._episode_count
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估策略
        
        Args:
            num_episodes: 评估episode数
            
        Returns:
            评估结果
        """
        rewards = []
        lengths = []
        successes = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                action = self.policy.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(info.get("success", False))
        
        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_length": np.mean(lengths),
            "success_rate": np.mean(successes),
        }
