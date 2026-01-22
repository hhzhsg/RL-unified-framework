"""
评估循环

专门用于策略评估
"""
from typing import Dict, Any, List, Optional
import numpy as np

from .base_loop import BaseLoop
from ..interfaces import EnvInterface, PolicyInterface


class EvaluationLoop(BaseLoop):
    """
    评估循环
    
    职责:
    - 在环境中评估策略
    - 收集评估指标
    - 支持多episode评估
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        env: EnvInterface,
        config: Dict[str, Any],
    ):
        super().__init__()
        
        self.policy = policy
        self.env = env
        self.config = config
        
        # 评估结果
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_successes: List[bool] = []
        
        # 当前episode状态
        self._current_obs = None
        self._episode_reward = 0.0
        self._episode_length = 0
    
    def step(self) -> Dict[str, Any]:
        """执行单步评估"""
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()
        
        # 确定性动作
        action = self.policy.act(self._current_obs, deterministic=True)
        
        # 执行
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self._episode_reward += reward
        self._episode_length += 1
        
        step_info = {"reward": reward}
        
        if done:
            self._episode_rewards.append(self._episode_reward)
            self._episode_lengths.append(self._episode_length)
            self._episode_successes.append(info.get("success", False))
            
            step_info["episode_reward"] = self._episode_reward
            step_info["episode_length"] = self._episode_length
            step_info["episode_success"] = info.get("success", False)
            
            # 重置
            self._episode_reward = 0.0
            self._episode_length = 0
            self._current_obs, _ = self.env.reset()
        else:
            self._current_obs = next_obs
        
        return step_info
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        评估指定数量的episode
        
        Args:
            num_episodes: 评估episode数
            
        Returns:
            评估结果
        """
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_successes = []
        
        while len(self._episode_rewards) < num_episodes:
            self.step()
            self._step_count += 1
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, float]:
        """获取评估统计"""
        if not self._episode_rewards:
            return {}
        
        return {
            "num_episodes": len(self._episode_rewards),
            "avg_reward": np.mean(self._episode_rewards),
            "std_reward": np.std(self._episode_rewards),
            "min_reward": np.min(self._episode_rewards),
            "max_reward": np.max(self._episode_rewards),
            "avg_length": np.mean(self._episode_lengths),
            "success_rate": np.mean(self._episode_successes),
        }
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_successes = []
