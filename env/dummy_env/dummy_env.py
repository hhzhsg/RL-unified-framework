"""
Dummy 测试环境

用于框架测试和调试，不依赖任何外部库
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np

from env.base_env import BaseEnv
from core.orchestration import register_env


@register_env("dummy")
class DummyEnv(BaseEnv):
    """
    简单的 Dummy 环境用于测试
    
    状态空间: Box(state_dim,)
    动作空间: Box(action_dim,)
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        max_episode_steps: int = 200,
        **kwargs
    ):
        # 构建 config 传给父类
        config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_episode_steps": max_episode_steps,
            **kwargs
        }
        super().__init__(config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        
        self._state = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self._state = np.random.randn(self.state_dim).astype(np.float32)
        self._step_count = 0
        
        obs = {"state": self._state.copy()}
        info = {}
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行动作"""
        action = np.asarray(action, dtype=np.float32)
        
        # 简单的状态转移：状态 = 旧状态 * 0.9 + 动作的影响 + 噪声
        noise = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        action_effect = np.zeros(self.state_dim, dtype=np.float32)
        action_effect[:self.action_dim] = action * 0.1
        
        self._state = self._state * 0.9 + action_effect + noise
        self._step_count += 1
        
        # 奖励：状态范数的负值（鼓励状态接近零）
        reward = -np.linalg.norm(self._state) * 0.1
        
        # 终止条件
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        
        obs = {"state": self._state.copy()}
        info = {"step": self._step_count}
        
        return obs, float(reward), terminated, truncated, info
    
    def close(self) -> None:
        """关闭环境"""
        pass
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        return {"state": {"shape": (self.state_dim,), "dtype": "float32"}}
    
    @property
    def action_space(self) -> Dict[str, Any]:
        return {"shape": (self.action_dim,), "dtype": "float32", "low": -1.0, "high": 1.0}
