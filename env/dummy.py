"""
Dummy 环境

用于测试和验证的简单环境
"""
from typing import Optional
import numpy as np

from .base import BaseEnv
from data import Action, EnvOutput, Observation, RobotState
from config import EnvConfig


class DummyEnv(BaseEnv):
    """
    Dummy 环境
    
    简单的测试环境:
    - 状态: 随机向量
    - 动作: 任意向量
    - 奖励: 动作与状态的相似度
    - 终止: 步数达到上限或随机终止
    """
    
    def __init__(self, config: EnvConfig, deterministic: bool = False):
        """
        Args:
            config: 环境配置
            deterministic: 是否确定性 (用于测试)
        """
        super().__init__(config)
        self.deterministic = deterministic
        self._rng = np.random.default_rng(42 if deterministic else None)
        self._current_state = None
        self._goal_state = None
    
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        """重置环境"""
        self._step_count = 0
        
        # 初始状态
        self._current_state = self._rng.standard_normal(self.state_dim).astype(np.float32)
        
        # 目标状态
        self._goal_state = self._rng.standard_normal(self.state_dim).astype(np.float32)
        
        return EnvOutput(
            obs=Observation(images={}, language=""),
            robot_state=RobotState(raw_state=self._current_state.copy()),
            reward=0.0,
            done=False,
            info={"task_id": task_id},
        )
    
    def step(self, action: Action) -> EnvOutput:
        """执行动作"""
        self._step_count += 1
        
        # 简单动力学: 状态 = 状态 + 动作
        action_data = action.data[:self.state_dim] if len(action.data) >= self.state_dim else \
                      np.pad(action.data, (0, self.state_dim - len(action.data)))
        
        self._current_state = self._current_state + 0.1 * action_data
        self._current_state = np.clip(self._current_state, -10, 10).astype(np.float32)
        
        # 奖励: 与目标的负距离
        distance = np.linalg.norm(self._current_state - self._goal_state)
        reward = -distance / self.state_dim
        
        # 成功判定
        success = distance < 1.0
        
        # 终止条件
        done = self._step_count >= self.max_episode_steps or success
        if not self.deterministic:
            done = done or (self._rng.random() < 0.01)  # 1% 随机终止
        
        return EnvOutput(
            obs=Observation(images={}, language=""),
            robot_state=RobotState(raw_state=self._current_state.copy()),
            reward=float(reward),
            done=done,
            info={
                "success": success,
                "distance": distance,
                "step": self._step_count,
            },
        )
    
    def seed(self, seed: int):
        """设置随机种子"""
        self._rng = np.random.default_rng(seed)
