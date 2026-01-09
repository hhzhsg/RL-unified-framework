"""
DummyEnv: 调试用虚拟环境

支持:
- 可配置状态/动作维度
- 确定性模式（用于调试）
- 可自定义奖励函数
- 简单的目标达成任务
"""
from typing import Optional, Callable
import numpy as np

from .base_env import BaseEnv
from data import Observation, RobotState, Action, EnvOutput
from config import EnvConfig


class DummyEnv(BaseEnv):
    """
    虚拟环境，用于调试和测试
    
    默认任务：让状态接近目标位置
    奖励：-||state - target||^2
    成功：当 ||state - target|| < threshold
    """
    
    def __init__(self, 
                 config: Optional[EnvConfig] = None,
                 deterministic: bool = False,
                 reward_fn: Optional[Callable] = None):
        """
        Args:
            config: 环境配置
            deterministic: 是否确定性（用于调试）
            reward_fn: 自定义奖励函数 (state, action, next_state) -> reward
        """
        if config is None:
            config = EnvConfig(name="dummy")
        super().__init__(config)
        
        self.deterministic = deterministic
        self.reward_fn = reward_fn
        
        # 目标位置（用于奖励计算）
        self._target_state: Optional[np.ndarray] = None
        self._current_state: Optional[np.ndarray] = None
        self._current_robot_state: Optional[RobotState] = None
        
        # 成功阈值
        self._success_threshold = 0.5
    
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        self._step_count = 0
        
        # 初始化状态
        if self.deterministic:
            np.random.seed(42)
        
        # 使用 raw_state 统一管理维度
        self._current_state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        self._target_state = np.zeros(self.state_dim, dtype=np.float32)  # 目标是原点
        
        self._current_robot_state = RobotState(
            joint_pos=np.zeros(7, dtype=np.float32),  # placeholder
            raw_state=self._current_state.copy()  # 使用 raw_state
        )
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=self._current_robot_state,
            reward=0.0,
            done=False,
            info={"task_id": task_id or "reach_target"},
        )
    
    def step(self, action: Action) -> EnvOutput:
        self._step_count += 1
        
        # 获取动作数据
        action_data = np.asarray(action.data).flatten()
        
        # 确保动作维度匹配
        if len(action_data) < self.action_dim:
            action_data = np.pad(action_data, (0, self.action_dim - len(action_data)))
        else:
            action_data = action_data[:self.action_dim]
        
        # 状态转移：state' = state + action * scale + noise
        scale = 0.1
        if self.deterministic:
            noise = np.zeros_like(self._current_state)
        else:
            noise = np.random.randn(self.state_dim).astype(np.float32) * 0.01
        
        # 动作影响状态（简单线性映射）
        # 如果 action_dim != state_dim，只影响前 min(action_dim, state_dim) 个维度
        affected_dims = min(self.action_dim, self.state_dim)
        state_delta = np.zeros(self.state_dim, dtype=np.float32)
        state_delta[:affected_dims] = action_data[:affected_dims] * scale
        
        self._current_state = self._current_state + state_delta + noise
        
        # 计算奖励
        if self.reward_fn:
            reward = self.reward_fn(self._current_state, action_data, self._target_state)
        else:
            # 默认奖励：负的距离平方
            dist = np.linalg.norm(self._current_state - self._target_state)
            reward = -dist * 0.1
        
        # 更新 robot_state
        self._current_robot_state = RobotState(
            joint_pos=np.zeros(7, dtype=np.float32),
            raw_state=self._current_state.copy()
        )
        
        # 判断成功和结束
        dist_to_target = np.linalg.norm(self._current_state - self._target_state)
        success = dist_to_target < self._success_threshold
        done = self._step_count >= self.config.max_episode_steps or success
        
        if success:
            reward = 10.0  # 成功奖励
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=self._current_robot_state,
            reward=reward,
            done=done,
            info={
                "success": success, 
                "step": self._step_count,
                "dist_to_target": dist_to_target,
            },
        )
    
    def _get_observation(self) -> Observation:
        """生成观测"""
        images = {}
        
        if not self.deterministic:
            for cam_name in self.config.obs_cameras:
                images[cam_name] = np.random.randint(
                    0, 256,
                    (self.config.image_size, self.config.image_size, 3),
                    dtype=np.uint8
                )
        else:
            # 确定性模式：黑色图像
            for cam_name in self.config.obs_cameras:
                images[cam_name] = np.zeros(
                    (self.config.image_size, self.config.image_size, 3),
                    dtype=np.uint8
                )
        
        return Observation(
            images=images,
            language="reach the target position",
        )
    
    def set_target(self, target: np.ndarray):
        """设置目标位置"""
        self._target_state = np.asarray(target, dtype=np.float32)
