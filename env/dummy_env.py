"""
VLA-RL DummyEnv - 测试用虚拟环境

用于验证训练流程，不需要真实环境
"""
from typing import Optional, Callable
import numpy as np

from .base_env import BaseEnv
from config import EnvConfig
from data import Action, EnvOutput, Observation, RobotState


class DummyEnv(BaseEnv):
    """
    虚拟环境
    
    简单的目标到达任务:
    - 状态: 当前位置
    - 动作: 位移
    - 奖励: 负的距离
    - 成功: 距离目标小于阈值
    """
    
    def __init__(self, config: EnvConfig, 
                 reward_fn: Optional[Callable] = None,
                 deterministic: bool = False):
        """
        Args:
            config: 环境配置
            reward_fn: 自定义奖励函数 fn(state, action, next_state) -> float
            deterministic: 是否确定性 (无噪声)
        """
        super().__init__(config)
        
        self.reward_fn = reward_fn
        self.deterministic = deterministic
        
        # 状态
        self._current_state = np.zeros(self.state_dim, dtype=np.float32)
        self._target_state = np.zeros(self.state_dim, dtype=np.float32)
        self._success_threshold = 0.5
    
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        """重置环境"""
        self._step_count = 0
        
        # 初始化状态
        if self.deterministic:
            self._current_state = np.zeros(self.state_dim, dtype=np.float32)
            self._target_state = np.ones(self.state_dim, dtype=np.float32)
        else:
            self._current_state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
            self._target_state = np.random.randn(self.state_dim).astype(np.float32)
        
        robot_state = RobotState(
            joint_pos=np.zeros(7, dtype=np.float32),
            raw_state=self._current_state.copy()
        )
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=robot_state,
            reward=0.0,
            done=False,
            info={"task_id": task_id or "reach_target"},
        )
    
    def step(self, action: Action) -> EnvOutput:
        """执行动作"""
        self._step_count += 1
        
        # 获取动作数据
        action_data = np.asarray(action.data).flatten()
        
        # 确保动作维度匹配
        if len(action_data) < self.action_dim:
            action_data = np.pad(action_data, (0, self.action_dim - len(action_data)))
        else:
            action_data = action_data[:self.action_dim]
        
        # 状态转移: state' = state + action * scale + noise
        scale = 0.1
        if self.deterministic:
            noise = np.zeros_like(self._current_state)
        else:
            noise = np.random.randn(self.state_dim).astype(np.float32) * 0.01
        
        # 动作影响状态
        affected_dims = min(self.action_dim, self.state_dim)
        state_delta = np.zeros(self.state_dim, dtype=np.float32)
        state_delta[:affected_dims] = action_data[:affected_dims] * scale
        
        self._current_state = self._current_state + state_delta + noise
        
        # 计算奖励
        if self.reward_fn:
            reward = self.reward_fn(self._current_state, action_data, self._target_state)
        else:
            dist = np.linalg.norm(self._current_state - self._target_state)
            reward = -dist * 0.1
        
        # 判断成功和结束
        dist_to_target = np.linalg.norm(self._current_state - self._target_state)
        success = dist_to_target < self._success_threshold
        done = self._step_count >= self.config.max_episode_steps or success
        
        if success:
            reward = 10.0
        
        robot_state = RobotState(
            joint_pos=np.zeros(7, dtype=np.float32),
            raw_state=self._current_state.copy()
        )
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=robot_state,
            reward=float(reward),
            done=done,
            info={"success": success, "step": self._step_count, "dist": dist_to_target},
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
        return Observation(images=images, language="reach the target")
    
    def set_target(self, target: np.ndarray):
        """设置目标"""
        self._target_state = np.asarray(target, dtype=np.float32)
