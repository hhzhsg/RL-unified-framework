"""
DummyEnv: 调试用虚拟环境
"""
from typing import Optional
import numpy as np

from .base_env import BaseEnv
from data import Observation, RobotState, Action, EnvOutput
from config import EnvConfig


class DummyEnv(BaseEnv):
    """
    虚拟环境，用于调试和测试
    生成随机数据
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        if config is None:
            config = EnvConfig(name="dummy")
        super().__init__(config)
        
        self._current_robot_state = None
    
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        self._step_count = 0
        
        # 随机初始化机器人状态
        self._current_robot_state = RobotState(
            joint_pos=np.random.randn(7).astype(np.float32) * 0.1,
            joint_vel=np.zeros(7, dtype=np.float32),
            ee_pos=np.array([0.5, 0.0, 0.3], dtype=np.float32),
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gripper_pos=0.0,
        )
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=self._current_robot_state,
            reward=0.0,
            done=False,
            info={"task_id": task_id or "default"},
        )
    
    def step(self, action: Action) -> EnvOutput:
        self._step_count += 1
        
        # 简单的状态转移
        action_data = np.asarray(action.data).flatten()[:7]
        noise = np.random.randn(7).astype(np.float32) * 0.01
        
        new_joint_pos = self._current_robot_state.joint_pos + action_data * 0.1 + noise
        
        self._current_robot_state = RobotState(
            joint_pos=new_joint_pos,
            joint_vel=action_data,
            ee_pos=self._current_robot_state.ee_pos + np.random.randn(3).astype(np.float32) * 0.01,
            ee_quat=self._current_robot_state.ee_quat,
            gripper_pos=float(action_data[-1]) if len(action_data) > 0 else 0.0,
        )
        
        # 随机奖励
        reward = -np.sum(np.abs(new_joint_pos)) * 0.01
        
        # 结束条件
        done = self._step_count >= self.config.max_episode_steps
        success = False
        
        # 10% 概率提前成功
        if np.random.rand() < 0.1:
            done = True
            success = True
            reward = 1.0
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=self._current_robot_state,
            reward=reward,
            done=done,
            info={"success": success, "step": self._step_count},
        )
    
    def _get_observation(self) -> Observation:
        """生成随机观测"""
        images = {}
        for cam_name in self.config.obs_cameras:
            images[cam_name] = np.random.randint(
                0, 256,
                (self.config.image_size, self.config.image_size, 3),
                dtype=np.uint8
            )
        
        return Observation(
            images=images,
            language="pick up the object",
        )
