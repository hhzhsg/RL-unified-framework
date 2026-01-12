"""
VLA-RL 核心数据类型定义

定义框架中所有模块共享的数据结构:
- Observation: 观测数据 (图像 + 语言)
- RobotState: 机器人状态
- Action: 动作
- EnvOutput: 环境输出
- Transition: 单步数据
- Episode: 完整轨迹
- Batch: 训练批次
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator
import numpy as np
import torch


@dataclass
class Observation:
    """
    观测数据
    
    Attributes:
        images: 相机图像字典 {camera_name: (H, W, C) ndarray}
        language: 语言指令
    """
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"images": self.images, "language": self.language}


@dataclass
class RobotState:
    """
    机器人状态
    
    Attributes:
        joint_pos: 关节位置 (7,)
        joint_vel: 关节速度 (7,)
        ee_pos: 末端位置 (3,)
        ee_quat: 末端四元数 (4,)
        gripper: 夹爪状态
        raw_state: 原始状态向量 (用于简单环境)
    """
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ee_quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32))
    gripper: float = 0.0
    raw_state: Optional[np.ndarray] = None
    
    def to_array(self) -> np.ndarray:
        """转换为单一数组"""
        if self.raw_state is not None:
            return self.raw_state
        return np.concatenate([
            self.joint_pos, self.joint_vel, 
            self.ee_pos, self.ee_quat, 
            [self.gripper]
        ])


@dataclass
class Action:
    """
    动作
    
    Attributes:
        data: 动作数据
        space: 动作空间类型 ("joint" | "cartesian" | "delta")
    """
    data: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    space: str = "joint"


@dataclass
class EnvOutput:
    """
    环境输出
    
    Attributes:
        obs: 观测
        robot_state: 机器人状态
        reward: 奖励
        done: 是否结束
        info: 额外信息
    """
    obs: Observation
    robot_state: RobotState
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """
    单步数据
    
    Attributes:
        obs: 当前观测
        robot_state: 当前机器人状态
        action: 执行的动作
        reward: 获得的奖励
        next_obs: 下一步观测
        next_robot_state: 下一步状态
        done: 是否结束
        source: 数据来源 ("demo" | "rollout" | "intervention")
    """
    obs: Observation
    robot_state: RobotState
    action: Action
    reward: float
    next_obs: Observation
    next_robot_state: RobotState
    done: bool
    source: str = "rollout"


@dataclass
class Episode:
    """
    完整轨迹
    
    Attributes:
        transitions: Transition 列表
        success: 是否成功
        task_id: 任务标识
        metadata: 元数据
    """
    transitions: List[Transition] = field(default_factory=list)
    success: bool = False
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)
    
    def add(self, transition: Transition):
        self.transitions.append(transition)
    
    @property
    def source(self) -> str:
        if self.transitions:
            return self.transitions[0].source
        return "unknown"


@dataclass
class Batch:
    """
    训练批次
    
    所有字段为 Tensor，shape: (batch_size, ...)
    """
    obs: Dict[str, torch.Tensor]
    robot_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: Dict[str, torch.Tensor]
    next_robot_state: torch.Tensor
    done: torch.Tensor
    source: List[str] = field(default_factory=list)
    
    def to(self, device: str) -> "Batch":
        """移动到指定设备"""
        return Batch(
            obs={k: v.to(device) for k, v in self.obs.items()},
            robot_state=self.robot_state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_obs={k: v.to(device) for k, v in self.next_obs.items()},
            next_robot_state=self.next_robot_state.to(device),
            done=self.done.to(device),
            source=self.source,
        )
    
    def __len__(self) -> int:
        return self.robot_state.shape[0]
