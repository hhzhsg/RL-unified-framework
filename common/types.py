"""
数据类型定义

框架中所有模块共享的数据结构
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch


@dataclass
class Observation:
    """
    观测数据
    
    Attributes:
        images: 相机图像字典 {camera_name: array}
        state: 本体状态向量
        language: 语言指令（可选）
    """
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    state: Optional[np.ndarray] = None
    language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "images": self.images,
            "state": self.state,
            "language": self.language,
        }


@dataclass
class Action:
    """
    动作数据
    
    Attributes:
        data: 动作数据
        space: 动作空间类型
    """
    data: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    space: str = "joint"  # "joint" | "cartesian" | "delta_ee"


@dataclass
class Transition:
    """
    单步转移数据
    
    Attributes:
        obs: 当前观测
        action: 执行的动作
        reward: 获得的奖励
        next_obs: 下一步观测
        done: 是否结束
        info: 额外信息
        source: 数据来源
    """
    obs: Dict[str, Any]
    action: np.ndarray
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    source: str = "rollout"  # "demo" | "rollout" | "intervention"


@dataclass
class Episode:
    """
    完整轨迹
    
    Attributes:
        transitions: Transition列表
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
    
    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)
    
    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class Batch:
    """
    训练批次
    
    所有字段为Tensor
    """
    obs: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: Dict[str, torch.Tensor]
    done: torch.Tensor
    source: Optional[List[str]] = None
    
    def to(self, device: Union[str, torch.device]) -> "Batch":
        """移动到指定设备"""
        return Batch(
            obs={k: v.to(device) for k, v in self.obs.items()},
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_obs={k: v.to(device) for k, v in self.next_obs.items()},
            done=self.done.to(device),
            source=self.source,
        )
    
    def __len__(self) -> int:
        return self.action.shape[0]
    
    @classmethod
    def from_transitions(cls, transitions: List[Transition]) -> "Batch":
        """从Transition列表构建Batch"""
        if not transitions:
            raise ValueError("Cannot create Batch from empty transitions")
        
        # 提取字段
        obs_keys = transitions[0].obs.keys()
        
        obs = {k: torch.stack([torch.as_tensor(t.obs[k]) for t in transitions]) for k in obs_keys}
        action = torch.stack([torch.as_tensor(t.action) for t in transitions])
        reward = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_obs = {k: torch.stack([torch.as_tensor(t.next_obs[k]) for t in transitions]) for k in obs_keys}
        done = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        source = [t.source for t in transitions]
        
        return cls(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, source=source)
