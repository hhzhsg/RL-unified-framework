"""
VLA-RL 数据类型定义

核心类型:
- Observation: 感知输入 (图像、语言)
- RobotState: 机器人本体状态
- Action: 动作
- Transition: 单步数据 (s, a, r, s', done)
- Episode: 完整轨迹
- Batch: 训练批次
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
import numpy as np


@dataclass
class Observation:
    """感知输入 (给 Policy 用)"""
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # {"front": (H,W,3), ...}
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for name, img in self.images.items():
            result[f"image_{name}"] = img
        if self.language is not None:
            result["language"] = self.language
        return result


@dataclass
class RobotState:
    """机器人状态"""
    joint_pos: np.ndarray                    # 关节位置 (N,)
    joint_vel: Optional[np.ndarray] = None   # 关节速度 (N,)
    ee_pos: Optional[np.ndarray] = None      # 末端位置 (3,)
    ee_quat: Optional[np.ndarray] = None     # 末端姿态 (4,) 四元数
    gripper_pos: Optional[float] = None      # 夹爪开度
    raw_state: Optional[np.ndarray] = None   # 原始完整状态向量 (优先使用)
    
    def to_array(self) -> np.ndarray:
        """转为向量 (用于网络输入)"""
        # 如果有原始状态，直接返回
        if self.raw_state is not None:
            return self.raw_state
        
        # 否则拼接各字段
        parts = [self.joint_pos]
        if self.joint_vel is not None:
            parts.append(self.joint_vel)
        if self.ee_pos is not None:
            parts.append(self.ee_pos)
        if self.ee_quat is not None:
            parts.append(self.ee_quat)
        if self.gripper_pos is not None:
            parts.append(np.array([self.gripper_pos]))
        return np.concatenate(parts)


@dataclass
class Action:
    """动作"""
    data: np.ndarray                         # 动作向量
    space: str = "joint"                     # "joint" | "cartesian" | "delta"


@dataclass
class EnvOutput:
    """环境输出"""
    obs: Observation
    robot_state: RobotState
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """单步数据"""
    obs: Observation
    robot_state: RobotState
    action: Action
    reward: float
    next_obs: Observation
    next_robot_state: RobotState
    done: bool
    source: str = "rollout"                  # "demo" | "rollout" | "intervention"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs": self.obs.to_dict(),
            "robot_state": self.robot_state.to_array(),
            "action": self.action.data,
            "reward": self.reward,
            "next_obs": self.next_obs.to_dict(),
            "next_robot_state": self.next_robot_state.to_array(),
            "done": self.done,
            "source": self.source,
        }


@dataclass
class Episode:
    """完整轨迹"""
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
        """获取数据来源 (取第一个 transition 的 source)"""
        if self.transitions:
            return self.transitions[0].source
        return "unknown"
    
    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """计算每步的 Return"""
        returns = []
        G = 0.0
        for t in reversed(self.transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)
        return returns


@dataclass
class Batch:
    """训练批次数据"""
    obs: Dict[str, np.ndarray]               # 观测 (batched)
    robot_state: np.ndarray                  # 机器人状态 (B, state_dim)
    action: np.ndarray                       # 动作 (B, action_dim) 或 (B, H, action_dim)
    reward: np.ndarray                       # 奖励 (B,)
    next_obs: Dict[str, np.ndarray]          # 下一观测 (batched)
    next_robot_state: np.ndarray             # 下一状态 (B, state_dim)
    done: np.ndarray                         # 是否结束 (B,)
    source: List[str]                        # 数据来源
    returns: Optional[np.ndarray] = None     # Return (B,) 用于 RECAP
    advantage: Optional[np.ndarray] = None   # Advantage (B,) 用于 RECAP
    
    def __len__(self) -> int:
        return len(self.reward)
    
    def to(self, device: str) -> "Batch":
        """移动到指定设备 (用于 PyTorch)"""
        import torch
        
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            elif isinstance(x, dict):
                return {k: to_tensor(v) for k, v in x.items()}
            return x
        
        return Batch(
            obs=to_tensor(self.obs),
            robot_state=to_tensor(self.robot_state),
            action=to_tensor(self.action),
            reward=to_tensor(self.reward),
            next_obs=to_tensor(self.next_obs),
            next_robot_state=to_tensor(self.next_robot_state),
            done=to_tensor(self.done),
            source=self.source,
            returns=to_tensor(self.returns) if self.returns is not None else None,
            advantage=to_tensor(self.advantage) if self.advantage is not None else None,
        )
