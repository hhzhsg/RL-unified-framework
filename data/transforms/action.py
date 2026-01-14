"""
动作转换

动作空间相关的转换:
- NormalizeAction: 动作归一化
- DeltaAction: 绝对动作转增量动作
- ActionToTensor: 动作转 Tensor
"""
from typing import Dict, Any, Optional, List
import numpy as np

from .base import BaseTransform


class NormalizeAction(BaseTransform):
    """
    动作归一化
    
    将动作归一化到 [-1, 1] 范围
    """
    
    def __init__(self, 
                 low: np.ndarray | List[float],
                 high: np.ndarray | List[float]):
        """
        Args:
            low: 动作下界
            high: 动作上界
        """
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.range = self.high - self.low
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in data:
            return data
        
        action = data["action"]
        normalized = 2.0 * (action - self.low) / self.range - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        data["action"] = normalized
        
        return data
    
    def unnormalize(self, action: np.ndarray) -> np.ndarray:
        """反归一化"""
        return self.low + (action + 1.0) * self.range / 2.0


class DeltaAction(BaseTransform):
    """
    绝对动作转增量动作
    
    delta_action = action - state
    """
    
    def __init__(self, 
                 action_key: str = "action",
                 state_key: str = "state",
                 mask: Optional[List[bool]] = None):
        """
        Args:
            action_key: 动作数据的 key
            state_key: 状态数据的 key
            mask: 哪些维度使用增量 (True=增量, False=绝对)
        """
        self.action_key = action_key
        self.state_key = state_key
        self.mask = mask
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.action_key not in data or self.state_key not in data:
            return data
        
        action = data[self.action_key]
        state = data[self.state_key]
        
        # 确保维度匹配
        min_dim = min(len(action), len(state))
        delta = action.copy()
        
        if self.mask is None:
            delta[:min_dim] = action[:min_dim] - state[:min_dim]
        else:
            for i, use_delta in enumerate(self.mask):
                if use_delta and i < min_dim:
                    delta[i] = action[i] - state[i]
        
        data[self.action_key] = delta
        return data


class NormalizeState(BaseTransform):
    """
    状态归一化
    
    使用均值和标准差归一化状态
    """
    
    def __init__(self,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 eps: float = 1e-8):
        """
        Args:
            mean: 均值 (None 表示使用运行时计算)
            std: 标准差 (None 表示使用运行时计算)
            eps: 防止除零
        """
        self.mean = mean
        self.std = std
        self.eps = eps
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "state" not in data:
            return data
        
        state = data["state"]
        
        if self.mean is not None and self.std is not None:
            state = (state - self.mean) / (self.std + self.eps)
        
        data["state"] = state
        return data
    
    def fit(self, states: np.ndarray):
        """从数据计算均值和标准差"""
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0)


class ActionToTensor(BaseTransform):
    """动作转 PyTorch Tensor"""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        
        if "action" in data:
            action = data["action"]
            if isinstance(action, np.ndarray):
                data["action"] = torch.from_numpy(action.copy()).float()
        
        if "state" in data:
            state = data["state"]
            if isinstance(state, np.ndarray):
                data["state"] = torch.from_numpy(state.copy()).float()
        
        return data
