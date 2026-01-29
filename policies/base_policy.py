"""Policy基类"""
from abc import abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from core.interfaces import PolicyInterface


class BasePolicy(PolicyInterface):
    """
    策略基类
    
    所有框架内策略的基类，提供：
    - act(): 推理动作
    - get_weights() / load_weights(): 权重同步（用于 HIL）
    - reset(): 状态重置
    
    子类只需实现 forward() 和 act()
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._device = torch.device(config.get("device", "cpu"))
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def reset(self) -> None:
        """重置策略状态（有状态策略如 RNN 需要覆盖）"""
        pass
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取权重（用于 HIL 同步）"""
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """加载权重（用于 HIL 同步）"""
        self.load_state_dict(weights)
    
    def to_device(self, device: torch.device) -> "BasePolicy":
        self._device = device
        return self.to(device)
