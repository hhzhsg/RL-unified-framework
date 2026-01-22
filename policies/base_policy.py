"""Policy基类"""
from abc import abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from core.interfaces import PolicyInterface


class BasePolicy(PolicyInterface):
    """策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._device = torch.device(config.get("device", "cpu"))
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def reset(self) -> None:
        pass
    
    def to_device(self, device: torch.device) -> "BasePolicy":
        self._device = device
        return self.to(device)
