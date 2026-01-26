"""算法基类"""
from abc import abstractmethod
from typing import Dict, Any
import torch
from core.interfaces import AlgorithmInterface, OffPolicyAlgorithmInterface


class BaseAlgorithm(AlgorithmInterface):
    """算法基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._train_step = 0
        # 自动检测设备，如果 CUDA 不可用则使用 CPU
        device_str = config.get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
    
    @property
    def train_step(self) -> int:
        return self._train_step
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        raise NotImplementedError


class BaseOffPolicyAlgorithm(BaseAlgorithm, OffPolicyAlgorithmInterface):
    """Off-Policy算法基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._tau = config.get("tau", 0.005)
    
    @property
    def tau(self) -> float:
        return self._tau
    
    @abstractmethod
    def update_target(self) -> None:
        pass
