"""
VLA-RL Algorithm 基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from model import ModelGroup
from data import Batch
from config import AlgorithmConfig


class BaseAlgorithm(ABC):
    """
    算法基类
    
    所有算法实现需要继承此类并实现:
    - train_step(): 单步训练
    """
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig):
        self.model_group = model_group
        self.config = config
        self._train_step_count = 0
    
    @abstractmethod
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """
        训练一步
        
        Args:
            batch: 训练批次数据
            
        Returns:
            包含各项 loss 和 metrics 的字典
        """
        pass
    
    @property
    def name(self) -> str:
        """算法名称"""
        return self.__class__.__name__
    
    @property
    def train_steps(self) -> int:
        """已训练步数"""
        return self._train_step_count
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态 (用于保存)"""
        return {
            "model_group": self.model_group.state_dict(),
            "train_steps": self._train_step_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态"""
        self.model_group.load_state_dict(state_dict["model_group"])
        self._train_step_count = state_dict["train_steps"]
