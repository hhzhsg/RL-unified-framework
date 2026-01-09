"""
VLA-RL Algorithm 基类

所有算法继承此类，实现 train_step() 方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from model import ModelGroup
from data import Batch
from config import AlgorithmConfig


class BaseAlgorithm(ABC):
    """
    算法基类
    
    所有算法实现需要继承此类并实现:
    - train_step(): 单步训练
    
    可选:
    - REQUIRED_MODELS: 声明需要的模型列表
    - _validate_model_group(): 验证 model_group
    """
    
    # 子类可覆盖，声明需要的模型
    REQUIRED_MODELS: List[str] = []
    
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
    
    def _validate_model_group(self):
        """验证 model_group 包含所需模型 (子类可覆盖)"""
        if not self.REQUIRED_MODELS:
            return
        
        missing = [name for name in self.REQUIRED_MODELS if name not in self.model_group]
        if missing:
            raise ValueError(
                f"{self.name} requires models {self.REQUIRED_MODELS}, "
                f"but missing: {missing}. "
                f"Available: {self.model_group.model_names}"
            )
    
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
