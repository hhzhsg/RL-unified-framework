"""
Algorithm接口定义

训练算法必须遵循此接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class AlgorithmInterface(ABC):
    """算法接口"""
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步更新
        
        Args:
            batch: 训练批量数据
            
        Returns:
            metrics: 训练指标字典
        """
        pass
    
    @abstractmethod
    def get_policy(self) -> "PolicyInterface":
        """获取当前策略（用于推理）"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存算法状态"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """加载算法状态"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称"""
        pass
    
    @property
    @abstractmethod
    def train_step(self) -> int:
        """当前训练步数"""
        pass


class OffPolicyAlgorithmInterface(AlgorithmInterface):
    """Off-Policy算法接口"""
    
    @abstractmethod
    def update_target(self) -> None:
        """更新目标网络"""
        pass
    
    @property
    @abstractmethod
    def tau(self) -> float:
        """软更新系数"""
        pass


class OnPolicyAlgorithmInterface(AlgorithmInterface):
    """On-Policy算法接口"""
    
    @abstractmethod
    def compute_advantages(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """计算优势函数"""
        pass
