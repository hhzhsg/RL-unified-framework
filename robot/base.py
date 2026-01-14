"""
机器人适配器基类

定义不同机器人平台的数据适配接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotSpec:
    """机器人规格"""
    name: str
    state_dim: int
    action_dim: int
    camera_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]


class BaseRobotAdapter(ABC):
    """
    机器人适配器基类
    
    负责:
    - HDF5 数据格式转换
    - 状态/动作归一化
    - 相机数据处理
    """
    
    def __init__(self, spec: RobotSpec):
        self.spec = spec
    
    @property
    def name(self) -> str:
        return self.spec.name
    
    @property
    def state_dim(self) -> int:
        return self.spec.state_dim
    
    @property
    def action_dim(self) -> int:
        return self.spec.action_dim
    
    @property
    def camera_keys(self) -> List[str]:
        return self.spec.camera_keys
    
    @abstractmethod
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理原始数据
        
        Args:
            raw_data: HDF5 原始数据
            
        Returns:
            标准化数据格式
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: np.ndarray) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            model_output: 模型输出动作
            
        Returns:
            机器人执行动作
        """
        pass
    
    def get_state_mapping(self) -> Dict[str, str]:
        """获取状态字段映射 (标准名 -> HDF5 路径)"""
        return {}
    
    def get_action_mapping(self) -> Dict[str, str]:
        """获取动作字段映射"""
        return {}
