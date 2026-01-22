"""
System接口定义

系统级接口，定义组件间的协作规范
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ComponentCapability:
    """组件能力声明"""
    supports_images: bool = False
    supports_proprioception: bool = True
    supports_language: bool = False
    action_space: str = "continuous"  # "continuous" | "discrete" | "mixed"
    requires_target_network: bool = False
    requires_replay_buffer: bool = True
    supports_intervention: bool = False


class SystemInterface(ABC):
    """系统接口"""
    
    @abstractmethod
    def build(self, config: Dict[str, Any]) -> None:
        """
        构建系统
        
        Args:
            config: 系统配置
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """运行系统"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止系统"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        pass


class LoopInterface(ABC):
    """运行循环接口"""
    
    @abstractmethod
    def run(self, num_steps: int) -> Dict[str, Any]:
        """
        运行循环
        
        Args:
            num_steps: 运行步数
            
        Returns:
            运行结果统计
        """
        pass
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """执行单步"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止循环"""
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """是否正在运行"""
        pass


class SyncInterface(ABC):
    """同步接口"""
    
    @abstractmethod
    def push(self, data: Dict[str, Any], tag: str = "default") -> None:
        """推送数据"""
        pass
    
    @abstractmethod
    def pull(self, tag: str = "default") -> Optional[Dict[str, Any]]:
        """拉取数据"""
        pass
    
    @abstractmethod
    def get_version(self, tag: str = "default") -> int:
        """获取版本号"""
        pass
