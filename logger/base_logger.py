"""
Logger 基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import numpy as np


class LogLevel(Enum):
    """日志级别"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class BaseLogger(ABC):
    """
    日志基类
    
    所有 Logger 需要实现:
    - log_scalar(): 记录标量
    - log_scalars(): 批量记录标量
    - close(): 关闭资源
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._step = 0
    
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        记录单个标量
        
        Args:
            tag: 指标名称 (支持 "/" 分层，如 "train/loss")
            value: 指标值
            step: 步数（None 则使用内部计数）
        """
        pass
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        批量记录标量
        
        Args:
            metrics: {tag: value} 字典
            step: 步数
            prefix: 标签前缀
        """
        for tag, value in metrics.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            self.log_scalar(full_tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """记录直方图（子类可选实现）"""
        pass
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """
        记录图像
        
        Args:
            tag: 图像名称
            image: numpy 数组 (H, W, C) 或 (H, W)
            step: 步数
        """
        pass
    
    def log_video(self, tag: str, frames: List[np.ndarray], step: Optional[int] = None, fps: int = 30):
        """
        记录视频
        
        Args:
            tag: 视频名称
            frames: 帧列表 [(H, W, C), ...]
            step: 步数
            fps: 帧率
        """
        pass
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """记录文本"""
        pass
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        pass
    
    def info(self, message: str):
        """输出 INFO 级别消息"""
        self._log_message(message, LogLevel.INFO)
    
    def warning(self, message: str):
        """输出 WARNING 级别消息"""
        self._log_message(message, LogLevel.WARNING)
    
    def error(self, message: str):
        """输出 ERROR 级别消息"""
        self._log_message(message, LogLevel.ERROR)
    
    def debug(self, message: str):
        """输出 DEBUG 级别消息"""
        self._log_message(message, LogLevel.DEBUG)
    
    def _log_message(self, message: str, level: LogLevel):
        """内部消息记录方法（子类可重写）"""
        pass
    
    def set_step(self, step: int):
        """设置当前步数"""
        self._step = step
    
    def step(self) -> int:
        """获取当前步数"""
        return self._step
    
    def increment_step(self, delta: int = 1):
        """增加步数"""
        self._step += delta
    
    @abstractmethod
    def close(self):
        """关闭 Logger，释放资源"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
