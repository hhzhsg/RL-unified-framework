"""
Composite Logger - 组合多个 Logger
"""
from typing import Dict, Any, Optional, List
import numpy as np

from .base_logger import BaseLogger, LogLevel


class CompositeLogger(BaseLogger):
    """
    组合多个 Logger
    
    将所有操作转发给子 Logger
    """
    
    def __init__(self, loggers: List[BaseLogger]):
        super().__init__("composite")
        self.loggers = loggers
    
    def add(self, logger: BaseLogger):
        """添加 Logger"""
        self.loggers.append(logger)
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        for logger in self.loggers:
            logger.log_scalars(metrics, step, prefix)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_image(tag, image, step)
    
    def log_video(self, tag: str, frames: List[np.ndarray], step: Optional[int] = None, fps: int = 30):
        for logger in self.loggers:
            logger.log_video(tag, frames, step, fps)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_text(tag, text, step)
    
    def log_config(self, config: Dict[str, Any]):
        for logger in self.loggers:
            logger.log_config(config)
    
    def _log_message(self, message: str, level: LogLevel):
        for logger in self.loggers:
            logger._log_message(message, level)
    
    def set_step(self, step: int):
        self._step = step
        for logger in self.loggers:
            logger.set_step(step)
    
    def close(self):
        for logger in self.loggers:
            logger.close()
