"""
日志工具

简单的日志记录功能
"""
import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    设置 logger
    
    Args:
        name: logger 名称
        level: 日志级别
        log_file: 日志文件路径 (可选)
        
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler (可选)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """
    指标记录器
    
    记录训练过程中的各种指标
    """
    
    def __init__(self, name: str = "metrics"):
        self.name = name
        self.history: Dict[str, list] = {}
        self._step = 0
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        if step is not None:
            self._step = step
        
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((self._step, value))
    
    def get_latest(self, key: str) -> Optional[float]:
        """获取最新值"""
        if key not in self.history or not self.history[key]:
            return None
        return self.history[key][-1][1]
    
    def get_average(self, key: str, window: int = 100) -> Optional[float]:
        """获取滑动平均"""
        if key not in self.history or not self.history[key]:
            return None
        values = [v for _, v in self.history[key][-window:]]
        return sum(values) / len(values)
    
    def summary(self) -> Dict[str, float]:
        """获取所有指标的最新值"""
        return {k: v[-1][1] for k, v in self.history.items() if v}
