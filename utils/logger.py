"""日志工具"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time


class Logger:
    """日志器"""
    
    def __init__(self, log_dir: str = "./logs", name: str = "rl_framework"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self._metrics_history = []
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        self._metrics_history.append(metrics)
        
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items())
        self.info(f"[Step {step}] {metrics_str}")
    
    def save_metrics(self):
        """保存指标历史"""
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self._metrics_history, f, indent=2)
