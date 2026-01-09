"""
File Logger - JSON/CSV 文件记录
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import csv
from datetime import datetime

from .base_logger import BaseLogger, LogLevel


class FileLogger(BaseLogger):
    """
    基础文件日志
    """
    
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        super().__init__(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建带时间戳的日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)


class JSONLogger(FileLogger):
    """
    JSON 文件日志
    
    每行一个 JSON 对象，便于后续分析
    """
    
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        super().__init__(log_dir, name)
        
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"
        self.messages_file = self.run_dir / "messages.jsonl"
        
        # 缓冲区
        self._buffer: List[Dict] = []
        self._buffer_size = 100
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self._step
        
        record = {
            "step": step,
            "tag": tag,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        self._buffer.append(record)
        
        if len(self._buffer) >= self._buffer_size:
            self._flush()
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        if step is None:
            step = self._step
        
        record = {
            "step": step,
            "metrics": {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()},
            "timestamp": datetime.now().isoformat(),
        }
        self._buffer.append(record)
        
        if len(self._buffer) >= self._buffer_size:
            self._flush()
    
    def log_config(self, config: Dict[str, Any]):
        """保存配置"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _log_message(self, message: str, level: LogLevel):
        record = {
            "level": level.name,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.messages_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def _flush(self):
        """写入缓冲区"""
        if not self._buffer:
            return
        
        with open(self.metrics_file, 'a') as f:
            for record in self._buffer:
                f.write(json.dumps(record) + '\n')
        
        self._buffer.clear()
    
    def close(self):
        self._flush()


class CSVLogger(FileLogger):
    """
    CSV 文件日志
    
    适合表格形式的数据
    """
    
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        super().__init__(log_dir, name)
        
        self.metrics_file = self.run_dir / "metrics.csv"
        self._columns: List[str] = ["step", "timestamp"]
        self._rows: List[Dict] = []
        self._current_row: Dict[str, Any] = {}
        self._header_written = False
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self._step
        
        # 添加新列
        if tag not in self._columns:
            self._columns.append(tag)
        
        # 更新当前行
        self._current_row["step"] = step
        self._current_row["timestamp"] = datetime.now().isoformat()
        self._current_row[tag] = value
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        if step is None:
            step = self._step
        
        self._current_row["step"] = step
        self._current_row["timestamp"] = datetime.now().isoformat()
        
        for tag, value in metrics.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            if full_tag not in self._columns:
                self._columns.append(full_tag)
            self._current_row[full_tag] = value
        
        # 提交当前行
        self._rows.append(self._current_row)
        self._current_row = {}
        
        if len(self._rows) >= 100:
            self._flush()
    
    def _flush(self):
        """写入文件"""
        if not self._rows:
            return
        
        mode = 'a' if self._header_written else 'w'
        with open(self.metrics_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._columns, extrasaction='ignore')
            
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            
            writer.writerows(self._rows)
        
        self._rows.clear()
    
    def close(self):
        if self._current_row:
            self._rows.append(self._current_row)
        self._flush()
