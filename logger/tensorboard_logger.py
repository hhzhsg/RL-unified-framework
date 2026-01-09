"""
TensorBoard Logger
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from .base_logger import BaseLogger, LogLevel


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard 日志记录
    
    支持：
    - 标量
    - 直方图
    - 图像
    - 视频（需要 moviepy）
    """
    
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        super().__init__(name)
        
        self.log_dir = Path(log_dir) / "tensorboard" / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 延迟导入 TensorBoard
        self._writer = None
        self._init_writer()
    
    def _init_writer(self):
        """初始化 SummaryWriter"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            print("[Warning] TensorBoard not available. Install with: pip install tensorboard")
            self._writer = None
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        self._writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        for tag, value in metrics.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            self._writer.add_scalar(full_tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        self._writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        # TensorBoard 期望 (C, H, W) 格式
        if len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))
        
        self._writer.add_image(tag, image, step)
    
    def log_video(self, tag: str, frames: List[np.ndarray], step: Optional[int] = None, fps: int = 30):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        # TensorBoard 期望 (N, T, C, H, W) 格式
        video = np.stack(frames)  # (T, H, W, C)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)
        video = np.expand_dims(video, 0)  # (1, T, C, H, W)
        
        self._writer.add_video(tag, video, step, fps=fps)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        if self._writer is None:
            return
        
        if step is None:
            step = self._step
        
        self._writer.add_text(tag, text, step)
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置为文本"""
        if self._writer is None:
            return
        
        import json
        config_text = json.dumps(config, indent=2, default=str)
        self._writer.add_text("config", f"```\n{config_text}\n```", 0)
    
    def log_graph(self, model, input_sample):
        """记录模型图"""
        if self._writer is None:
            return
        
        try:
            self._writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"[Warning] Failed to log model graph: {e}")
    
    def flush(self):
        """刷新缓冲区"""
        if self._writer is not None:
            self._writer.flush()
    
    def close(self):
        if self._writer is not None:
            self._writer.close()
