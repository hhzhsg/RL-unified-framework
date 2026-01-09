"""
WandB Logger
"""
from typing import Dict, Any, Optional, List
import numpy as np

from .base_logger import BaseLogger, LogLevel


class WandBLogger(BaseLogger):
    """
    Weights & Biases 日志记录
    
    支持：
    - 标量
    - 直方图
    - 图像
    - 视频
    - 配置
    - 模型追踪
    """
    
    def __init__(self, 
                 project: str = "vla-rl",
                 name: str = None,
                 config: Dict[str, Any] = None,
                 tags: List[str] = None,
                 notes: str = None,
                 mode: str = "online"):  # "online", "offline", "disabled"
        """
        Args:
            project: WandB 项目名
            name: 运行名称
            config: 配置字典
            tags: 标签列表
            notes: 运行说明
            mode: 运行模式
        """
        super().__init__(name or "run")
        
        self._run = None
        self._init_wandb(project, name, config, tags, notes, mode)
    
    def _init_wandb(self, project, name, config, tags, notes, mode):
        """初始化 WandB"""
        try:
            import wandb
            
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                mode=mode,
                reinit=True,
            )
            
            # 保存 wandb 模块引用
            self._wandb = wandb
            
        except ImportError:
            print("[Warning] WandB not available. Install with: pip install wandb")
            self._run = None
            self._wandb = None
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        self._wandb.log({tag: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        log_dict = {}
        for tag, value in metrics.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            log_dict[full_tag] = value
        
        self._wandb.log(log_dict, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        self._wandb.log({tag: self._wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None, caption: str = None):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        wandb_image = self._wandb.Image(image, caption=caption)
        self._wandb.log({tag: wandb_image}, step=step)
    
    def log_video(self, tag: str, frames: List[np.ndarray], step: Optional[int] = None, fps: int = 30):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        # WandB 期望 (T, H, W, C) 格式
        video = np.stack(frames)
        wandb_video = self._wandb.Video(video, fps=fps)
        self._wandb.log({tag: wandb_video}, step=step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        if self._run is None:
            return
        
        if step is None:
            step = self._step
        
        # WandB 使用 Table 记录文本
        table = self._wandb.Table(columns=["text"])
        table.add_data(text)
        self._wandb.log({tag: table}, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """更新配置"""
        if self._run is None:
            return
        
        self._wandb.config.update(config)
    
    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "model"):
        """记录 artifact（如模型权重）"""
        if self._run is None:
            return
        
        artifact = self._wandb.Artifact(name, type=artifact_type)
        artifact.add_file(artifact_path)
        self._run.log_artifact(artifact)
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """监控模型梯度"""
        if self._run is None:
            return
        
        self._wandb.watch(model, log=log, log_freq=log_freq)
    
    def _log_message(self, message: str, level: LogLevel):
        """记录消息到 Notes"""
        if self._run is None:
            return
        
        # WandB 没有直接的消息 API，使用 alert 或忽略
        if level == LogLevel.ERROR:
            self._wandb.alert(
                title="Error",
                text=message,
                level=self._wandb.AlertLevel.ERROR,
            )
    
    def close(self):
        if self._run is not None:
            self._run.finish()
