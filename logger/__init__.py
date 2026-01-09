"""
VLA-RL Logger 模块

统一的日志管理，支持：
- Console 输出（带颜色）
- TensorBoard
- WandB
- JSON/CSV 文件
- 视频录制
"""
from .base_logger import BaseLogger, LogLevel
from .console_logger import ConsoleLogger
from .file_logger import FileLogger, JSONLogger, CSVLogger
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandBLogger
from .composite_logger import CompositeLogger
from .metrics import MetricsTracker, EpisodeMetrics

# 便捷创建函数
def create_logger(
    name: str = "train",
    log_dir: str = "./logs",
    use_console: bool = True,
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    use_file: bool = True,
    wandb_project: str = None,
    wandb_config: dict = None,
    **kwargs
) -> BaseLogger:
    """
    创建 Logger
    
    Args:
        name: 实验名称
        log_dir: 日志目录
        use_console: 是否使用控制台输出
        use_tensorboard: 是否使用 TensorBoard
        use_wandb: 是否使用 WandB
        use_file: 是否保存到文件
        wandb_project: WandB 项目名
        wandb_config: WandB 配置
        
    Returns:
        CompositeLogger 实例
    """
    loggers = []
    
    if use_console:
        loggers.append(ConsoleLogger(name=name))
    
    if use_file:
        loggers.append(JSONLogger(log_dir=log_dir, name=name))
    
    if use_tensorboard:
        loggers.append(TensorBoardLogger(log_dir=log_dir, name=name))
    
    if use_wandb:
        loggers.append(WandBLogger(
            project=wandb_project or "vla-rl",
            name=name,
            config=wandb_config,
        ))
    
    if len(loggers) == 1:
        return loggers[0]
    
    return CompositeLogger(loggers)


__all__ = [
    "BaseLogger",
    "LogLevel",
    "ConsoleLogger",
    "FileLogger",
    "JSONLogger",
    "CSVLogger",
    "TensorBoardLogger",
    "WandBLogger",
    "CompositeLogger",
    "MetricsTracker",
    "EpisodeMetrics",
    "create_logger",
]
