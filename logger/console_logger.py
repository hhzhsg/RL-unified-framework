"""
Console Logger - 控制台输出（带颜色）
"""
from typing import Dict, Any, Optional
from datetime import datetime
import sys

from .base_logger import BaseLogger, LogLevel


# ANSI 颜色码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def colorize(text: str, color: str) -> str:
    """添加颜色"""
    return f"{color}{text}{Colors.RESET}"


class ConsoleLogger(BaseLogger):
    """
    控制台日志输出
    
    特点：
    - 带颜色的格式化输出
    - 可配置详细程度
    - 支持进度显示
    """
    
    def __init__(self, 
                 name: str = "train",
                 log_level: LogLevel = LogLevel.INFO,
                 use_color: bool = True,
                 show_timestamp: bool = True,
                 float_precision: int = 4):
        """
        Args:
            name: Logger 名称
            log_level: 最低日志级别
            use_color: 是否使用颜色
            show_timestamp: 是否显示时间戳
            float_precision: 浮点数精度
        """
        super().__init__(name)
        self.log_level = log_level
        self.use_color = use_color
        self.show_timestamp = show_timestamp
        self.float_precision = float_precision
        
        # 每个标签的最后一次记录步数（避免重复输出）
        self._last_logged: Dict[str, int] = {}
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _format_value(self, value: float) -> str:
        """格式化数值"""
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.{self.float_precision}e}"
        return f"{value:.{self.float_precision}f}"
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量（通常不单独输出，用于 log_scalars）"""
        pass  # 由 log_scalars 统一处理
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """批量记录并输出"""
        if step is None:
            step = self._step
        
        # 构建输出字符串
        parts = []
        
        # 时间戳
        if self.show_timestamp:
            timestamp = self._get_timestamp()
            if self.use_color:
                timestamp = colorize(timestamp, Colors.CYAN)
            parts.append(f"[{timestamp}]")
        
        # 步数
        step_str = f"Step {step}"
        if self.use_color:
            step_str = colorize(step_str, Colors.GREEN)
        parts.append(step_str)
        
        # 前缀
        if prefix:
            if self.use_color:
                prefix = colorize(f"[{prefix}]", Colors.MAGENTA)
            else:
                prefix = f"[{prefix}]"
            parts.append(prefix)
        
        # 指标
        metric_parts = []
        for tag, value in metrics.items():
            formatted = self._format_value(value)
            if self.use_color:
                tag_colored = colorize(tag, Colors.YELLOW)
                metric_parts.append(f"{tag_colored}={formatted}")
            else:
                metric_parts.append(f"{tag}={formatted}")
        
        parts.append(" | ".join(metric_parts))
        
        print(" ".join(parts))
    
    def _log_message(self, message: str, level: LogLevel):
        """输出日志消息"""
        if level.value < self.log_level.value:
            return
        
        parts = []
        
        # 时间戳
        if self.show_timestamp:
            timestamp = self._get_timestamp()
            if self.use_color:
                timestamp = colorize(timestamp, Colors.CYAN)
            parts.append(f"[{timestamp}]")
        
        # 级别
        level_str = level.name
        if self.use_color:
            color_map = {
                LogLevel.DEBUG: Colors.WHITE,
                LogLevel.INFO: Colors.GREEN,
                LogLevel.WARNING: Colors.YELLOW,
                LogLevel.ERROR: Colors.RED,
            }
            level_str = colorize(level_str, color_map.get(level, Colors.WHITE))
        parts.append(f"[{level_str}]")
        
        # 消息
        parts.append(message)
        
        output = " ".join(parts)
        
        if level == LogLevel.ERROR:
            print(output, file=sys.stderr)
        else:
            print(output)
    
    def log_progress(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        """输出进度条"""
        bar_length = 40
        progress = current / total
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if self.use_color:
            bar = colorize(bar, Colors.GREEN)
        
        line = f"\r{prefix} |{bar}| {current}/{total} {progress*100:.1f}% {suffix}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if current >= total:
            print()  # 换行
    
    def log_separator(self, char: str = "=", length: int = 60):
        """输出分隔线"""
        line = char * length
        if self.use_color:
            line = colorize(line, Colors.CYAN)
        print(line)
    
    def log_header(self, title: str, char: str = "=", length: int = 60):
        """输出标题"""
        self.log_separator(char, length)
        
        # 居中标题
        padding = (length - len(title) - 2) // 2
        header = f"{char * padding} {title} {char * padding}"
        
        if self.use_color:
            header = colorize(header, Colors.BOLD + Colors.CYAN)
        print(header)
        
        self.log_separator(char, length)
    
    def close(self):
        """关闭（控制台无需特殊处理）"""
        pass
