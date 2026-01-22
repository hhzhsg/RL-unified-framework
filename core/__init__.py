"""
Core模块

系统主权层 - 只包含接口定义、运行时循环、系统组装和同步机制
禁止在此编写具体算法实现
"""
from . import interfaces
from . import runtime
from . import orchestration
from . import synchronization

__all__ = [
    "interfaces",
    "runtime",
    "orchestration",
    "synchronization",
]
