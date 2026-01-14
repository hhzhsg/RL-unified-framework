"""
网络基类

纯神经网络模块的基类
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """
    网络基类
    
    与 BasePolicy 的区别:
    - BaseNetwork: 纯神经网络，只有 forward()
    - BasePolicy: 策略，有 act()，知道如何从观测到动作
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        pass
