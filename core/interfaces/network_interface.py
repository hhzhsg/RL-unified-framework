"""
Network接口定义

底层网络模块接口
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class NetworkInterface(nn.Module, ABC):
    """网络模块接口"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """输出维度"""
        pass


class EncoderInterface(NetworkInterface):
    """编码器接口"""
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """输入维度"""
        pass
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """隐层维度"""
        pass


class MLPInterface(NetworkInterface):
    """MLP接口"""
    
    @property
    @abstractmethod
    def hidden_dims(self) -> List[int]:
        """隐藏层维度列表"""
        pass


class ImageEncoderInterface(EncoderInterface):
    """图像编码器接口"""
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int, int]:
        """输入形状 (C, H, W)"""
        pass
