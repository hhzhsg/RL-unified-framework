"""图像编码器"""
from typing import Tuple
import torch
import torch.nn as nn
from core.interfaces import ImageEncoderInterface
from core.orchestration import register_policy


@register_policy("image_encoder")
class ImageEncoder(ImageEncoderInterface):
    """CNN图像编码器"""
    
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int = 256):
        super().__init__()
        
        self._input_shape = input_shape
        self._output_dim = output_dim
        
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算卷积输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.conv(dummy).shape[1]
        
        self.fc = nn.Linear(conv_out_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return self.fc(features)
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape
    
    @property
    def input_dim(self) -> int:
        c, h, w = self._input_shape
        return c * h * w
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def latent_dim(self) -> int:
        return self._output_dim
