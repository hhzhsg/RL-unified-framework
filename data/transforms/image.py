"""
图像转换

图像预处理相关的转换:
- ResizeImage: 调整图像大小
- NormalizeImage: 图像归一化
- ImageToTensor: 图像转 Tensor
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .base import BaseTransform


class ResizeImage(BaseTransform):
    """
    调整图像大小
    
    Args:
        size: 目标大小 (H, W) 或 int (正方形)
        keys: 要处理的图像 key 列表，None 表示所有
    """
    
    def __init__(self, size: int | Tuple[int, int], keys: Optional[List[str]] = None):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        images = data.get("images", {})
        if not images:
            return data
        
        keys_to_process = self.keys or list(images.keys())
        
        for key in keys_to_process:
            if key in images:
                img = images[key]
                images[key] = self._resize(img)
        
        data["images"] = images
        return data
    
    def _resize(self, img: np.ndarray) -> np.ndarray:
        """调整图像大小 (使用简单的最近邻插值)"""
        try:
            import cv2
            return cv2.resize(img, (self.size[1], self.size[0]))
        except ImportError:
            # 简单的最近邻插值
            h, w = img.shape[:2]
            new_h, new_w = self.size
            row_ratio = h / new_h
            col_ratio = w / new_w
            
            row_idx = (np.arange(new_h) * row_ratio).astype(int)
            col_idx = (np.arange(new_w) * col_ratio).astype(int)
            
            return img[row_idx[:, None], col_idx]


class NormalizeImage(BaseTransform):
    """
    图像归一化
    
    将 [0, 255] 归一化到 [0, 1] 或 [-1, 1]
    """
    
    def __init__(self, 
                 mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
                 std: Tuple[float, ...] = (0.5, 0.5, 0.5),
                 keys: Optional[List[str]] = None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        images = data.get("images", {})
        if not images:
            return data
        
        keys_to_process = self.keys or list(images.keys())
        
        for key in keys_to_process:
            if key in images:
                img = images[key].astype(np.float32) / 255.0
                img = (img - self.mean) / self.std
                images[key] = img
        
        data["images"] = images
        return data


class ImageToTensor(BaseTransform):
    """
    图像转 PyTorch Tensor
    
    将 (H, W, C) 转为 (C, H, W)
    """
    
    def __init__(self, keys: Optional[List[str]] = None):
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        
        images = data.get("images", {})
        if not images:
            return data
        
        keys_to_process = self.keys or list(images.keys())
        
        for key in keys_to_process:
            if key in images:
                img = images[key]
                if isinstance(img, np.ndarray):
                    # (H, W, C) -> (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
                    img = torch.from_numpy(img.copy())
                images[key] = img
        
        data["images"] = images
        return data
