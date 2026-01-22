"""图像变换"""
from typing import Dict, Any, Tuple
import numpy as np
from core.interfaces import TransformInterface
from core.orchestration import register_transform


@register_transform("resize")
class ResizeImage(TransformInterface):
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        for k, v in data.items():
            if k.startswith("images.") and isinstance(v, np.ndarray):
                if v.shape[-2:] != self.size:
                    data[k] = cv2.resize(v, self.size)
        return data


@register_transform("normalize")
class NormalizeImage(TransformInterface):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if k.startswith("images.") and isinstance(v, np.ndarray):
                v = v.astype(np.float32) / 255.0
                data[k] = (v - self.mean) / self.std
        return data


@register_transform("crop_roi")
class CropROI(TransformInterface):
    """ROI裁剪"""
    def __init__(self, crop_params: Dict[str, Tuple[int, int, int, int]]):
        self.crop_params = crop_params  # {key: (top, left, height, width)}
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, (t, l, h, w) in self.crop_params.items():
            if k in data and isinstance(data[k], np.ndarray):
                data[k] = data[k][..., t:t+h, l:l+w]
        return data
