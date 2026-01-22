"""
权重订阅器

推理端使用，订阅最新权重
"""
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from .base_sync import BaseSynchronizer


class WeightSubscriber:
    """
    权重订阅器
    
    封装同步器，提供权重订阅接口
    """
    
    def __init__(self, synchronizer: BaseSynchronizer, tag: str = "weights"):
        self.synchronizer = synchronizer
        self.tag = tag
        self._last_version = -1
    
    def try_update(self, model: nn.Module, device: torch.device = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        尝试更新模型权重
        
        Args:
            model: 要更新的模型
            device: 目标设备
            
        Returns:
            updated: 是否更新了
            metadata: 元数据
        """
        data = self.synchronizer.pull(tag=self.tag)
        
        if data is None:
            return False, None
        
        version = data.get("version", 0)
        if version <= self._last_version:
            return False, None
        
        state_dict = data["state_dict"]
        
        # 移动到目标设备
        if device is not None:
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        self._last_version = version
        
        return True, data.get("metadata")
    
    @property
    def last_version(self) -> int:
        return self._last_version
