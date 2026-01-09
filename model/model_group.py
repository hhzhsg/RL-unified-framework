"""
VLA-RL ModelGroup: 多模型管理

统一管理训练中的多个模型:
- 注册/获取模型
- 冻结/解冻控制
- 批量保存/加载
"""
from typing import Dict, Iterator, Optional, List, Any
import torch
import torch.nn as nn


class ModelGroup:
    """
    模型组管理器
    
    管理多个相关模型 (如 policy + q1 + q2 + target_q1 + target_q2)
    支持冻结/解冻、统一保存加载
    
    命名规范:
    - policy: 策略网络
    - q1, q2: Q 网络
    - target_q1, target_q2: 目标 Q 网络
    - vf: 价值函数
    """
    
    def __init__(self):
        self._models: Dict[str, nn.Module] = {}
        self._frozen: Dict[str, bool] = {}
    
    def add(self, name: str, model: nn.Module, frozen: bool = False):
        """
        添加模型
        
        Args:
            name: 模型名称
            model: PyTorch 模型
            frozen: 是否冻结
        """
        self._models[name] = model
        self._frozen[name] = frozen
        
        if frozen:
            self.freeze(name)
    
    def get(self, name: str) -> nn.Module:
        """获取模型"""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def __getitem__(self, name: str) -> nn.Module:
        return self.get(name)
    
    def __contains__(self, name: str) -> bool:
        return name in self._models
    
    def freeze(self, name: str):
        """冻结模型参数"""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        model = self._models[name]
        for param in model.parameters():
            param.requires_grad = False
        self._frozen[name] = True
    
    def unfreeze(self, name: str):
        """解冻模型参数"""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        model = self._models[name]
        for param in model.parameters():
            param.requires_grad = True
        self._frozen[name] = False
    
    def is_frozen(self, name: str) -> bool:
        """检查模型是否冻结"""
        return self._frozen.get(name, False)
    
    def trainable_parameters(self, names: Optional[List[str]] = None) -> Iterator[nn.Parameter]:
        """
        获取可训练参数
        
        Args:
            names: 指定模型名称列表，None 表示所有未冻结模型
        """
        if names is None:
            names = [n for n in self._models.keys() if not self._frozen[n]]
        
        for name in names:
            if name in self._models and not self._frozen.get(name, False):
                for param in self._models[name].parameters():
                    if param.requires_grad:
                        yield param
    
    def state_dict(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取模型权重
        
        Args:
            names: 指定模型名称列表，None 表示所有模型
        """
        if names is None:
            names = list(self._models.keys())
        
        return {name: self._models[name].state_dict() for name in names if name in self._models}
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        加载模型权重
        
        Args:
            state_dict: {"model_name": state_dict, ...}
            strict: 是否严格匹配
        """
        for name, sd in state_dict.items():
            if name in self._models:
                self._models[name].load_state_dict(sd, strict=strict)
    
    def to(self, device: str) -> "ModelGroup":
        """移动到指定设备"""
        for model in self._models.values():
            model.to(device)
        return self
    
    def train(self, names: Optional[List[str]] = None):
        """设置为训练模式"""
        if names is None:
            names = list(self._models.keys())
        
        for name in names:
            if name in self._models:
                self._models[name].train()
    
    def eval(self, names: Optional[List[str]] = None):
        """设置为评估模式"""
        if names is None:
            names = list(self._models.keys())
        
        for name in names:
            if name in self._models:
                self._models[name].eval()
    
    @property
    def model_names(self) -> List[str]:
        """所有模型名称"""
        return list(self._models.keys())
    
    def summary(self) -> Dict[str, Dict]:
        """获取模型摘要"""
        summary = {}
        for name, model in self._models.items():
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            summary[name] = {
                "num_params": num_params,
                "num_trainable": num_trainable,
                "frozen": self._frozen[name],
            }
        return summary
