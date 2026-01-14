"""
模型组管理器

统一管理算法所需的多个模型
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn


class ModelGroup:
    """
    模型组管理器
    
    管理算法所需的多个模型，支持:
    - 统一的设备管理
    - 模型冻结/解冻
    - 统一保存/加载
    
    Example:
        group = ModelGroup()
        group.add("policy", policy_net)
        group.add("q1", q_net)
        group.add("target_q1", target_q, frozen=True)
        
        group.to("cuda")
        group.save("checkpoint.pt")
    """
    
    def __init__(self):
        self._models: Dict[str, nn.Module] = {}
        self._frozen: Dict[str, bool] = {}
    
    def add(self, name: str, model: nn.Module, frozen: bool = False):
        """
        添加模型
        
        Args:
            name: 模型名称
            model: 模型实例
            frozen: 是否冻结参数
        """
        self._models[name] = model
        self._frozen[name] = frozen
        
        if frozen:
            self.freeze(name)
    
    def get(self, name: str) -> Optional[nn.Module]:
        """获取模型"""
        return self._models.get(name)
    
    def __getitem__(self, name: str) -> nn.Module:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {self.model_names}")
        return self._models[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self._models
    
    @property
    def model_names(self) -> List[str]:
        """所有模型名称"""
        return list(self._models.keys())
    
    def freeze(self, name: str):
        """冻结模型参数"""
        if name in self._models:
            for param in self._models[name].parameters():
                param.requires_grad = False
            self._frozen[name] = True
    
    def unfreeze(self, name: str):
        """解冻模型参数"""
        if name in self._models:
            for param in self._models[name].parameters():
                param.requires_grad = True
            self._frozen[name] = False
    
    def is_frozen(self, name: str) -> bool:
        """检查模型是否冻结"""
        return self._frozen.get(name, False)
    
    def to(self, device: str):
        """移动所有模型到指定设备"""
        for model in self._models.values():
            model.to(device)
        return self
    
    def train(self):
        """设置所有未冻结模型为训练模式"""
        for name, model in self._models.items():
            if not self._frozen[name]:
                model.train()
    
    def eval(self):
        """设置所有模型为评估模式"""
        for model in self._models.values():
            model.eval()
    
    def state_dict(self) -> Dict[str, Any]:
        """获取所有模型状态"""
        return {name: model.state_dict() for name, model in self._models.items()}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载所有模型状态"""
        for name, model_state in state_dict.items():
            if name in self._models:
                self._models[name].load_state_dict(model_state)
    
    def save(self, path: str):
        """保存到文件"""
        torch.save({
            "models": self.state_dict(),
            "frozen": self._frozen,
        }, path)
    
    def load(self, path: str):
        """从文件加载"""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["models"])
        self._frozen = checkpoint.get("frozen", {})
    
    def parameters(self, trainable_only: bool = True):
        """
        获取参数迭代器
        
        Args:
            trainable_only: 是否只返回可训练参数
        """
        for name, model in self._models.items():
            if trainable_only and self._frozen[name]:
                continue
            yield from model.parameters()
    
    def __repr__(self) -> str:
        parts = []
        for name in self._models:
            status = "frozen" if self._frozen[name] else "trainable"
            parts.append(f"{name}({status})")
        return f"ModelGroup({', '.join(parts)})"
