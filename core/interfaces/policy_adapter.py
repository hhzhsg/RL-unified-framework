"""
策略适配器接口

解耦 HIL 控制逻辑和具体模型实现
支持接入任意策略模型（SAC、pi0、OpenVLA 等）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch


class PolicyAdapter(ABC):
    """
    策略适配器抽象基类
    
    职责：
    - 统一不同模型的推理接口
    - 统一权重同步接口（支持全量/部分同步）
    - 处理模型特定的输入输出格式转换
    
    使用示例：
        # SAC 模型
        adapter = SACPolicyAdapter(sac_policy)
        
        # pi0.5 模型（只同步 LoRA）
        adapter = Pi0PolicyAdapter(pi0_model, sync_mode="lora")
        
        # Actor 侧使用
        action = adapter.act(obs)
        adapter.load_weights(weights_from_learner)
        
        # Learner 侧使用
        weights = adapter.get_weights()
    """
    
    @abstractmethod
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        """
        推理动作
        
        Args:
            obs: 观测字典，格式由具体实现定义
            deterministic: 是否确定性推理
            
        Returns:
            动作（numpy array 或其他格式）
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取需要同步的权重
        
        对于小模型：返回全部参数
        对于大模型：可能只返回 adapter/LoRA 参数
        
        Returns:
            权重字典
        """
        pass
    
    @abstractmethod
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        加载权重
        
        Args:
            weights: 权重字典（与 get_weights 返回格式对应）
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """模型所在设备"""
        pass
    
    def reset(self) -> None:
        """重置策略状态（可选，用于有状态的策略）"""
        pass
    
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理观测（可选覆盖）
        
        用于将环境输出转换为模型输入格式
        """
        return obs
    
    def postprocess_action(self, action: Any) -> Any:
        """
        后处理动作（可选覆盖）
        
        用于将模型输出转换为环境输入格式
        """
        return action


class TrainablePolicyAdapter(PolicyAdapter):
    """
    可训练的策略适配器
    
    扩展 PolicyAdapter，添加训练相关接口
    用于 Learner 端
    """
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播（用于训练）"""
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            batch: 训练批次
            
        Returns:
            loss: 总损失
            metrics: 指标字典
        """
        pass
    
    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """获取优化器"""
        pass
    
    def get_trainable_parameters(self):
        """
        获取可训练参数
        
        对于大模型：可能只返回 adapter/LoRA 参数
        """
        raise NotImplementedError("Subclass should implement this if needed")


# ============ 权重同步模式 ============

class WeightSyncMode:
    """权重同步模式"""
    FULL = "full"           # 全量同步
    LORA = "lora"           # 只同步 LoRA 参数
    ADAPTER = "adapter"     # 只同步 Adapter 参数
    HEAD = "head"           # 只同步输出头
    CUSTOM = "custom"       # 自定义


# ============ 辅助函数 ============

def filter_weights_by_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefixes: list,
    include: bool = True
) -> Dict[str, torch.Tensor]:
    """
    按前缀过滤权重
    
    Args:
        state_dict: 完整权重
        prefixes: 前缀列表
        include: True=只包含匹配项，False=排除匹配项
    """
    result = {}
    for key, value in state_dict.items():
        matches = any(key.startswith(p) for p in prefixes)
        if (include and matches) or (not include and not matches):
            result[key] = value
    return result


def filter_weights_by_keyword(
    state_dict: Dict[str, torch.Tensor],
    keywords: list,
    include: bool = True
) -> Dict[str, torch.Tensor]:
    """
    按关键词过滤权重
    
    Args:
        state_dict: 完整权重
        keywords: 关键词列表（如 ["lora", "adapter"]）
        include: True=只包含匹配项，False=排除匹配项
    """
    result = {}
    for key, value in state_dict.items():
        matches = any(kw in key.lower() for kw in keywords)
        if (include and matches) or (not include and not matches):
            result[key] = value
    return result
