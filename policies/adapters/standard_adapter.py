"""
标准策略适配器

将框架内的 PolicyInterface 和 AlgorithmInterface 适配为 PolicyAdapter
用于 HIL 训练中的 Actor/Learner 端
"""
from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np

from core.interfaces.policy_adapter import (
    PolicyAdapter,
    TrainablePolicyAdapter,
    WeightSyncMode,
    filter_weights_by_prefix,
)
from core.interfaces.policy_interface import PolicyInterface
from core.interfaces.algorithm_interface import AlgorithmInterface


class StandardPolicyAdapter(PolicyAdapter):
    """
    标准策略适配器
    
    封装 PolicyInterface，用于 Actor 端
    
    使用示例:
        policy = SACPolicy(config)
        adapter = StandardPolicyAdapter(policy)
        
        action = adapter.act(obs)
        weights = adapter.get_weights()
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        sync_mode: str = WeightSyncMode.FULL,
        sync_prefixes: list = None,
    ):
        """
        Args:
            policy: PolicyInterface 实例
            sync_mode: 权重同步模式
            sync_prefixes: 自定义同步的参数前缀（sync_mode=CUSTOM 时使用）
        """
        self.policy = policy
        self.sync_mode = sync_mode
        self.sync_prefixes = sync_prefixes or []
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        return self.policy.act(obs, deterministic=deterministic)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        state_dict = self.policy.state_dict()
        
        if self.sync_mode == WeightSyncMode.FULL:
            return state_dict
        elif self.sync_mode == WeightSyncMode.CUSTOM:
            return filter_weights_by_prefix(state_dict, self.sync_prefixes, include=True)
        else:
            # 其他模式暂时返回全量
            return state_dict
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.sync_mode == WeightSyncMode.FULL:
            self.policy.load_state_dict(weights)
        else:
            # 部分加载
            current_state = self.policy.state_dict()
            current_state.update(weights)
            self.policy.load_state_dict(current_state)
    
    @property
    def device(self) -> torch.device:
        return self.policy.device
    
    def reset(self) -> None:
        if hasattr(self.policy, 'reset'):
            self.policy.reset()


class AlgorithmAdapter(TrainablePolicyAdapter):
    """
    算法适配器
    
    封装 AlgorithmInterface，用于 Learner 端
    提供训练和权重同步功能
    
    使用示例:
        algorithm = SACAlgorithm(policy, config)
        adapter = AlgorithmAdapter(algorithm)
        
        metrics = adapter.update(batch)
        weights = adapter.get_weights()
    """
    
    def __init__(
        self,
        algorithm: AlgorithmInterface,
        sync_mode: str = WeightSyncMode.FULL,
        sync_prefixes: list = None,
    ):
        """
        Args:
            algorithm: AlgorithmInterface 实例
            sync_mode: 权重同步模式
            sync_prefixes: 自定义同步的参数前缀
        """
        self.algorithm = algorithm
        self.sync_mode = sync_mode
        self.sync_prefixes = sync_prefixes or []
        self._policy = algorithm.get_policy()
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        return self._policy.act(obs, deterministic=deterministic)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        state_dict = self._policy.state_dict()
        
        if self.sync_mode == WeightSyncMode.FULL:
            return state_dict
        elif self.sync_mode == WeightSyncMode.CUSTOM:
            return filter_weights_by_prefix(state_dict, self.sync_prefixes, include=True)
        else:
            return state_dict
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.sync_mode == WeightSyncMode.FULL:
            self._policy.load_state_dict(weights)
        else:
            current_state = self._policy.state_dict()
            current_state.update(weights)
            self._policy.load_state_dict(current_state)
    
    @property
    def device(self) -> torch.device:
        return self._policy.device
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._policy.forward(obs)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """通过 algorithm.update() 计算损失并更新"""
        metrics = self.algorithm.update(batch)
        # 返回虚拟 loss（实际更新已在 update 中完成）
        return torch.tensor(0.0), metrics
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError("Use algorithm.update() instead")
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """直接调用算法更新"""
        return self.algorithm.update(batch)
    
    def save(self, path: str) -> None:
        """保存算法状态"""
        self.algorithm.save(path)
    
    def load(self, path: str) -> None:
        """加载算法状态"""
        self.algorithm.load(path)
