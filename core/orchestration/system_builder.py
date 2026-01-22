"""
系统构建器

负责从配置构建完整系统
"""
from typing import Dict, Any, Type, Optional
from dataclasses import dataclass

from ..interfaces import (
    EnvInterface, 
    BufferInterface, 
    SamplerInterface,
    PolicyInterface,
    AlgorithmInterface,
    SyncInterface,
)
from .component_registry import ComponentRegistry


@dataclass
class SystemComponents:
    """系统组件容器"""
    env: Optional[EnvInterface] = None
    buffers: Dict[str, BufferInterface] = None
    sampler: Optional[SamplerInterface] = None
    policy: Optional[PolicyInterface] = None
    algorithm: Optional[AlgorithmInterface] = None
    weight_sync: Optional[SyncInterface] = None
    
    def __post_init__(self):
        if self.buffers is None:
            self.buffers = {}


class SystemBuilder:
    """
    系统构建器
    
    根据配置构建完整的训练/推理系统
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.components = SystemComponents()
    
    def build_from_config(self, config: Dict[str, Any]) -> SystemComponents:
        """
        从配置构建系统
        
        Args:
            config: 系统配置
            
        Returns:
            构建好的系统组件
        """
        # 1. 构建环境
        if "env" in config:
            self.components.env = self._build_env(config["env"])
        
        # 2. 构建Buffer
        if "buffers" in config:
            for name, buf_config in config["buffers"].items():
                self.components.buffers[name] = self._build_buffer(buf_config)
        
        # 3. 构建Sampler
        if "sampler" in config:
            self.components.sampler = self._build_sampler(config["sampler"])
        
        # 4. 构建Policy
        if "policy" in config:
            self.components.policy = self._build_policy(config["policy"])
        
        # 5. 构建Algorithm
        if "algorithm" in config:
            self.components.algorithm = self._build_algorithm(
                config["algorithm"],
                self.components.policy
            )
        
        # 6. 构建同步器
        if "sync" in config:
            self.components.weight_sync = self._build_sync(config["sync"])
        
        return self.components
    
    def _build_env(self, config: Dict[str, Any]) -> EnvInterface:
        """构建环境"""
        env_type = config.pop("type")
        env_cls = self.registry.get("env", env_type)
        return env_cls(**config)
    
    def _build_buffer(self, config: Dict[str, Any]) -> BufferInterface:
        """构建Buffer"""
        buf_type = config.pop("type")
        buf_cls = self.registry.get("buffer", buf_type)
        return buf_cls(**config)
    
    def _build_sampler(self, config: Dict[str, Any]) -> SamplerInterface:
        """构建Sampler"""
        sampler_type = config.pop("type")
        sampler_cls = self.registry.get("sampler", sampler_type)
        return sampler_cls(**config)
    
    def _build_policy(self, config: Dict[str, Any]) -> PolicyInterface:
        """构建Policy"""
        policy_type = config.pop("type")
        policy_cls = self.registry.get("policy", policy_type)
        return policy_cls(**config)
    
    def _build_algorithm(self, config: Dict[str, Any], policy: PolicyInterface) -> AlgorithmInterface:
        """构建Algorithm"""
        algo_type = config.pop("type")
        algo_cls = self.registry.get("algorithm", algo_type)
        return algo_cls(policy=policy, **config)
    
    def _build_sync(self, config: Dict[str, Any]) -> SyncInterface:
        """构建同步器"""
        sync_type = config.pop("type")
        sync_cls = self.registry.get("sync", sync_type)
        return sync_cls(**config)
