"""
数据中心

统一管理三种数据源:
- demo: 预训练的专家数据 (离线数据，如HDF5)
- rollout: policy实时推理的轨迹数据 (在线RL收集)
- intervention: rollout过程中人类专家介入的数据 (在线纠正)
"""
from typing import Dict, List, Optional, Union, Any

from data.samplers.base_sampler import BaseSampler
from data.samplers.uniform_sampler import UniformSampler
from data.samplers.hilserl_sampler import HILSERLSampler


def create_sampler(strategy: str, **kwargs) -> BaseSampler:
    """Simple factory for samplers used by DataHub when needed."""
    if strategy == "uniform":
        return UniformSampler(**kwargs)
    if strategy == "hilserl":
        return HILSERLSampler(**kwargs)
    # default fallbacks
    return UniformSampler(**kwargs)


class DataHub:
    """
    数据中心
    
    职责:
    1. 管理三种 Buffer:
       - demo: 预训练专家数据 (HDF5Buffer)
       - rollout: 在线交互轨迹 (ReplayBuffer)
       - intervention: 人工介入纠正 (InterventionBuffer, 持久化)
    2. 提供统一的写入/采样接口
    3. 支持不同采样策略 (demo_only, rollout_only, mixed)
    
    Example:
        hub = DataHub()
        
        # 注册 demo buffer (预训练数据)
        hub.register_buffer("demo", HDF5DemoBuffer("data.hdf5"))
        
        # rollout 和 intervention 会自动创建
        
        # 在线交互: 写入 rollout
        hub.write(transition, source="rollout")
        
        # 人工介入: 写入 intervention
        hub.write(intervention_transition, source="intervention")
        
        # 采样: 混合三种数据源
        batch = hub.sample(batch_size=64, strategy="mixed")
        batch = hub.sample(batch_size=64, strategy="mixed")
    """
    
    def __init__(self, 
                 rollout_capacity: int = 100000,
                 intervention_capacity: int = 10000,
                 intervention_save_path: Optional[str] = None,
                 intervention_save_interval: int = 100,
                 load_intervention: bool = False):
        """
        Args:
            rollout_capacity: rollout buffer 容量
            intervention_capacity: intervention buffer 容量
            intervention_save_path: intervention 持久化路径
            intervention_save_interval: intervention 异步保存间隔
            load_intervention: 是否加载历史 intervention 数据
        """
        self._buffers: Dict[str, "BaseBuffer"] = {}
        self._samplers: Dict[str, BaseSampler] = {}
        self._default_sampler = "demo_only"
        
        # 自动创建 rollout/intervention buffer
        from data.buffers.replay_buffer import ReplayBuffer
        from data.buffers.intervention_buffer import InterventionBuffer
        self._buffers["rollout"] = ReplayBuffer(capacity=rollout_capacity)
        self._buffers["intervention"] = InterventionBuffer(
            capacity=intervention_capacity,
            save_path=intervention_save_path,
            save_interval=intervention_save_interval,
        )
        
        # 加载历史 intervention 数据
        if load_intervention and intervention_save_path:
            loaded = self._buffers["intervention"].load()
            if loaded > 0:
                print(f"[DataHub] Loaded {loaded} historical intervention samples")

    @property
    def buffers(self) -> Dict[str, "BaseBuffer"]:
        """公开内部 buffers 字典（兼容 TrainingLoop 调用）。"""
        return self._buffers

    def add(self, data: Dict[str, Any], source: str = "rollout"):
        """向指定 buffer 写入数据（兼容旧 API 名称 add）。"""
        return self.write(data, source=source)
    
    def register_buffer(self, name: str, buffer: "BaseBuffer"):
        """
        注册 buffer
        
        Args:
            name: buffer 名称 ("demo" | "rollout" | "intervention")
            buffer: buffer 实例
        """
        self._buffers[name] = buffer
    
    def get_buffer(self, name: str) -> Optional["BaseBuffer"]:
        """获取指定 buffer"""
        return self._buffers.get(name)
    
    @property
    def demo_buffer(self) -> Optional["BaseBuffer"]:
        return self._buffers.get("demo")
    
    @property
    def rollout_buffer(self) -> "BaseBuffer":
        return self._buffers["rollout"]
    
    @property
    def intervention_buffer(self) -> Optional["BaseBuffer"]:
        return self._buffers.get("intervention")
    
    def write(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], source: str = "rollout"):
        """
        写入数据
        
        Args:
            data: transition dict 或 transitions 列表
            source: 目标 buffer 名称
        """
        buffer = self._buffers.get(source)
        if buffer is None:
            raise ValueError(f"Buffer '{source}' not registered")
        
        # 支持单条或批量写入
        if isinstance(data, list):
            for t in data:
                buffer.add(t)
        else:
            buffer.add(data)
    
    def sample(self, batch_size: int, strategy: str = "demo_only", **kwargs) -> Dict[str, Any]:
        """
        采样数据
        
        Args:
            batch_size: 采样数量
            strategy: 采样策略 ("demo_only" | "rollout_only" | "mixed")
            **kwargs: 传递给采样器的参数
            
        Returns:
            训练 batch (dict)
        """
        # 获取或创建采样器
        if strategy not in self._samplers:
            self._samplers[strategy] = create_sampler(strategy, **kwargs)
        
        sampler = self._samplers[strategy]
        return sampler.sample(self._buffers, batch_size)
    
    def statistics(self) -> Dict[str, int]:
        """获取各 buffer 统计信息"""
        stats = {}
        for name, buffer in self._buffers.items():
            stats[f"{name}_size"] = len(buffer)
            if hasattr(buffer, "num_episodes"):
                stats[f"{name}_episodes"] = buffer.num_episodes
            if hasattr(buffer, "total_saved"):
                stats[f"{name}_saved"] = buffer.total_saved
        return stats
    
    def close(self):
        """关闭 DataHub，保存所有未保存的 intervention 数据"""
        intervention_buf = self._buffers.get("intervention")
        if intervention_buf and hasattr(intervention_buf, "close"):
            intervention_buf.close()
    
    def __repr__(self) -> str:
        stats = self.statistics()
        parts = [f"{k}={v}" for k, v in stats.items()]
        return f"DataHub({', '.join(parts)})"
