"""
VLA-RL DataHub 增强版

新增功能:
1. 数据集级别的权重配置
2. 动态调整采样比例
3. 支持多数据集 co-training
"""
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import random
import numpy as np
import torch

from buffer.base_buffer import BaseBuffer
from buffer.rollout_buffer import RolloutBuffer
from data import Transition, Batch, RobotState, Action


@dataclass
class DatasetConfig:
    """
    单个数据集的配置
    
    Attributes:
        name: 数据集名称
        buffer: 对应的 buffer 实例
        weight: 采样权重 (会归一化)
        source_tag: 数据来源标签
    """
    name: str
    buffer: BaseBuffer
    weight: float = 1.0
    source_tag: str = "default"


@dataclass 
class SamplingConfig:
    """
    采样配置
    
    Attributes:
        strategy: 采样策略名称
        weights: 各数据集权重 {dataset_name: weight}
        dynamic_reweight: 是否动态调整权重
        reweight_fn: 动态权重计算函数
    """
    strategy: str = "weighted"
    weights: Dict[str, float] = field(default_factory=dict)
    dynamic_reweight: bool = False
    reweight_fn: Optional[Callable] = None


class DataHub:
    """
    数据中心  - 支持多数据集加权采样
    
    Features:
    - 数据集级别的权重配置
    - 动态调整采样比例
    - 支持 co-training 多个异构数据集
    
    Example:
        # 创建 DataHub
        data_hub = DataHub()
        
        # 注册多个数据集
        data_hub.register_dataset("demo_expert", demo_buffer, weight=0.3, source_tag="demo")
        data_hub.register_dataset("demo_novice", novice_buffer, weight=0.1, source_tag="demo")
        data_hub.register_dataset("rollout", rollout_buffer, weight=0.5, source_tag="rollout")
        data_hub.register_dataset("intervention", intervention_buffer, weight=0.1, source_tag="intervention")
        
        # 采样
        batch = data_hub.sample(batch_size=64)
        
        # 动态调整权重
        data_hub.set_weight("rollout", 0.7)
        data_hub.set_weight("demo_expert", 0.2)
    """
    
    def __init__(self):
        self._datasets: Dict[str, DatasetConfig] = {}
        self._sampling_config = SamplingConfig()
        
        # 为了向后兼容，保留直接访问
        self.demo_buffer: Optional[BaseBuffer] = None
        self.rollout_buffer: Optional[BaseBuffer] = None
        self.intervention_buffer: Optional[BaseBuffer] = None
    
    # ==================== 数据集管理 ====================
    
    def register_dataset(self, name: str, buffer: BaseBuffer, 
                         weight: float = 1.0, source_tag: str = "default"):
        """
        注册数据集
        
        Args:
            name: 数据集唯一名称
            buffer: Buffer 实例
            weight: 采样权重 (未归一化)
            source_tag: 数据来源标签 ("demo" | "rollout" | "intervention")
        """
        self._datasets[name] = DatasetConfig(
            name=name,
            buffer=buffer,
            weight=weight,
            source_tag=source_tag,
        )
        
        if source_tag == "demo" and self.demo_buffer is None:
            self.demo_buffer = buffer
        elif source_tag == "rollout" and self.rollout_buffer is None:
            self.rollout_buffer = buffer
        elif source_tag == "intervention" and self.intervention_buffer is None:
            self.intervention_buffer = buffer
    
    def unregister_dataset(self, name: str):
        """移除数据集"""
        if name in self._datasets:
            del self._datasets[name]
    
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        """获取数据集配置"""
        return self._datasets.get(name)
    
    @property
    def dataset_names(self) -> List[str]:
        """所有数据集名称"""
        return list(self._datasets.keys())
    
    # ==================== 权重管理 ====================
    
    def set_weight(self, name: str, weight: float):
        """
        设置单个数据集的权重
        
        Args:
            name: 数据集名称
            weight: 新权重值
        """
        if name not in self._datasets:
            raise KeyError(f"数据集 '{name}' 不存在")
        self._datasets[name].weight = weight
    
    def set_weights(self, weights: Dict[str, float]):
        """
        批量设置权重
        
        Args:
            weights: {dataset_name: weight}
        """
        for name, weight in weights.items():
            self.set_weight(name, weight)
    
    def get_normalized_weights(self) -> Dict[str, float]:
        """
        获取归一化后的权重
        
        Returns:
            {dataset_name: normalized_weight}
        """
        # 过滤空 buffer
        active = {
            name: cfg.weight 
            for name, cfg in self._datasets.items() 
            if len(cfg.buffer) > 0
        }
        
        if not active:
            return {}
        
        total = sum(active.values())
        return {name: w / total for name, w in active.items()}
    
    def enable_dynamic_reweight(self, reweight_fn: Callable[[Dict[str, int], int], Dict[str, float]]):
        """
        启用动态权重调整
        
        Args:
            reweight_fn: 权重计算函数
                输入: (buffer_sizes: Dict[str, int], train_step: int)
                输出: 新权重 Dict[str, float]
        
        Example:
            def my_reweight(sizes, step):
                # 随着训练进行，逐渐增加 rollout 的权重
                rollout_ratio = min(0.8, 0.3 + step * 0.0001)
                return {
                    "demo": 1 - rollout_ratio,
                    "rollout": rollout_ratio,
                }
            
            data_hub.enable_dynamic_reweight(my_reweight)
        """
        self._sampling_config.dynamic_reweight = True
        self._sampling_config.reweight_fn = reweight_fn
    
    def disable_dynamic_reweight(self):
        """禁用动态权重调整"""
        self._sampling_config.dynamic_reweight = False
        self._sampling_config.reweight_fn = None
    
    def _maybe_reweight(self, train_step: int = 0):
        """内部方法: 动态调整权重"""
        if not self._sampling_config.dynamic_reweight:
            return
        
        if self._sampling_config.reweight_fn is None:
            return
        
        sizes = {name: len(cfg.buffer) for name, cfg in self._datasets.items()}
        new_weights = self._sampling_config.reweight_fn(sizes, train_step)
        
        for name, weight in new_weights.items():
            if name in self._datasets:
                self._datasets[name].weight = weight
    
    # ==================== 采样 ====================
    
    def sample(self, batch_size: int, train_step: int = 0) -> Batch:
        """
        加权采样
        
        Args:
            batch_size: 采样数量
            train_step: 当前训练步数 (用于动态权重)
            
        Returns:
            Batch 对象
        """
        # 动态调整权重
        self._maybe_reweight(train_step)
        
        # 获取归一化权重
        weights = self.get_normalized_weights()
        
        if not weights:
            raise ValueError("没有可用的数据集")
        
        # 按权重计算各数据集采样数量
        dataset_names = list(weights.keys())
        dataset_weights = [weights[name] for name in dataset_names]
        
        # 分配采样数量
        counts = self._allocate_samples(batch_size, dataset_weights)
        
        # 从各数据集采样
        all_transitions: List[Transition] = []
        
        for name, count in zip(dataset_names, counts):
            if count > 0:
                cfg = self._datasets[name]
                transitions = cfg.buffer.sample_transitions(count)
                all_transitions.extend(transitions)
        
        # 打乱顺序
        random.shuffle(all_transitions)
        
        return self._transitions_to_batch(all_transitions)
    
    def _allocate_samples(self, total: int, weights: List[float]) -> List[int]:
        """
        按权重分配采样数量
        
        使用概率舍入确保总数正确
        """
        # 计算期望数量
        expected = [total * w for w in weights]
        
        # 向下取整
        counts = [int(e) for e in expected]
        
        # 剩余数量
        remainder = total - sum(counts)
        
        # 按小数部分概率分配剩余
        if remainder > 0:
            fractions = [e - int(e) for e in expected]
            # 按小数部分排序，分配给最大的几个
            indices = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
            for i in range(remainder):
                counts[indices[i]] += 1
        
        return counts
    
    def sample_by_source(self, batch_size: int, source_weights: Dict[str, float]) -> Batch:
        """
        按 source_tag 采样
        
        Args:
            batch_size: 采样数量
            source_weights: {source_tag: weight}，如 {"demo": 0.3, "rollout": 0.7}
            
        Returns:
            Batch 对象
        """
        # 聚合同一 source 的数据集
        source_to_datasets: Dict[str, List[str]] = {}
        for name, cfg in self._datasets.items():
            if len(cfg.buffer) > 0:
                source_to_datasets.setdefault(cfg.source_tag, []).append(name)
        
        # 过滤有数据的 source
        active_sources = {s: w for s, w in source_weights.items() if s in source_to_datasets}
        
        if not active_sources:
            raise ValueError("没有匹配的数据源")
        
        # 归一化
        total = sum(active_sources.values())
        normalized = {s: w / total for s, w in active_sources.items()}
        
        # 分配数量到 source
        source_names = list(normalized.keys())
        source_weights_list = [normalized[s] for s in source_names]
        source_counts = self._allocate_samples(batch_size, source_weights_list)
        
        # 从各 source 采样
        all_transitions: List[Transition] = []
        
        for source, count in zip(source_names, source_counts):
            if count > 0:
                dataset_names = source_to_datasets[source]
                # 在同一 source 内按数据集权重采样
                ds_weights = [self._datasets[n].weight for n in dataset_names]
                ds_total = sum(ds_weights)
                ds_normalized = [w / ds_total for w in ds_weights]
                ds_counts = self._allocate_samples(count, ds_normalized)
                
                for ds_name, ds_count in zip(dataset_names, ds_counts):
                    if ds_count > 0:
                        transitions = self._datasets[ds_name].buffer.sample_transitions(ds_count)
                        all_transitions.extend(transitions)
        
        random.shuffle(all_transitions)
        return self._transitions_to_batch(all_transitions)
    
    def _transitions_to_batch(self, transitions: List[Transition]) -> Batch:
        """将 Transition 列表转换为 Batch"""
        robot_states = []
        actions = []
        rewards = []
        next_robot_states = []
        dones = []
        sources = []
        
        for t in transitions:
            if t.robot_state.raw_state is not None:
                robot_states.append(t.robot_state.raw_state)
            else:
                robot_states.append(t.robot_state.to_array())
            
            if t.next_robot_state.raw_state is not None:
                next_robot_states.append(t.next_robot_state.raw_state)
            else:
                next_robot_states.append(t.next_robot_state.to_array())
            
            actions.append(t.action.data)
            rewards.append(t.reward)
            dones.append(float(t.done))
            sources.append(t.source)
        
        return Batch(
            obs={},
            robot_state=torch.FloatTensor(np.array(robot_states)),
            action=torch.FloatTensor(np.array(actions)),
            reward=torch.FloatTensor(np.array(rewards)),
            next_obs={},
            next_robot_state=torch.FloatTensor(np.array(next_robot_states)),
            done=torch.FloatTensor(np.array(dones)),
            source=sources,
        )
    
    # ==================== 统计信息 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "num_datasets": len(self._datasets),
            "datasets": {},
            "total_transitions": 0,
            "weights": self.get_normalized_weights(),
        }
        
        for name, cfg in self._datasets.items():
            ds_stats = {
                "size": len(cfg.buffer),
                "weight": cfg.weight,
                "source_tag": cfg.source_tag,
            }
            stats["datasets"][name] = ds_stats
            stats["total_transitions"] += len(cfg.buffer)
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        weights = stats["weights"]
        
        print("=" * 50)
        print("DataHub Statistics")
        print("=" * 50)
        print(f"Total datasets: {stats['num_datasets']}")
        print(f"Total transitions: {stats['total_transitions']}")
        print("-" * 50)
        print(f"{'Dataset':<20} {'Size':<10} {'Weight':<10} {'Source':<10}")
        print("-" * 50)
        
        for name, ds_stats in stats["datasets"].items():
            norm_weight = weights.get(name, 0)
            print(f"{name:<20} {ds_stats['size']:<10} {norm_weight:.2%} ({ds_stats['weight']:.2f})  {ds_stats['source_tag']:<10}")
        
        print("=" * 50)


# ==================== 便捷函数 ====================

def create_data_hub(
    demo_paths: Optional[List[str]] = None,
    rollout_capacity: int = 100000,
    intervention_capacity: int = 10000,
    demo_weight: float = 0.3,
    rollout_weight: float = 0.6,
    intervention_weight: float = 0.1,
) -> DataHub:
    """
    创建标准配置的 DataHub
    
    Args:
        demo_paths: Demo 数据路径
        rollout_capacity: Rollout 缓冲区容量
        intervention_capacity: Intervention 缓冲区容量
        demo_weight: Demo 数据权重
        rollout_weight: Rollout 数据权重
        intervention_weight: Intervention 数据权重
        
    Returns:
        配置好的 DataHub 实例
    """
    hub = DataHub()
    
    # 注册 Demo Buffer
    demo_buffer = RolloutBuffer(max_size=rollout_capacity)
    hub.register_dataset("demo", demo_buffer, weight=demo_weight, source_tag="demo")
    
    # 注册 Rollout Buffer
    rollout_buffer = RolloutBuffer(max_size=rollout_capacity)
    hub.register_dataset("rollout", rollout_buffer, weight=rollout_weight, source_tag="rollout")
    
    # 注册 Intervention Buffer
    intervention_buffer = RolloutBuffer(max_size=intervention_capacity)
    hub.register_dataset("intervention", intervention_buffer, weight=intervention_weight, source_tag="intervention")
    
    return hub


# ==================== 动态权重函数示例 ====================

def linear_rollout_increase(sizes: Dict[str, int], step: int, 
                            initial_rollout: float = 0.3,
                            final_rollout: float = 0.8,
                            warmup_steps: int = 10000) -> Dict[str, float]:
    """
    线性增加 rollout 权重
    
    随着训练进行，逐渐从 demo 过渡到 rollout
    """
    progress = min(1.0, step / warmup_steps)
    rollout_weight = initial_rollout + (final_rollout - initial_rollout) * progress
    demo_weight = 1.0 - rollout_weight
    
    return {
        "demo": demo_weight,
        "rollout": rollout_weight,
    }


def buffer_size_proportional(sizes: Dict[str, int], step: int,
                             min_weight: float = 0.1) -> Dict[str, float]:
    """
    按 buffer 大小比例调整权重
    
    数据越多的 buffer 权重越大，但设置最小权重
    """
    total = sum(sizes.values())
    if total == 0:
        return {name: 1.0 / len(sizes) for name in sizes}
    
    weights = {}
    for name, size in sizes.items():
        raw_weight = size / total
        weights[name] = max(min_weight, raw_weight)
    
    # 归一化
    total_weight = sum(weights.values())
    return {name: w / total_weight for name, w in weights.items()}
