"""
训练循环

管理训练流程，支持多阶段训练
"""
from typing import Dict, Optional, Callable, TYPE_CHECKING
import torch

from .model_group import ModelGroup
from .weight_sync import BaseWeightSync
from data import DataHub, Batch
from config import TrainingConfig

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from algorithm import BaseAlgorithm


class TrainingLoop:
    """
    训练循环
    
    负责:
    - 从 DataHub 采样数据
    - 调用算法训练
    - 同步权重到推理端
    - 记录日志
    
    Example:
        loop = TrainingLoop(
            algorithm=sac,
            data_hub=data_hub,
            config=config,
        )
        loop.train(num_steps=10000)
    """
    
    def __init__(self,
                 algorithm: "BaseAlgorithm",
                 data_hub: DataHub,
                 config: TrainingConfig,
                 weight_sync: Optional[BaseWeightSync] = None,
                 device: str = "cuda"):
        """
        Args:
            algorithm: 训练算法
            data_hub: 数据中心
            config: 训练配置
            weight_sync: 权重同步器
            device: 训练设备
        """
        self.algorithm = algorithm
        self.data_hub = data_hub
        self.config = config
        self.weight_sync = weight_sync
        self.device = device
        
        self._step = 0
        self._callbacks: Dict[str, Callable] = {}
    
    def train(self, 
              num_steps: int,
              sample_strategy: str = "demo_only",
              log_freq: int = 100,
              sync_freq: int = 100,
              checkpoint_freq: int = 1000,
              checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """
        执行训练
        
        Args:
            num_steps: 训练步数
            sample_strategy: 采样策略
            log_freq: 日志频率
            sync_freq: 权重同步频率
            checkpoint_freq: checkpoint 保存频率
            checkpoint_path: checkpoint 保存路径
            
        Returns:
            训练统计信息
        """
        total_metrics = {}
        
        for step in range(num_steps):
            # 采样
            batch = self.data_hub.sample(
                batch_size=self.config.batch_size,
                strategy=sample_strategy,
            )
            batch = batch.to(self.device)
            
            # 训练一步
            metrics = self.algorithm.train_step(batch)
            
            # 累积 metrics
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            
            self._step += 1
            
            # 日志
            if self._step % log_freq == 0:
                avg_metrics = {k: v / log_freq for k, v in total_metrics.items()}
                self._log(avg_metrics)
                total_metrics = {}
            
            # 权重同步
            if self.weight_sync and self._step % sync_freq == 0:
                self._sync_weights()
            
            # Checkpoint
            if checkpoint_path and self._step % checkpoint_freq == 0:
                self._save_checkpoint(checkpoint_path)
            
            # 回调
            if "on_step" in self._callbacks:
                self._callbacks["on_step"](self._step, metrics)
        
        return {"total_steps": self._step}
    
    def _log(self, metrics: Dict[str, float]):
        """打印日志"""
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"[Step {self._step}] {metrics_str}")
    
    def _sync_weights(self):
        """同步权重"""
        policy = self.algorithm.model_group.get("policy")
        if policy:
            self.weight_sync.push(
                {"policy": policy.state_dict()},
                version=self._step,
            )
    
    def _save_checkpoint(self, path: str):
        """保存 checkpoint"""
        save_path = f"{path}_step{self._step}.pt"
        self.algorithm.model_group.save(save_path)
        print(f"[Checkpoint] Saved to {save_path}")
    
    def register_callback(self, name: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[name] = callback
    
    @property
    def step(self) -> int:
        return self._step