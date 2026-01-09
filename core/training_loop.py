"""
VLA-RL 训练循环
"""
from typing import List, Optional, Callable, Dict, Any
import time

from .stage import Stage
from .weight_sync import BaseWeightSync
from algorithm import BaseAlgorithm, create_algorithm
from model import ModelGroup
from buffer import DataHub
from data import Batch
from config import TrainingConfig, StageConfig


class TrainingLoop:
    """
    训练循环
    支持单阶段/多阶段训练
    """
    
    def __init__(self,
                 model_group: ModelGroup,
                 data_hub: DataHub,
                 config: TrainingConfig,
                 algo_config: 'AlgorithmConfig' = None,
                 weight_sync: Optional[BaseWeightSync] = None,
                 device: str = "cuda"):
        """
        Args:
            model_group: 模型组
            data_hub: 数据中心
            config: 训练配置
            algo_config: 算法配置（学习率、batch_size等）
            weight_sync: 权重同步器 (可选，用于 Online 模式)
            device: 训练设备
        """
        self.model_group = model_group
        self.data_hub = data_hub
        self.config = config
        self.algo_config = algo_config
        self.weight_sync = weight_sync
        self.device = device
        
        # 构建 stages
        self.stages: List[Stage] = [
            Stage.from_config(stage_config) 
            for stage_config in config.stages
        ]
        
        # 当前阶段索引
        self.current_stage_idx = 0
        
        # 算法实例缓存 (每个阶段可能用不同算法)
        self._algorithms: Dict[str, BaseAlgorithm] = {}
        
        # 运行状态
        self._running = False
        self._total_steps = 0
        self._weight_version = 0
    
    @property
    def current_stage(self) -> Optional[Stage]:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def run(self, callback: Optional[Callable[[int, Dict], None]] = None):
        """
        运行训练
        
        Args:
            callback: 可选回调，每步调用 callback(step, metrics)
        """
        self._running = True
        self.model_group.to(self.device)
        
        print(f"[Training] Starting with {len(self.stages)} stage(s)")
        
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            print(f"\n[Training] Stage {stage_idx + 1}/{len(self.stages)}: {stage.name}")
            print(f"  Algorithm: {stage.algorithm_name}")
            print(f"  Max steps: {stage.max_steps}")
            print(f"  Active models: {stage.active_models}")
            print(f"  Sample strategy: {stage.sample_strategy}")
            
            # 获取/创建算法
            algorithm = self._get_algorithm(stage)
            
            # 设置模型冻结状态
            self._setup_model_freeze(stage)
            
            # 运行该阶段
            self._run_stage(stage, algorithm, callback)
            
            if not self._running:
                break
        
        print(f"\n[Training] Finished. Total steps: {self._total_steps}")
    
    def _run_stage(self, stage: Stage, algorithm: BaseAlgorithm, 
                   callback: Optional[Callable]):
        """运行单个阶段"""
        while not stage.is_finished and self._running:
            # 采样
            try:
                batch = self.data_hub.sample(
                    batch_size=algorithm.config.batch_size,
                    strategy=stage.sample_strategy,
                    **stage.sample_kwargs
                )
            except ValueError as e:
                print(f"[Training] Warning: {e}. Waiting for data...")
                time.sleep(0.1)
                continue
            
            # 训练
            batch = batch.to(self.device)
            metrics = algorithm.train_step(batch)
            
            # 更新计数
            stage.current_step += 1
            self._total_steps += 1
            
            # 日志
            if self._total_steps % self.config.log_freq == 0:
                print(f"[Training] Step {self._total_steps} | Stage {stage.name} "
                      f"({stage.current_step}/{stage.max_steps}) | {metrics}")
            
            # 同步权重
            if self.weight_sync and self._total_steps % self.config.save_freq == 0:
                self._sync_weights()
            
            # 回调
            if callback:
                callback(self._total_steps, metrics)
    
    def _get_algorithm(self, stage: Stage) -> BaseAlgorithm:
        """获取或创建算法实例"""
        if stage.algorithm_name not in self._algorithms:
            algorithm = create_algorithm(
                stage.algorithm_name,
                self.model_group,
                config=self.algo_config,
            )
            self._algorithms[stage.algorithm_name] = algorithm
        
        return self._algorithms[stage.algorithm_name]
    
    def _setup_model_freeze(self, stage: Stage):
        """设置该阶段的模型冻结状态"""
        # 先冻结所有
        for name in self.model_group.model_names:
            self.model_group.freeze(name)
        
        # 解冻活跃模型
        for name in stage.active_models:
            if name in self.model_group:
                self.model_group.unfreeze(name)
    
    def _sync_weights(self):
        """同步权重到推理端"""
        if self.weight_sync is None:
            return
        
        self._weight_version += 1
        state_dict = self.model_group.state_dict()
        self.weight_sync.push(state_dict, self._weight_version)
        print(f"[Training] Pushed weights v{self._weight_version}")
    
    def stop(self):
        """停止训练"""
        self._running = False
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        import torch
        import os
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        checkpoint = {
            "model_group": self.model_group.state_dict(),
            "total_steps": self._total_steps,
            "current_stage_idx": self.current_stage_idx,
            "stages": [(s.name, s.current_step) for s in self.stages],
        }
        torch.save(checkpoint, path)
        print(f"[Training] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        import torch
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model_group.load_state_dict(checkpoint["model_group"])
        self._total_steps = checkpoint["total_steps"]
        self.current_stage_idx = checkpoint["current_stage_idx"]
        
        # 恢复阶段进度
        for (name, step), stage in zip(checkpoint["stages"], self.stages):
            if stage.name == name:
                stage.current_step = step
        
        print(f"[Training] Loaded checkpoint from {path}, step {self._total_steps}")
