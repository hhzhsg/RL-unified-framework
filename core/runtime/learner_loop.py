"""
Learner Loop（适用于 Offline / Online RL，无人类干预）

业界标准命名：Actor-Learner 架构
- Actor: 与环境交互，收集数据
- Learner: 从数据学习，更新策略
- Evaluator: 评估策略性能

适用场景:
- Offline RL: 从离线数据集训练
- Online RL: 配合 ActorLoop 使用（ActorLoop 收集数据 → LearnerLoop 训练）

与 HIL 的关系:
- HIL 场景请使用 HILLearnerLoop，它额外负责:
  - 接收 Actor 发送的 transitions
  - 将 intervention 和 rollout 分流到不同 buffer
  - 加权采样（intervention 2x）
  - 发布权重到 Actor

职责:
- 从 DataHub 采样数据
- 调用 Algorithm.update() 更新策略
- 定期保存 checkpoint
- 可选：同步权重到 Actor
"""
from typing import Dict, Any, Optional
import os
import torch

from .base_loop import BaseLoop
from ..interfaces import AlgorithmInterface, SamplerInterface, SyncInterface


class LearnerLoop(BaseLoop):
    """
    Learner Loop - 从数据学习
    
    职责:
    - 从 DataHub 采样数据
    - 调用 Algorithm.update() 更新策略
    - 同步权重到 Actor
    - 保存 checkpoint
    
    使用示例:
        learner = LearnerLoop(algorithm, data_hub, sampler, config, weight_sync)
        learner.run(num_steps=100000)
    """
    
    def __init__(
        self,
        algorithm: AlgorithmInterface,
        data_hub: Any,  # DataHub
        sampler: SamplerInterface,
        config: Dict[str, Any],
        weight_sync: Optional[SyncInterface] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.algorithm = algorithm
        self.data_hub = data_hub
        self.sampler = sampler
        self.config = config
        self.weight_sync = weight_sync
        self.device = device
        self._manual_step_increment = True
        
        # 配置
        self.batch_size = config.get("batch_size", 256)
        self.sync_freq = config.get("sync_freq", 100)
        self.checkpoint_freq = config.get("checkpoint_freq", 1000)
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    
    def step(self) -> Dict[str, Any]:
        """执行单步学习"""
        # 1. 采样
        batch = self.sampler.sample(self.data_hub.buffers, self.batch_size)
        batch = self._to_device(batch)

        # 2. 更新
        metrics = self.algorithm.update(batch)

        # 3. 递增步计数
        self._step_count += 1

        # 4. 同步权重到 Actor
        if self.weight_sync and self._step_count % self.sync_freq == 0:
            self._sync_weights()

        # 5. 保存 checkpoint
        if self._step_count % self.checkpoint_freq == 0:
            self._save_checkpoint(self._step_count)

        return metrics
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """移动数据到设备"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def _sync_weights(self) -> None:
        """同步权重到 Actor"""
        policy = self.algorithm.get_policy()
        state_dict = policy.state_dict()
        self.weight_sync.push(
            {"policy": state_dict},
            tag="policy_weights"
        )
    
    def _save_checkpoint(self, step: int | None = None) -> None:
        """保存 checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        use_step = step if step is not None else self._step_count
        path = f"{self.checkpoint_dir}/step_{use_step}.pt"
        self.algorithm.save(path)
        print(f"[Learner] Checkpoint saved: {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "train_steps": self._step_count,
        }


# 向后兼容别名
TrainingLoop = LearnerLoop
