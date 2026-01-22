"""
训练循环

负责从Buffer采样数据，调用Algorithm更新
"""
from typing import Dict, Any, Optional
import torch

from .base_loop import BaseLoop
from ..interfaces import AlgorithmInterface, SamplerInterface, SyncInterface


class TrainingLoop(BaseLoop):
    """
    训练循环
    
    职责:
    - 从DataHub采样数据
    - 调用Algorithm更新
    - 同步权重到推理端
    - 保存checkpoint
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
        """执行单步训练：先采样/更新，随后由本类递增步计数并在递增后执行同步与保存。"""

        # 1. 采样
        batch = self.sampler.sample(self.data_hub.buffers, self.batch_size)
        batch = self._to_device(batch)

        # 2. 更新
        metrics = self.algorithm.update(batch)

        # 3. 由 TrainingLoop 自行递增步计数（避免 BaseLoop 再次自增）
        self._step_count += 1

        # 4. 同步权重（基于已完成的步数）
        if self.weight_sync and self._step_count % self.sync_freq == 0:
            self._sync_weights()

        # 5. 保存 checkpoint（基于已完成的步数）
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
        """同步权重"""
        policy = self.algorithm.get_policy()
        state_dict = policy.state_dict()
        self.weight_sync.push(
            {"policy": state_dict},
            tag="policy_weights"
        )
    
    def _save_checkpoint(self, step: int | None = None) -> None:
        """保存checkpoint。若提供 `step` 则使用该值作为文件名，否则使用当前计数。"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        use_step = step if step is not None else self._step_count
        path = f"{self.checkpoint_dir}/step_{use_step}.pt"
        self.algorithm.save(path)
        print(f"[Checkpoint] Saved to {path}")
