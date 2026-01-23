"""
HIL Learner Loop

Human-in-the-Loop Learner 循环，与具体模型/算法解耦

负责:
1. 管理多个 Replay Buffer（demo, rollout, intervention）
2. 接收 Actor 发送的 transitions
3. 加权采样（RLPD：intervention 2x 权重）
4. 调用算法更新
5. 定期发布权重到 Actor

模型无关：通过 TrainableAdapter 接口接入任意算法
"""
from typing import Dict, Any, Optional, List, Protocol, Callable
from dataclasses import dataclass
import time
import os
import torch
import numpy as np

from core.runtime.base_loop import BaseLoop


# ============ 协议定义 ============

class TrainableAdapterProtocol(Protocol):
    """可训练适配器协议"""
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]: ...
    def get_weights(self) -> Dict[str, torch.Tensor]: ...
    def save(self, path: str) -> None: ...


class LearnerServerProtocol(Protocol):
    """Learner 服务器协议"""
    def recv_transitions(self, block: bool = False, timeout: float = 0.1) -> List[Dict[str, Any]]: ...
    def publish_weights(self, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None) -> None: ...


class BufferProtocol(Protocol):
    """Buffer 协议"""
    def add(self, data: Dict[str, Any]) -> None: ...
    def sample(self, batch_size: int) -> Dict[str, Any]: ...
    def __len__(self) -> int: ...


class SamplerProtocol(Protocol):
    """采样器协议"""
    def sample(self, buffers: Dict[str, Any], batch_size: int) -> Dict[str, Any]: ...


# ============ 配置 ============

@dataclass
class HILLearnerConfig:
    """HIL Learner 配置"""
    # 训练参数
    batch_size: int = 256
    utd_ratio: int = 1  # Update-to-Data ratio
    
    # 采样权重（RLPD）
    demo_weight: float = 1.0
    rollout_weight: float = 1.0
    intervention_weight: float = 2.0  # 关键：人类干预数据 2 倍权重
    
    # Buffer 容量
    rollout_buffer_capacity: int = 100000
    intervention_buffer_capacity: int = 50000
    
    # 训练启动条件
    training_starts: int = 1000  # 最小数据量
    
    # 权重发布
    policy_push_frequency: int = 100  # 每 N 步发布权重
    
    # Checkpoint
    checkpoint_freq: int = 1000
    checkpoint_dir: str = "./checkpoints/hil"
    
    # 设备
    device: str = "cuda"


# ============ HIL Learner Loop ============

class HILLearnerLoop(BaseLoop):
    """
    Human-in-the-Loop Learner 循环
    
    与算法解耦的设计：
    - 通过 TrainableAdapter 接口接入任意算法
    - 算法只需实现 update() / get_weights() / save()
    
    使用示例：
        # 使用 SAC 算法
        from core.interfaces.adapters import AlgorithmAdapter
        adapter = AlgorithmAdapter(sac_algorithm)
        learner = HILLearnerLoop(adapter, server, config)
        
        # 使用自定义 pi0 训练器
        adapter = Pi0TrainerAdapter(pi0_trainer, sync_mode="lora")
        learner = HILLearnerLoop(adapter, server, config)
    """
    
    def __init__(
        self,
        trainable_adapter: TrainableAdapterProtocol,
        learner_server: LearnerServerProtocol,
        config: HILLearnerConfig,
        demo_buffer: Optional[BufferProtocol] = None,
        custom_sampler: Optional[SamplerProtocol] = None,
        rollout_buffer: Optional[BufferProtocol] = None,
        intervention_buffer: Optional[BufferProtocol] = None,
    ):
        super().__init__()
        
        self.adapter = trainable_adapter
        self.server = learner_server
        self.config = config
        
        # 初始化 Buffer（允许外部注入）
        if rollout_buffer is not None:
            self.rollout_buffer = rollout_buffer
        else:
            from data.buffers.replay_buffer import ReplayBuffer
            self.rollout_buffer = ReplayBuffer(capacity=config.rollout_buffer_capacity)
        
        if intervention_buffer is not None:
            self.intervention_buffer = intervention_buffer
        else:
            from data.buffers.intervention_buffer import InterventionBuffer
            self.intervention_buffer = InterventionBuffer(capacity=config.intervention_buffer_capacity)
        
        self.demo_buffer = demo_buffer
        
        # 采样器（允许外部注入）
        if custom_sampler is not None:
            self.sampler = custom_sampler
        else:
            from data.samplers.hilserl_sampler import HILSERLSampler
            self.sampler = HILSERLSampler(weights={
                "demo": config.demo_weight,
                "rollout": config.rollout_weight,
                "intervention": config.intervention_weight,
            })
        
        # 状态
        self._training_started = False
        self._initial_weights_published = False
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 统计
        self._transitions_received = 0
        self._interventions_received = 0
    
    def wait_for_minimum_data(self, timeout: float = 300.0) -> bool:
        """
        等待最小数据量
        
        持续接收 transitions 直到达到 training_starts 阈值
        """
        print(f"[HIL-Learner] Waiting for {self.config.training_starts} transitions...")
        start_time = time.time()
        last_report = 0
        
        while self._get_total_online_size() < self.config.training_starts:
            if time.time() - start_time > timeout:
                print("[HIL-Learner] Timeout waiting for data")
                return False
            
            # 接收并路由 transitions
            transitions = self.server.recv_transitions(block=True, timeout=1.0)
            self._route_transitions(transitions)
            
            # 进度报告（每 100 条）
            current_size = self._get_total_online_size()
            if current_size >= last_report + 100:
                print(f"[HIL-Learner] Collected {current_size}/{self.config.training_starts}")
                last_report = current_size
        
        print(f"[HIL-Learner] Minimum data collected: {self._get_total_online_size()}")
        self._training_started = True
        return True
    
    def publish_initial_weights(self) -> None:
        """发布初始权重到 Actor"""
        weights = self.adapter.get_weights()
        self.server.publish_weights(weights, metadata={"step": 0, "type": "initial"})
        self._initial_weights_published = True
        print("[HIL-Learner] Initial weights published")
    
    def start_training(self) -> None:
        """
        直接启动训练（跳过数据等待）
        
        用于已有数据或不需要等待的场景
        """
        self._training_started = True
        if not self._initial_weights_published:
            self.publish_initial_weights()
    
    def step(self) -> Dict[str, Any]:
        """执行单步 Learner 逻辑"""
        # 0. 检查初始化
        if not self._training_started:
            raise RuntimeError("Must call wait_for_minimum_data() or start_training() first")
        
        # 1. 接收新 transitions（非阻塞）
        transitions = self.server.recv_transitions(block=False)
        self._route_transitions(transitions)
        
        # 2. UTD 更新
        metrics_list = []
        for _ in range(self.config.utd_ratio):
            batch = self._sample_batch()
            if batch is None:
                break
            
            batch = self._to_device(batch)
            metrics = self.adapter.update(batch)
            metrics_list.append(metrics)
        
        # 3. 合并 metrics
        if metrics_list:
            avg_metrics = {}
            for key in metrics_list[0]:
                values = [m[key] for m in metrics_list if key in m]
                if values and isinstance(values[0], (int, float)):
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}
        
        # 4. 发布权重
        if self._step_count > 0 and self._step_count % self.config.policy_push_frequency == 0:
            self._publish_weights()
        
        # 5. 保存 checkpoint
        if self._step_count > 0 and self._step_count % self.config.checkpoint_freq == 0:
            self._save_checkpoint()
        
        # 6. 添加统计信息
        avg_metrics.update({
            "rollout_buffer_size": len(self.rollout_buffer),
            "intervention_buffer_size": len(self.intervention_buffer),
            "demo_buffer_size": len(self.demo_buffer) if self.demo_buffer else 0,
            "transitions_received": self._transitions_received,
            "intervention_ratio": self._interventions_received / max(self._transitions_received, 1),
        })
        
        return avg_metrics
    
    def _route_transitions(self, transitions: List[Dict[str, Any]]) -> None:
        """将 transitions 路由到正确的 buffer"""
        for t in transitions:
            self._transitions_received += 1
            
            is_intervention = t.get("is_intervention", False)
            if is_intervention:
                self._interventions_received += 1
                self.intervention_buffer.add(t)
            else:
                self.rollout_buffer.add(t)
    
    def _sample_batch(self) -> Optional[Dict[str, Any]]:
        """加权采样"""
        buffers = {}
        if self.demo_buffer is not None and len(self.demo_buffer) > 0:
            buffers["demo"] = self.demo_buffer
        if len(self.rollout_buffer) > 0:
            buffers["rollout"] = self.rollout_buffer
        if len(self.intervention_buffer) > 0:
            buffers["intervention"] = self.intervention_buffer
        
        if not buffers:
            return None
        
        try:
            return self.sampler.sample(buffers, self.config.batch_size)
        except ValueError:
            return None
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """移动 batch 到训练设备"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                try:
                    if np.issubdtype(v.dtype, np.number):
                        result[k] = torch.from_numpy(v).float().to(self.device)
                    else:
                        result[k] = v
                except Exception:
                    result[k] = v
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def _publish_weights(self) -> None:
        """发布最新权重"""
        weights = self.adapter.get_weights()
        self.server.publish_weights(weights, metadata={"step": self._step_count})
    
    def _save_checkpoint(self) -> None:
        """保存 checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = f"{self.config.checkpoint_dir}/step_{self._step_count}.pt"
        self.adapter.save(path)
        print(f"[HIL-Learner] Checkpoint saved: {path}")
    
    def _get_total_online_size(self) -> int:
        """获取在线数据总量"""
        return len(self.rollout_buffer) + len(self.intervention_buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "train_steps": self._step_count,
            "rollout_buffer_size": len(self.rollout_buffer),
            "intervention_buffer_size": len(self.intervention_buffer),
            "demo_buffer_size": len(self.demo_buffer) if self.demo_buffer else 0,
            "transitions_received": self._transitions_received,
            "interventions_received": self._interventions_received,
            "intervention_ratio": self._interventions_received / max(self._transitions_received, 1),
        }
    
    def add_demo_data(self, transitions: List[Dict[str, Any]]) -> None:
        """手动添加 demo 数据"""
        if self.demo_buffer is None:
            from data.buffers.replay_buffer import ReplayBuffer
            self.demo_buffer = ReplayBuffer(capacity=len(transitions) * 2)
        
        for t in transitions:
            t["source"] = "demo"
            self.demo_buffer.add(t)


# ============ 向后兼容别名 ============
HILSerlLearnerLoop = HILLearnerLoop
HILSerlLearnerConfig = HILLearnerConfig
