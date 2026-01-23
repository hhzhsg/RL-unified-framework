"""
HIL-SERL Learner Loop

负责:
1. 管理三个Replay Buffer（demo, rollout, intervention）
2. 接收Actor发送的transitions
3. RLPD加权采样（intervention 2x权重）
4. SAC策略更新
5. 定期发布权重到Actor
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import torch
import numpy as np

from core.runtime.base_loop import BaseLoop
from core.interfaces import AlgorithmInterface


@dataclass
class HILSerlLearnerConfig:
    """HIL-SERL Learner配置"""
    # 训练参数
    batch_size: int = 256
    utd_ratio: int = 1  # Update-to-Data ratio
    
    # 采样权重（RLPD）
    demo_weight: float = 1.0
    rollout_weight: float = 1.0
    intervention_weight: float = 2.0  # 关键：人类干预数据2倍权重
    
    # Buffer
    demo_buffer_path: Optional[str] = None
    rollout_buffer_capacity: int = 100000
    intervention_buffer_capacity: int = 50000
    
    # 训练启动条件
    training_starts: int = 1000  # 最小数据量
    
    # 权重发布
    policy_push_frequency: int = 100  # 每N步发布权重
    
    # Checkpoint
    checkpoint_freq: int = 1000
    checkpoint_dir: str = "./checkpoints/hilserl"
    
    # 设备
    device: str = "cuda"


class HILSerlLearnerLoop(BaseLoop):
    """
    HIL-SERL Learner循环
    
    核心职责:
    - 管理三个Buffer
    - 接收并路由transitions
    - RLPD加权采样
    - SAC更新
    - 发布权重
    """
    
    def __init__(
        self,
        algorithm: AlgorithmInterface,
        learner_server,  # LearnerServerInterface
        config: HILSerlLearnerConfig,
        demo_buffer: Optional[Any] = None,
    ):
        super().__init__()
        
        self.algorithm = algorithm
        self.server = learner_server
        self.config = config
        
        # 初始化三个Buffer
        from data.buffers.replay_buffer import ReplayBuffer
        from data.buffers.intervention_buffer import InterventionBuffer
        
        self.demo_buffer = demo_buffer  # 可以预加载
        self.rollout_buffer = ReplayBuffer(capacity=config.rollout_buffer_capacity)
        self.intervention_buffer = InterventionBuffer(capacity=config.intervention_buffer_capacity)
        
        # 采样器
        from data.samplers.hilserl_sampler import HILSERLSampler
        self.sampler = HILSERLSampler(weights={
            "demo": config.demo_weight,
            "rollout": config.rollout_weight,
            "intervention": config.intervention_weight,
        })
        
        # 状态
        self._training_started = False
        self._initial_weights_published = False
        self.device = torch.device(config.device)
        
        # 统计
        self._transitions_received = 0
        self._interventions_received = 0
    
    def wait_for_minimum_data(self, timeout: float = 300.0) -> bool:
        """
        等待最小数据量
        
        持续接收transitions直到达到training_starts阈值
        
        Args:
            timeout: 超时时间（秒）
        Returns:
            是否收集到足够数据
        """
        print(f"[Learner] Waiting for {self.config.training_starts} transitions...")
        start_time = time.time()
        
        while self._get_total_online_size() < self.config.training_starts:
            if time.time() - start_time > timeout:
                print("[Learner] Timeout waiting for data")
                return False
            
            # 接收并路由transitions
            transitions = self.server.recv_transitions(block=True, timeout=1.0)
            self._route_transitions(transitions)
            
            # 进度报告
            current_size = self._get_total_online_size()
            if current_size % 100 == 0 and current_size > 0:
                print(f"[Learner] Collected {current_size}/{self.config.training_starts}")
        
        print(f"[Learner] Minimum data collected: {self._get_total_online_size()}")
        self._training_started = True
        return True
    
    def publish_initial_weights(self) -> None:
        """发布初始权重到Actor"""
        policy = self.algorithm.get_policy()
        state_dict = policy.state_dict()
        self.server.publish_weights(state_dict, metadata={"step": 0, "type": "initial"})
        self._initial_weights_published = True
        print("[Learner] Initial weights published")
    
    def step(self) -> Dict[str, Any]:
        """执行单步Learner逻辑"""
        # 0. 确保已启动训练
        if not self._training_started:
            raise RuntimeError("Must call wait_for_minimum_data() first")
        if not self._initial_weights_published:
            raise RuntimeError("Must call publish_initial_weights() first")
        
        # 1. 接收新transitions（非阻塞）
        transitions = self.server.recv_transitions(block=False)
        self._route_transitions(transitions)
        
        # 2. UTD更新
        metrics_list = []
        for _ in range(self.config.utd_ratio):
            # 2a. 采样batch
            batch = self._sample_batch()
            if batch is None:
                break
            
            # 2b. 移动到设备
            batch = self._to_device(batch)
            
            # 2c. SAC更新
            metrics = self.algorithm.update(batch)
            metrics_list.append(metrics)
        
        # 3. 合并metrics
        if metrics_list:
            avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        else:
            avg_metrics = {}
        
        # 4. 发布权重
        if self._step_count % self.config.policy_push_frequency == 0:
            self._publish_weights()
        
        # 5. 保存checkpoint
        if self._step_count % self.config.checkpoint_freq == 0:
            self._save_checkpoint()
        
        # 添加统计信息
        avg_metrics.update({
            "rollout_buffer_size": len(self.rollout_buffer),
            "intervention_buffer_size": len(self.intervention_buffer),
            "demo_buffer_size": len(self.demo_buffer) if self.demo_buffer else 0,
            "intervention_ratio": self._interventions_received / max(self._transitions_received, 1),
        })
        
        return avg_metrics
    
    def _route_transitions(self, transitions: List[Dict[str, Any]]) -> None:
        """将transitions路由到正确的buffer"""
        for t in transitions:
            self._transitions_received += 1
            
            is_intervention = t.get("is_intervention", False)
            if is_intervention:
                self._interventions_received += 1
                self.intervention_buffer.add(t)
            else:
                self.rollout_buffer.add(t)
    
    def _sample_batch(self) -> Optional[Dict[str, Any]]:
        """RLPD加权采样"""
        # 构建buffer字典
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
        """移动batch到训练设备"""
        result = {}
        for k, v in batch.items():
            # numpy arrays with numeric dtype -> convert to torch
            if isinstance(v, np.ndarray):
                try:
                    if np.issubdtype(v.dtype, np.number):
                        result[k] = torch.from_numpy(v).float().to(self.device)
                    else:
                        # non-numeric arrays (e.g. strings) keep as-is
                        result[k] = v
                except Exception:
                    # fallback: keep original if conversion fails
                    result[k] = v
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def _publish_weights(self) -> None:
        """发布最新权重"""
        policy = self.algorithm.get_policy()
        state_dict = policy.state_dict()
        self.server.publish_weights(
            state_dict,
            metadata={"step": self._step_count}
        )
    
    def _save_checkpoint(self) -> None:
        """保存checkpoint"""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = f"{self.config.checkpoint_dir}/step_{self._step_count}.pt"
        self.algorithm.save(path)
        print(f"[Learner] Checkpoint saved: {path}")
    
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


# ============ 高级封装：HIL-SERL训练器 ============

class HILSerlTrainer:
    """
    HIL-SERL完整训练器
    
    封装Actor和Learner的启动与协调
    """
    
    def __init__(
        self,
        policy,
        env,
        algorithm,
        config: Dict[str, Any],
        demo_buffer = None,
        reward_classifier = None,
        mode: str = "local",  # "local" | "grpc"
    ):
        self.policy = policy
        self.env = env
        self.algorithm = algorithm
        self.config = config
        self.demo_buffer = demo_buffer
        self.reward_classifier = reward_classifier
        self.mode = mode
        
        # 创建通信组件
        from actor_learner_sync import (
            ActorLearnerConfig,
            create_learner_server,
            create_actor_client,
        )
        
        self.sync_config = ActorLearnerConfig(**config.get("sync", {}))
        self.server = create_learner_server(mode, self.sync_config)
        
        # local模式下，client需要server引用
        if mode == "local":
            self.client = create_actor_client(mode, server=self.server)
        else:
            self.client = create_actor_client(mode, config=self.sync_config)
    
    def run_local(self, num_steps: int):
        """
        本地模式运行（单进程，用于调试）
        
        交替执行Actor和Learner步骤
        """
        # 配置
        actor_config = HILSerlActorConfig(**self.config.get("actor", {}))
        learner_config = HILSerlLearnerConfig(**self.config.get("learner", {}))
        
        # 创建循环
        actor_loop = HILSerlActorLoop(
            policy=self.policy,
            env=self.env,
            actor_client=self.client,
            config=actor_config,
            reward_classifier=self.reward_classifier,
        )
        
        learner_loop = HILSerlLearnerLoop(
            algorithm=self.algorithm,
            learner_server=self.server,
            config=learner_config,
            demo_buffer=self.demo_buffer,
        )
        
        # 启动服务器
        self.server.start()
        self.client.connect()
        
        # 发布初始权重
        learner_loop.publish_initial_weights()
        
        # Actor同步初始权重
        actor_loop.wait_for_initial_weights(timeout=5.0)
        
        # 收集初始数据
        print("[Trainer] Collecting initial data...")
        for _ in range(learner_config.training_starts):
            actor_loop.step()
        
        learner_loop._training_started = True
        learner_loop._initial_weights_published = True
        
        # 主训练循环
        print(f"[Trainer] Starting training for {num_steps} steps...")
        for step in range(num_steps):
            # Actor执行
            actor_info = actor_loop.step()
            
            # Learner更新
            learner_info = learner_loop.step()
            
            # 日志
            if step % 100 == 0:
                print(f"[Step {step}] Actor: reward={actor_info.get('reward', 0):.2f}, "
                      f"Learner: critic_loss={learner_info.get('critic_loss', 0):.4f}")
        
        # 清理
        self.server.stop()
        self.client.disconnect()
        
        return {
            "actor_stats": actor_loop.get_statistics(),
            "learner_stats": learner_loop.get_statistics(),
        }
    
    def run_distributed(self, role: str, num_steps: int):
        """
        分布式模式运行
        
        Args:
            role: "actor" | "learner"
            num_steps: 运行步数
        """
        if role == "learner":
            return self._run_learner(num_steps)
        elif role == "actor":
            return self._run_actor(num_steps)
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def _run_learner(self, num_steps: int):
        """运行Learner进程"""
        learner_config = HILSerlLearnerConfig(**self.config.get("learner", {}))
        
        learner_loop = HILSerlLearnerLoop(
            algorithm=self.algorithm,
            learner_server=self.server,
            config=learner_config,
            demo_buffer=self.demo_buffer,
        )
        
        self.server.start()
        
        # 等待数据
        if not learner_loop.wait_for_minimum_data():
            raise RuntimeError("Failed to collect minimum data")
        
        # 发布初始权重
        learner_loop.publish_initial_weights()
        
        # 训练
        results = learner_loop.run(num_steps, log_freq=100)
        
        self.server.stop()
        return results
    
    def _run_actor(self, num_steps: int):
        """运行Actor进程"""
        actor_config = HILSerlActorConfig(**self.config.get("actor", {}))
        
        actor_loop = HILSerlActorLoop(
            policy=self.policy,
            env=self.env,
            actor_client=self.client,
            config=actor_config,
            reward_classifier=self.reward_classifier,
        )
        
        self.client.connect()
        
        # 等待初始权重
        if not actor_loop.wait_for_initial_weights():
            raise RuntimeError("Failed to receive initial weights")
        
        # 执行
        results = actor_loop.run(num_steps, log_freq=100)
        
        self.client.disconnect()
        return results
