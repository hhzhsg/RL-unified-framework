"""
HIL Loop（适用于 Human-in-the-Loop 分布式训练）

设计理念：
- Actor 和 Learner 是独立进程，通过 gRPC/本地队列通信
- 不需要"协调器"，各自独立运行
- 通过 config 中的 role 参数决定启动哪个

使用方式：
    # 终端 1：启动 Learner（通常有 GPU）
    python scripts/train.py --config xxx.yaml --role learner
    
    # 终端 2：启动 Actor（连接机器人）
    python scripts/train.py --config xxx.yaml --role actor

通信架构：
    Actor ──(transitions)──► Learner
    Actor ◄──(weights)────── Learner
"""
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field
import time
import os
import numpy as np
import torch

from .base_loop import BaseLoop
from core.synchronization.actor_learner import (
    ActorLearnerConfig,
    create_learner_server,
    create_actor_client,
)


# ============ 协议定义 ============

class PolicyAdapterProtocol(Protocol):
    """策略适配器协议（Actor 使用）"""
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any: ...
    def get_weights(self) -> Dict[str, torch.Tensor]: ...
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None: ...


class TrainableAdapterProtocol(Protocol):
    """可训练适配器协议（Learner 使用）"""
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]: ...
    def get_weights(self) -> Dict[str, torch.Tensor]: ...
    def save(self, path: str) -> None: ...


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
class HILActorConfig:
    """HIL Actor 配置"""
    deterministic: bool = False
    max_episode_steps: int = 200
    weight_sync_freq: int = 1
    transition_batch_size: int = 1
    require_initial_weights: bool = True
    # 奖励分类器
    use_reward_classifier: bool = False
    reward_classifier_threshold: float = 0.5
    camera_keys: List[str] = field(default_factory=lambda: ["images.front", "images.side"])


@dataclass
class HILLearnerConfig:
    """HIL Learner 配置"""
    batch_size: int = 256
    utd_ratio: int = 1
    training_starts: int = 1000
    policy_push_frequency: int = 100
    checkpoint_freq: int = 1000
    checkpoint_dir: str = "./checkpoints/hil"
    device: str = "cuda"
    # 采样权重
    demo_weight: float = 1.0
    rollout_weight: float = 1.0
    intervention_weight: float = 2.0
    # Buffer 容量
    rollout_buffer_capacity: int = 100000
    intervention_buffer_capacity: int = 50000


# ============ HIL Actor Loop ============

class HILActorLoop(BaseLoop):
    """
    HIL Actor 循环（独立进程运行）
    
    职责：
    - 与环境交互
    - 从 Wrapper 获取干预信息
    - 发送 transitions 到 Learner
    - 接收 Learner 的最新权重
    
    使用示例：
        actor = HILActorLoop(
            policy_adapter=adapter,
            env=env,
            config=config,
            sync_config=sync_config,
        )
        actor.run(num_steps=10000)
    """
    
    def __init__(
        self,
        policy_adapter: PolicyAdapterProtocol,
        env,  # EnvInterface，可能被 Wrapper 包装
        config: HILActorConfig,
        sync_config: Optional[ActorLearnerConfig] = None,
        reward_classifier=None,
        mode: str = "local",  # "local" | "grpc"
        actor_client=None,  # 可选：外部注入的 client（用于测试）
    ):
        super().__init__()
        # 手动管理 step_count，防止 BaseLoop.run() 重复递增
        self._manual_step_increment = True
        
        self.policy = policy_adapter
        self.env = env
        self.config = config
        self.reward_classifier = reward_classifier
        
        # 创建或使用注入的通信客户端
        if actor_client is not None:
            self.client = actor_client
        else:
            if sync_config is None:
                sync_config = ActorLearnerConfig()
            self.client = create_actor_client(mode, config=sync_config)
        
        # 状态
        self._current_obs = None
        self._current_info = None
        self._episode_step = 0
        self._episode_count = 0
        self._episode_reward = 0.0
        
        # 统计
        self._intervention_count = 0
        self._total_actions = 0
        self._weight_sync_count = 0
        
        # Transition 缓冲
        self._transition_buffer: List[Dict[str, Any]] = []
        
        # 初始权重状态
        self._initial_weights_synced = not config.require_initial_weights
    
    def setup(self) -> None:
        """启动前的设置"""
        print("[HIL-Actor] Connecting to Learner...")
        self.client.connect()
        
        if self.config.require_initial_weights:
            print("[HIL-Actor] Waiting for initial weights...")
            self._wait_for_initial_weights()
        
        print("[HIL-Actor] Ready!")
    
    def _wait_for_initial_weights(self, timeout: float = 60.0) -> None:
        """等待初始权重"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            weights = self.client.recv_weights(block=True, timeout=5.0)
            if weights is not None:
                self.policy.load_weights(weights)
                self._initial_weights_synced = True
                self._weight_sync_count += 1
                print("[HIL-Actor] Initial weights received!")
                return
        raise RuntimeError("Timeout waiting for initial weights")
    
    def step(self) -> Dict[str, Any]:
        """执行单步"""
        # 确保已初始化
        if not self._initial_weights_synced:
            self.setup()
        
        # 增加步计数
        self._step_count += 1
        
        # 尝试同步权重（非阻塞）
        if self._step_count % self.config.weight_sync_freq == 0:
            self._try_sync_weights()
        
        # 确保环境已重置
        if self._current_obs is None:
            self._reset_episode()
        
        # 获取策略动作
        policy_action = self.policy.act(self._current_obs, deterministic=self.config.deterministic)
        policy_action = np.asarray(policy_action)
        
        # 执行（Wrapper 决定实际动作）
        next_obs, env_reward, terminated, truncated, info = self.env.step(policy_action)
        done = terminated or truncated or (self._episode_step >= self.config.max_episode_steps)
        
        # 从 info 读取干预信息
        is_intervention = info.get("is_intervention", False)
        actual_action = info.get("intervene_action", policy_action) if is_intervention else policy_action
        
        # 计算奖励
        reward = self._compute_reward(env_reward, next_obs, info)
        
        # 构建 transition
        transition = {
            "obs": self._extract_state(self._current_obs),
            "action": actual_action,
            "policy_action": policy_action,
            "reward": reward,
            "next_obs": self._extract_state(next_obs),
            "done": done,
            "is_intervention": is_intervention,
            "source": "intervention" if is_intervention else "rollout",
        }
        
        # 发送到 Learner
        self._buffer_and_send(transition)
        
        # 更新统计
        self._episode_reward += reward
        self._episode_step += 1
        self._total_actions += 1
        if is_intervention:
            self._intervention_count += 1
        
        # 构建返回
        step_info = {
            "reward": reward,
            "done": done,
            "is_intervention": is_intervention,
        }
        
        if done:
            step_info.update({
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_step,
                "intervention_rate": self._intervention_count / max(self._total_actions, 1),
            })
            self._episode_count += 1
            self._reset_episode()
        else:
            self._current_obs = next_obs
            self._current_info = info
        
        return step_info
    
    def _reset_episode(self) -> None:
        """重置 episode"""
        self._current_obs, self._current_info = self.env.reset()
        self._episode_step = 0
        self._episode_reward = 0.0
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
    
    def _compute_reward(self, env_reward: float, next_obs: Dict, info: Dict) -> float:
        """计算奖励（可使用分类器）"""
        if not self.config.use_reward_classifier or self.reward_classifier is None:
            return env_reward
        # TODO: 实现奖励分类器
        return env_reward
    
    def _extract_state(self, obs) -> np.ndarray:
        """提取状态向量"""
        if isinstance(obs, np.ndarray):
            return obs
        for key in ["state", "observation", "proprio", "qpos"]:
            if key in obs:
                return np.asarray(obs[key])
        raise ValueError(f"Cannot extract state from: {obs.keys()}")
    
    def _buffer_and_send(self, transition: Dict) -> None:
        """缓冲并发送"""
        self._transition_buffer.append(transition)
        if len(self._transition_buffer) >= self.config.transition_batch_size:
            self.client.send_transitions(self._transition_buffer)
            self._transition_buffer = []
    
    def _try_sync_weights(self) -> bool:
        """尝试同步权重"""
        weights = self.client.recv_weights(block=False)
        if weights is not None:
            self.policy.load_weights(weights)
            self._weight_sync_count += 1
            return True
        return False
    
    def cleanup(self) -> None:
        """清理"""
        if self._transition_buffer:
            self.client.send_transitions(self._transition_buffer)
        self.client.disconnect()
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "episode_count": self._episode_count,
            "total_steps": self._step_count,
            "intervention_rate": self._intervention_count / max(self._total_actions, 1),
            "weight_sync_count": self._weight_sync_count,
        }


# ============ HIL Learner Loop ============

class HILLearnerLoop(BaseLoop):
    """
    HIL Learner 循环（独立进程运行）
    
    职责：
    - 接收 Actor 的 transitions
    - 分流到 rollout / intervention buffer
    - 加权采样训练
    - 发布权重到 Actor
    
    使用示例：
        learner = HILLearnerLoop(
            trainable_adapter=adapter,
            config=config,
            sync_config=sync_config,
        )
        learner.run(num_steps=100000)
    """
    
    def __init__(
        self,
        trainable_adapter: TrainableAdapterProtocol,
        config: HILLearnerConfig,
        sync_config: Optional[ActorLearnerConfig] = None,
        demo_buffer: Optional[BufferProtocol] = None,
        custom_sampler: Optional[SamplerProtocol] = None,
        mode: str = "local",
        learner_server=None,  # 可选：外部注入的 server（用于测试）
    ):
        super().__init__()
        # 手动管理 step_count，防止 BaseLoop.run() 重复递增
        self._manual_step_increment = True
        
        self.adapter = trainable_adapter
        self.config = config
        
        # 创建或使用注入的通信服务器
        if learner_server is not None:
            self.server = learner_server
        else:
            if sync_config is None:
                sync_config = ActorLearnerConfig()
            self.server = create_learner_server(mode, sync_config)
        
        # 初始化 Buffer
        from data.buffers.replay_buffer import ReplayBuffer
        from data.buffers.intervention_buffer import InterventionBuffer
        
        self.rollout_buffer = ReplayBuffer(capacity=config.rollout_buffer_capacity)
        self.intervention_buffer = InterventionBuffer(capacity=config.intervention_buffer_capacity)
        self.demo_buffer = demo_buffer
        
        # 采样器
        if custom_sampler:
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
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 统计
        self._transitions_received = 0
        self._interventions_received = 0
    
    def setup(self) -> None:
        """启动前的设置"""
        print("[HIL-Learner] Starting server...")
        self.server.start()
        
        # 先发布初始权重（让 Actor 可以开始工作）
        print("[HIL-Learner] Publishing initial weights...")
        self._publish_weights()
        
        # 再等待收集初始数据
        print(f"[HIL-Learner] Waiting for {self.config.training_starts} transitions...")
        self._collect_initial_data()
        
        self._training_started = True
        print("[HIL-Learner] Ready!")
    
    def _collect_initial_data(self) -> None:
        """收集初始数据"""
        while self._get_online_size() < self.config.training_starts:
            transitions = self.server.recv_transitions(block=True, timeout=1.0)
            self._route_transitions(transitions)
            
            size = self._get_online_size()
            if size % 100 == 0 and size > 0:
                print(f"[HIL-Learner] Collected {size}/{self.config.training_starts}")
    
    def step(self) -> Dict[str, Any]:
        """执行单步训练"""
        if not self._training_started:
            self.setup()
        
        # 接收新数据（非阻塞）
        transitions = self.server.recv_transitions(block=False)
        self._route_transitions(transitions)
        
        # UTD 更新
        metrics_list = []
        for _ in range(self.config.utd_ratio):
            batch = self._sample_batch()
            if batch is None:
                break
            batch = self._to_device(batch)
            metrics = self.adapter.update(batch)
            metrics_list.append(metrics)
        
        # 合并 metrics
        if metrics_list:
            avg_metrics = {k: np.mean([m.get(k, 0) for m in metrics_list]) 
                          for k in metrics_list[0]}
        else:
            avg_metrics = {}
        
        # 增加步计数
        self._step_count += 1
        
        # 发布权重
        if self._step_count > 0 and self._step_count % self.config.policy_push_frequency == 0:
            self._publish_weights()
        
        # 保存 checkpoint
        if self._step_count > 0 and self._step_count % self.config.checkpoint_freq == 0:
            self._save_checkpoint()
        
        # 添加统计
        avg_metrics.update({
            "rollout_size": len(self.rollout_buffer),
            "intervention_size": len(self.intervention_buffer),
            "intervention_ratio": self._interventions_received / max(self._transitions_received, 1),
        })
        
        return avg_metrics
    
    def _route_transitions(self, transitions: List[Dict]) -> None:
        """分流到对应 buffer"""
        for t in transitions:
            self._transitions_received += 1
            if t.get("is_intervention", False):
                self._interventions_received += 1
                self.intervention_buffer.add(t)
            else:
                self.rollout_buffer.add(t)
    
    def _sample_batch(self) -> Optional[Dict]:
        """采样"""
        buffers = {}
        if self.demo_buffer and len(self.demo_buffer) > 0:
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
    
    def _to_device(self, batch: Dict) -> Dict:
        """移动到设备"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                result[k] = torch.from_numpy(v).float().to(self.device)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def _publish_weights(self) -> None:
        """发布权重"""
        weights = self.adapter.get_weights()
        self.server.publish_weights(weights, {"step": self._step_count})
    
    def _save_checkpoint(self) -> None:
        """保存 checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = f"{self.config.checkpoint_dir}/step_{self._step_count}.pt"
        self.adapter.save(path)
        print(f"[HIL-Learner] Checkpoint: {path}")
    
    def _get_online_size(self) -> int:
        return len(self.rollout_buffer) + len(self.intervention_buffer)
    
    def cleanup(self) -> None:
        """清理"""
        self.server.stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "train_steps": self._step_count,
            "rollout_size": len(self.rollout_buffer),
            "intervention_size": len(self.intervention_buffer),
            "transitions_received": self._transitions_received,
            "intervention_ratio": self._interventions_received / max(self._transitions_received, 1),
        }


# ============ 向后兼容 ============
HILSerlActorLoop = HILActorLoop
HILSerlLearnerLoop = HILLearnerLoop
