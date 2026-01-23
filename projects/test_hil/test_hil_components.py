#!/usr/bin/env python3
"""
HIL 组件测试脚本

测试内容：
1. Actor-Learner 通信层
2. HIL 加权采样器
3. 适配器机制
4. 本地模式完整流程

运行方式：
    cd vla-rl
    python projects/test_hil/test_hil_components.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn


# ============ Dummy 组件 ============

class DummyEnv:
    """简单测试环境"""
    def __init__(self, state_dim=8, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._step = 0
    
    def reset(self):
        self._step = 0
        obs = {"state": np.zeros(self.state_dim, dtype=np.float32)}
        return obs, {}
    
    def step(self, action):
        self._step += 1
        next_obs = {"state": np.random.randn(self.state_dim).astype(np.float32) * 0.1}
        reward = -np.linalg.norm(action)
        done = self._step >= 100
        return next_obs, reward, done, False, {}


class DummyPolicy(nn.Module):
    """简单测试策略"""
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self._device = torch.device("cpu")
    
    def forward(self, obs):
        if isinstance(obs, dict):
            x = obs["state"]
        else:
            x = obs
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.net(x)
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.forward(obs)
            return action.numpy()
    
    @property
    def device(self):
        return self._device
    
    def reset(self):
        pass


class DummyAlgorithm:
    """简单测试算法"""
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self._train_step = 0
    
    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        pred = self.policy({"state": obs})
        loss = torch.nn.functional.mse_loss(pred, action)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._train_step += 1
        return {"loss": loss.item()}
    
    def get_policy(self):
        return self.policy
    
    def save(self, path):
        torch.save({"policy": self.policy.state_dict()}, path)
    
    def load(self, path):
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])


# ============ 测试函数 ============

def test_actor_learner_communication():
    """测试 1: Actor-Learner 通信层"""
    print("=" * 60)
    print("Test 1: Actor-Learner Communication")
    print("=" * 60)
    
    from core.synchronization.actor_learner import (
        create_learner_server,
        create_actor_client,
    )
    
    # 创建 server 和 client
    server = create_learner_server("local")
    client = create_actor_client("local", server=server)
    
    server.start()
    client.connect()
    
    # 测试 transition 发送
    transitions = [
        {"obs": np.zeros(8), "action": np.ones(4), "reward": 1.0, "done": False},
        {"obs": np.ones(8), "action": np.zeros(4), "reward": -1.0, "done": True},
    ]
    client.send_transitions(transitions)
    
    # 接收 transitions
    received = server.recv_transitions(block=True, timeout=1.0)
    assert len(received) == 2, f"Expected 2 transitions, got {len(received)}"
    print(f"  ✓ Sent {len(transitions)} transitions, received {len(received)}")
    
    # 测试权重发布
    dummy_weights = {"layer.weight": torch.randn(10, 10)}
    server.publish_weights(dummy_weights, metadata={"step": 100})
    
    weights = client.recv_weights(block=True, timeout=1.0)
    assert weights is not None
    assert "layer.weight" in weights
    print("  ✓ Weight synchronization works")
    
    client.disconnect()
    server.stop()
    print("✓ Test 1 PASSED\n")


def test_hilserl_sampler():
    """测试 2: HIL 加权采样器"""
    print("=" * 60)
    print("Test 2: HIL Weighted Sampler")
    print("=" * 60)
    
    from data.buffers.replay_buffer import ReplayBuffer
    from data.buffers.intervention_buffer import InterventionBuffer
    from data.samplers.hilserl_sampler import HILSERLSampler
    
    rollout_buf = ReplayBuffer(capacity=1000)
    intervention_buf = InterventionBuffer(capacity=1000)
    
    # 填充数据
    for i in range(100):
        rollout_buf.add({
            "obs": np.random.randn(8).astype(np.float32),
            "action": np.random.randn(4).astype(np.float32),
            "reward": float(i),
            "next_obs": np.random.randn(8).astype(np.float32),
            "done": False,
        })
    
    for i in range(50):
        intervention_buf.add({
            "obs": np.random.randn(8).astype(np.float32),
            "action": np.random.randn(4).astype(np.float32),
            "reward": float(i + 100),
            "next_obs": np.random.randn(8).astype(np.float32),
            "done": False,
        })
    
    # 创建采样器（intervention 2x 权重）
    sampler = HILSERLSampler(weights={
        "rollout": 1.0,
        "intervention": 2.0,
    })
    
    buffers = {"rollout": rollout_buf, "intervention": intervention_buf}
    batch = sampler.sample(buffers, batch_size=64)
    
    assert len(batch["obs"]) == 64
    print(f"  ✓ Sampled batch size: {len(batch['obs'])}")
    print("✓ Test 2 PASSED\n")


def test_adapter_mechanism():
    """测试 3: 适配器机制"""
    print("=" * 60)
    print("Test 3: Adapter Mechanism")
    print("=" * 60)
    
    from core.interfaces.adapters import StandardPolicyAdapter, AlgorithmAdapter
    
    # 创建 dummy 组件
    policy = DummyPolicy()
    algorithm = DummyAlgorithm(policy)
    
    # 测试 StandardPolicyAdapter
    policy_adapter = StandardPolicyAdapter(policy)
    
    obs = {"state": np.random.randn(8).astype(np.float32)}
    action = policy_adapter.act(obs)
    assert action.shape == (4,)
    print(f"  ✓ PolicyAdapter.act() works, action shape: {action.shape}")
    
    weights = policy_adapter.get_weights()
    assert len(weights) > 0
    print(f"  ✓ PolicyAdapter.get_weights() works, {len(weights)} params")
    
    # 测试 AlgorithmAdapter
    algo_adapter = AlgorithmAdapter(algorithm)
    
    batch = {
        "obs": np.random.randn(32, 8).astype(np.float32),
        "action": np.random.randn(32, 4).astype(np.float32),
    }
    metrics = algo_adapter.update(batch)
    assert "loss" in metrics
    print(f"  ✓ AlgorithmAdapter.update() works, loss: {metrics['loss']:.4f}")
    
    print("✓ Test 3 PASSED\n")


def test_local_hil_flow():
    """测试 4: 本地 HIL 完整流程"""
    print("=" * 60)
    print("Test 4: Local HIL Flow (Simplified)")
    print("=" * 60)
    
    from core.synchronization.actor_learner import (
        create_learner_server,
        create_actor_client,
    )
    from core.runtime.hil_actor_loop import HILActorLoop, HILActorConfig
    from core.runtime.hil_learner_loop import HILLearnerLoop, HILLearnerConfig
    from core.interfaces.adapters import StandardPolicyAdapter, AlgorithmAdapter
    
    # 创建组件
    env = DummyEnv()
    policy = DummyPolicy()
    algorithm = DummyAlgorithm(policy)
    
    policy_adapter = StandardPolicyAdapter(policy)
    algo_adapter = AlgorithmAdapter(algorithm)
    
    # 创建通信
    server = create_learner_server("local")
    client = create_actor_client("local", server=server)
    
    server.start()
    client.connect()
    
    # 创建 Actor
    actor_config = HILActorConfig(
        weight_sync_freq=10,
        transition_batch_size=1,
        require_initial_weights=True,
    )
    actor = HILActorLoop(
        policy_adapter=policy_adapter,
        env=env,
        actor_client=client,
        config=actor_config,
    )
    
    # 创建 Learner
    learner_config = HILLearnerConfig(
        batch_size=16,
        training_starts=50,
        policy_push_frequency=20,
        checkpoint_freq=1000,
        device="cpu",
    )
    learner = HILLearnerLoop(
        trainable_adapter=algo_adapter,
        learner_server=server,
        config=learner_config,
    )
    
    # 发布初始权重
    learner.publish_initial_weights()
    actor.wait_for_initial_weights(timeout=5.0)
    
    # 收集初始数据
    print("  Collecting initial data...")
    for i in range(learner_config.training_starts):
        actor.step()
    
    learner.start_training()
    print(f"  ✓ Collected {learner._get_total_online_size()} transitions")
    
    # 运行几步
    print("  Running training steps...")
    for step in range(20):
        actor.step()
        metrics = learner.step()
    
    stats = actor.get_statistics()
    print(f"  ✓ Actor: {stats['total_steps']} steps, {stats['episode_count']} episodes")
    
    stats = learner.get_statistics()
    print(f"  ✓ Learner: {stats['train_steps']} steps, {stats['transitions_received']} transitions")
    
    # 清理
    server.stop()
    client.disconnect()
    
    print("✓ Test 4 PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("HIL Components Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Actor-Learner Communication", test_actor_learner_communication),
        ("HIL Weighted Sampler", test_hilserl_sampler),
        ("Adapter Mechanism", test_adapter_mechanism),
        ("Local HIL Flow", test_local_hil_flow),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
