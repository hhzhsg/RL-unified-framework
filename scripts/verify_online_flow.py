#!/usr/bin/env python3
"""
VLA-RL Online 数据流验证脚本

验证内容:
1. DummyEnv 正确生成状态和奖励
2. InferenceLoop 正确收集 rollout 数据
3. RolloutBuffer 正确存储和采样
4. SAC 算法正确消费数据并训练
5. WeightSync 正确同步权重

使用方式:
    python scripts/verify_online_flow.py
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_step(msg):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}[Step] {msg}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_ok(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_fail(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.YELLOW}  → {msg}{Colors.END}")


def test_dummy_env():
    """测试 1: DummyEnv 基本功能"""
    print_step("Testing DummyEnv")
    
    from config import EnvConfig
    from env import DummyEnv
    from data import Action
    
    # 创建配置
    config = EnvConfig(
        name="dummy",
        state_dim=16,
        action_dim=4,
        max_episode_steps=100,
    )
    
    # 创建环境
    env = DummyEnv(config, deterministic=True)
    
    # 1. 测试 reset
    env_output = env.reset()
    assert env_output.robot_state.raw_state is not None, "raw_state should not be None"
    assert len(env_output.robot_state.raw_state) == 16, f"state_dim mismatch: {len(env_output.robot_state.raw_state)}"
    print_ok(f"reset() returns state with correct dim: {env_output.robot_state.raw_state.shape}")
    
    # 2. 测试 step
    action = Action(data=np.random.randn(4).astype(np.float32), space="joint")
    env_output = env.step(action)
    assert env_output.robot_state.raw_state is not None
    assert isinstance(env_output.reward, float)
    print_ok(f"step() returns: reward={env_output.reward:.4f}, done={env_output.done}")
    
    # 3. 测试 episode 完整运行
    env.reset()
    total_reward = 0
    for i in range(50):
        action = Action(data=np.random.randn(4).astype(np.float32) * 0.5, space="joint")
        env_output = env.step(action)
        total_reward += env_output.reward
        if env_output.done:
            break
    print_ok(f"Episode ran {i+1} steps, total_reward={total_reward:.4f}")
    
    # 4. 测试确定性模式
    env1 = DummyEnv(config, deterministic=True)
    env2 = DummyEnv(config, deterministic=True)
    out1 = env1.reset()
    out2 = env2.reset()
    assert np.allclose(out1.robot_state.raw_state, out2.robot_state.raw_state), "Deterministic mode failed"
    print_ok("Deterministic mode works correctly")
    
    return True


def test_inference_loop_collection():
    """测试 2: InferenceLoop 数据收集"""
    print_step("Testing InferenceLoop Data Collection")
    
    from config import EnvConfig, InferenceConfig
    from env import DummyEnv
    from model import MLPGaussianPolicy
    from buffer import DataHub
    from core import InferenceLoop
    from data import Observation, RobotState, Transition, Action
    
    # 创建组件
    env_config = EnvConfig(name="dummy", state_dim=16, action_dim=4, max_episode_steps=50)
    env = DummyEnv(env_config)
    
    policy = MLPGaussianPolicy(
        state_dim=16,
        action_dim=4,
        hidden_dims=[64, 64],
    )
    
    data_hub = DataHub(rollout_capacity=10000)
    
    inference_config = InferenceConfig(device="cpu", deterministic=False)
    
    inference_loop = InferenceLoop(
        policy=policy,
        env=env,
        config=inference_config,
        data_hub=data_hub,
    )
    
    # 收集数据
    initial_size = len(data_hub.rollout_buffer)
    print_info(f"Initial buffer size: {initial_size}")
    
    # 手动测试写入
    test_t = Transition(
        obs=Observation(),
        robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
        action=Action(data=np.random.randn(4).astype(np.float32), space="joint"),
        reward=1.0,
        next_obs=Observation(),
        next_robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
        done=False,
        source="rollout",
    )
    data_hub.write(test_t, source="rollout")
    print_info(f"After manual write: {len(data_hub.rollout_buffer)}")
    
    collected = inference_loop.collect_rollout(num_steps=500, source="rollout")
    
    final_size = len(data_hub.rollout_buffer)
    print_ok(f"Collected {collected} steps, buffer size: {initial_size} -> {final_size}")
    
    # 验证数据格式
    transitions = data_hub.rollout_buffer.sample_transitions(10)
    assert len(transitions) == 10, "Sampling failed"
    
    t = transitions[0]
    assert t.source == "rollout", f"Wrong source: {t.source}"
    assert t.robot_state.raw_state is not None, "raw_state is None"
    assert t.action is not None, "action is None"
    print_ok(f"Transition format correct: state={t.robot_state.raw_state.shape}, reward={t.reward:.4f}")
    
    return True


def test_rollout_buffer_sampling():
    """测试 3: RolloutBuffer 采样功能"""
    print_step("Testing RolloutBuffer Sampling")
    
    from buffer import RolloutBuffer
    from data import Transition, RobotState, Observation, Action
    
    buffer = RolloutBuffer(max_size=1000)
    
    # 添加数据
    for i in range(500):
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
            action=Action(data=np.random.randn(4).astype(np.float32), space="joint"),
            reward=float(i) * 0.01,
            next_obs=Observation(),
            next_robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
            done=(i % 50 == 49),
            source="rollout",
        )
        buffer.add_transition(t)
    
    print_ok(f"Added 500 transitions, buffer size: {len(buffer)}")
    
    # 测试采样
    samples = buffer.sample_transitions(64)
    assert len(samples) == 64, "Sampling size mismatch"
    print_ok(f"Sampled 64 transitions successfully")
    
    # 测试统计
    stats = buffer.get_statistics()
    print_info(f"Buffer stats: {stats}")
    
    # 测试 FIFO
    for i in range(600):  # 超过容量
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.zeros(16, dtype=np.float32)),
            action=Action(data=np.zeros(4, dtype=np.float32), space="joint"),
            reward=100.0,  # 新数据用特殊奖励
            next_obs=Observation(),
            next_robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.zeros(16, dtype=np.float32)),
            done=False,
            source="rollout",
        )
        buffer.add_transition(t)
    
    assert len(buffer) == 1000, f"FIFO failed: {len(buffer)}"
    print_ok(f"FIFO works: buffer capped at {len(buffer)}")
    
    return True


def test_datahub_sampling():
    """测试 4: DataHub 采样策略"""
    print_step("Testing DataHub Sampling Strategies")
    
    from buffer import DataHub
    from data import Transition, RobotState, Observation, Action
    
    data_hub = DataHub(rollout_capacity=1000)
    
    # 添加 rollout 数据
    for i in range(200):
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
            action=Action(data=np.random.randn(4).astype(np.float32), space="joint"),
            reward=np.random.randn(),
            next_obs=Observation(),
            next_robot_state=RobotState(joint_pos=np.zeros(7), raw_state=np.random.randn(16).astype(np.float32)),
            done=False,
            source="rollout",
        )
        data_hub.write(t, source="rollout")
    
    print_ok(f"Added 200 rollout transitions")
    
    # 测试 rollout_only 采样
    batch = data_hub.sample(batch_size=64, strategy="rollout_only")
    assert batch.robot_state.shape[0] == 64, f"Batch size mismatch: {batch.robot_state.shape}"
    print_ok(f"rollout_only sampling: batch shape = {batch.robot_state.shape}")
    
    # 验证 batch 可以转换到设备
    batch_cuda = batch.to("cpu")  # 测试设备转换
    assert isinstance(batch_cuda.robot_state, torch.Tensor)
    print_ok(f"Batch.to(device) works: robot_state is Tensor")
    
    return True


def test_sac_training_step():
    """测试 5: SAC 算法训练步"""
    print_step("Testing SAC Training Step")
    
    from config import AlgorithmConfig
    from model import ModelGroup, MLPGaussianPolicy
    from model.q_network import QNetwork
    from algorithm import SAC
    from data import Batch
    import copy
    
    state_dim = 16
    action_dim = 4
    batch_size = 32
    
    # 创建模型组
    model_group = ModelGroup()
    
    policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims=[64, 64])
    q1 = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    q2 = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    
    model_group.add("policy", policy)
    model_group.add("q1", q1)
    model_group.add("q2", q2)
    model_group.add("target_q1", copy.deepcopy(q1), frozen=True)
    model_group.add("target_q2", copy.deepcopy(q2), frozen=True)
    
    print_ok(f"ModelGroup created with: {model_group.model_names}")
    
    # 创建算法
    config = AlgorithmConfig(name="sac", lr=3e-4, batch_size=batch_size)
    sac = SAC(model_group, config)
    print_ok("SAC algorithm created")
    
    # 创建假数据
    batch = Batch(
        obs={},
        robot_state=torch.randn(batch_size, state_dim),
        action=torch.randn(batch_size, action_dim),
        reward=torch.randn(batch_size),
        next_obs={},
        next_robot_state=torch.randn(batch_size, state_dim),
        done=torch.zeros(batch_size),
        source=["rollout"] * batch_size,
    )
    
    # 训练一步
    metrics = sac.train_step(batch)
    print_ok(f"train_step() returned: {metrics}")
    
    # 验证指标
    assert "q_loss" in metrics
    assert "policy_loss" in metrics
    assert "alpha" in metrics
    print_ok("All expected metrics present")
    
    # 多步训练验证收敛趋势
    q_losses = []
    for i in range(10):
        batch = Batch(
            obs={},
            robot_state=torch.randn(batch_size, state_dim),
            action=torch.randn(batch_size, action_dim),
            reward=torch.randn(batch_size),
            next_obs={},
            next_robot_state=torch.randn(batch_size, state_dim),
            done=torch.zeros(batch_size),
            source=["rollout"] * batch_size,
        )
        metrics = sac.train_step(batch)
        q_losses.append(metrics["q_loss"])
    
    print_ok(f"10 training steps completed, q_loss range: [{min(q_losses):.4f}, {max(q_losses):.4f}]")
    
    return True


def test_weight_sync():
    """测试 6: 权重同步"""
    print_step("Testing WeightSync")
    
    from core import create_weight_sync
    from model import MLPGaussianPolicy
    import copy
    
    # 创建同步器
    weight_sync = create_weight_sync("queue")
    print_ok("QueueWeightSync created")
    
    # 创建两个策略
    policy_train = MLPGaussianPolicy(16, 4, hidden_dims=[32, 32])
    policy_infer = MLPGaussianPolicy(16, 4, hidden_dims=[32, 32])
    
    # 初始权重不同
    for p1, p2 in zip(policy_train.parameters(), policy_infer.parameters()):
        p2.data = torch.randn_like(p2.data)
    
    # 验证权重不同
    p1_sum = sum(p.sum().item() for p in policy_train.parameters())
    p2_sum = sum(p.sum().item() for p in policy_infer.parameters())
    assert abs(p1_sum - p2_sum) > 0.1, "Initial weights should be different"
    print_ok(f"Initial weight sums: train={p1_sum:.4f}, infer={p2_sum:.4f}")
    
    # 推送权重
    weight_sync.push(policy_train.state_dict(), version=1)
    print_ok("Pushed weights v1")
    
    # 拉取权重
    result = weight_sync.pull()
    assert result is not None, "Pull should return weights"
    state_dict, version = result
    assert version == 1
    print_ok(f"Pulled weights v{version}")
    
    # 加载权重
    policy_infer.load_state_dict(state_dict)
    
    # 验证权重相同
    p2_sum_new = sum(p.sum().item() for p in policy_infer.parameters())
    assert abs(p1_sum - p2_sum_new) < 0.001, "Weights should match after sync"
    print_ok(f"After sync: train={p1_sum:.4f}, infer={p2_sum_new:.4f}")
    
    # 测试空队列
    result = weight_sync.pull()
    assert result is None, "Should return None when queue is empty"
    print_ok("Empty queue returns None correctly")
    
    return True


def test_full_online_flow():
    """测试 7: 完整 Online 流程"""
    print_step("Testing Full Online Flow (Mini)")
    
    from config import EnvConfig, InferenceConfig, AlgorithmConfig
    from env import DummyEnv
    from model import ModelGroup, MLPGaussianPolicy
    from model.q_network import QNetwork
    from buffer import DataHub
    from core import InferenceLoop, create_weight_sync
    from algorithm import SAC
    import copy
    
    # 配置
    state_dim = 16
    action_dim = 4
    
    # 1. 创建所有组件
    env_config = EnvConfig(name="dummy", state_dim=state_dim, action_dim=action_dim, max_episode_steps=50)
    env = DummyEnv(env_config)
    
    model_group = ModelGroup()
    policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims=[64, 64])
    q1 = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    q2 = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    model_group.add("policy", policy)
    model_group.add("q1", q1)
    model_group.add("q2", q2)
    model_group.add("target_q1", copy.deepcopy(q1), frozen=True)
    model_group.add("target_q2", copy.deepcopy(q2), frozen=True)
    
    data_hub = DataHub(rollout_capacity=10000)
    weight_sync = create_weight_sync("shared_memory")
    
    inference_config = InferenceConfig(device="cpu", deterministic=False)
    inference_loop = InferenceLoop(
        policy=policy,
        env=env,
        config=inference_config,
        data_hub=data_hub,
        weight_sync=weight_sync,
    )
    
    algo_config = AlgorithmConfig(name="sac", lr=3e-4, batch_size=64)
    sac = SAC(model_group, algo_config)
    
    print_ok("All components created")
    
    # 2. 收集初始数据
    collected = inference_loop.collect_rollout(num_steps=200, source="rollout")
    print_ok(f"Collected {collected} initial steps")
    
    # 3. 训练循环
    train_steps = 0
    for i in range(5):
        # 收集数据
        inference_loop.collect_rollout(num_steps=50, source="rollout")
        
        # 训练
        for j in range(10):
            batch = data_hub.sample(batch_size=64, strategy="rollout_only")
            batch = batch.to("cpu")
            metrics = sac.train_step(batch)
            train_steps += 1
        
        # 同步权重
        weight_sync.push({"policy": policy.state_dict()}, version=i+1)
        
        # 推理端更新权重
        result = weight_sync.pull()
        if result:
            state_dict, version = result
            inference_loop.policy.load_state_dict(state_dict["policy"])
    
    print_ok(f"Completed {train_steps} training steps with weight sync")
    
    # 4. 评估
    eval_results = inference_loop.evaluate(num_episodes=3)
    print_ok(f"Evaluation: success_rate={eval_results['success_rate']:.2f}, "
             f"avg_reward={eval_results['avg_reward']:.4f}")
    
    # 5. 最终检查
    final_buffer_size = len(data_hub.rollout_buffer)
    print_ok(f"Final buffer size: {final_buffer_size}")
    
    return True


def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}VLA-RL Online Data Flow Verification{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    tests = [
        ("DummyEnv", test_dummy_env),
        ("InferenceLoop Collection", test_inference_loop_collection),
        ("RolloutBuffer Sampling", test_rollout_buffer_sampling),
        ("DataHub Sampling", test_datahub_sampling),
        ("SAC Training Step", test_sac_training_step),
        ("WeightSync", test_weight_sync),
        ("Full Online Flow", test_full_online_flow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            traceback.print_exc()
    
    # 总结
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    passed = 0
    for name, success, error in results:
        if success:
            print(f"{Colors.GREEN}✓ {name}{Colors.END}")
            passed += 1
        else:
            print(f"{Colors.RED}✗ {name}: {error}{Colors.END}")
    
    print(f"\n{Colors.BLUE}Result: {passed}/{len(tests)} tests passed{Colors.END}")
    
    if passed == len(tests):
        print(f"\n{Colors.GREEN}🎉 All tests passed! Online data flow is working correctly.{Colors.END}")
        print(f"\n{Colors.YELLOW}Next step: Run full training with:{Colors.END}")
        print(f"  python scripts/train_online.py --config config/train_config.yaml --name online_sac_dummy")
        return 0
    else:
        print(f"\n{Colors.RED}❌ Some tests failed. Please check the errors above.{Colors.END}")
        return 1


if __name__ == "__main__":
    exit(main())
