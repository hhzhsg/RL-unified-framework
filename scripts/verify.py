#!/usr/bin/env python3
"""
框架验证脚本

验证所有模块是否正常工作
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_types():
    """测试数据类型"""
    import numpy as np
    from data import Observation, RobotState, Action, Transition, Episode, Batch
    
    # Observation
    obs = Observation(images={"cam": np.zeros((224, 224, 3))}, language="test")
    assert obs.language == "test"
    
    # RobotState
    state = RobotState(raw_state=np.random.randn(16).astype(np.float32))
    arr = state.to_array()
    assert arr.shape == (16,)
    
    # Action
    action = Action(data=np.random.randn(7).astype(np.float32), space="joint")
    assert action.space == "joint"
    
    # Transition
    t = Transition(
        obs=obs,
        robot_state=state,
        action=action,
        reward=1.0,
        next_obs=obs,
        next_robot_state=state,
        done=False,
        source="demo",
    )
    assert t.source == "demo"
    
    # Episode
    ep = Episode()
    ep.add(t)
    assert len(ep) == 1
    
    # Batch
    batch = Batch.from_transitions([t, t])
    assert len(batch) == 2
    
    print("✓ Data types OK")


def test_policy():
    """测试策略"""
    import numpy as np
    from policy import MLPPolicy, MLPGaussianPolicy, ResidualPolicy
    from data import Observation, RobotState
    
    # MLPPolicy
    policy = MLPPolicy(state_dim=16, action_dim=7, hidden_dims=[64, 64])
    
    obs = Observation()
    state = RobotState(raw_state=np.random.randn(16).astype(np.float32))
    action = policy.act(obs, state)
    assert action.data.shape == (7,)
    
    # MLPGaussianPolicy
    gauss_policy = MLPGaussianPolicy(state_dim=16, action_dim=7, hidden_dims=[64, 64])
    action = gauss_policy.act(obs, state, deterministic=False)
    assert action.data.shape == (7,)
    
    # ResidualPolicy
    residual = ResidualPolicy(policy, gauss_policy, residual_scale=0.1)
    action = residual.act(obs, state)
    assert action.data.shape == (7,)
    
    print("✓ Policy OK")


def test_network():
    """测试网络"""
    import torch
    from network import MLP, QNetwork, VNetwork
    
    # MLP
    mlp = MLP(input_dim=16, output_dim=7, hidden_dims=[64, 64])
    x = torch.randn(32, 16)
    y = mlp(x)
    assert y.shape == (32, 7)
    
    # QNetwork
    q_net = QNetwork(state_dim=16, action_dim=7, hidden_dims=[64, 64])
    s = torch.randn(32, 16)
    a = torch.randn(32, 7)
    q = q_net(s, a)
    assert q.shape == (32, 1)
    
    # VNetwork
    v_net = VNetwork(state_dim=16, hidden_dims=[64, 64])
    v = v_net(s)
    assert v.shape == (32, 1)
    
    print("✓ Network OK")


def test_algorithm():
    """测试算法"""
    import torch
    import numpy as np
    from algorithm import BC, SAC, TD3BC, create_algorithm
    from policy import MLPPolicy, MLPGaussianPolicy
    from network import QNetwork
    from core import ModelGroup
    from data import Transition, Batch, Observation, RobotState, Action
    from config import AlgorithmConfig
    
    # 创建 dummy batch
    def make_batch():
        transitions = []
        for _ in range(32):
            t = Transition(
                obs=Observation(),
                robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
                action=Action(data=np.random.randn(7).astype(np.float32)),
                reward=0.0,
                next_obs=Observation(),
                next_robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
                done=False,
            )
            transitions.append(t)
        return Batch.from_transitions(transitions)
    
    # BC
    group = ModelGroup()
    group.add("policy", MLPPolicy(16, 7, [64, 64]))
    bc = BC(group, AlgorithmConfig(lr=1e-3))
    metrics = bc.train_step(make_batch())
    assert "loss" in metrics
    
    # SAC
    sac_group = ModelGroup()
    sac_group.add("policy", MLPGaussianPolicy(16, 7, [64, 64]))
    sac_group.add("q1", QNetwork(16, 7, [64, 64]))
    sac_group.add("q2", QNetwork(16, 7, [64, 64]))
    target_q1 = QNetwork(16, 7, [64, 64])
    target_q2 = QNetwork(16, 7, [64, 64])
    target_q1.load_state_dict(sac_group.get("q1").state_dict())
    target_q2.load_state_dict(sac_group.get("q2").state_dict())
    sac_group.add("target_q1", target_q1, frozen=True)
    sac_group.add("target_q2", target_q2, frozen=True)
    sac = SAC(sac_group, AlgorithmConfig(lr=1e-3))
    metrics = sac.train_step(make_batch())
    assert "q_loss" in metrics
    
    # create_algorithm
    algo = create_algorithm("bc", group)
    assert algo.name == "BC"
    
    print("✓ Algorithm OK")


def test_buffer():
    """测试缓冲区"""
    import numpy as np
    from buffer import ReplayBuffer
    from data import Transition, Episode, Observation, RobotState, Action
    
    # ReplayBuffer
    buf = ReplayBuffer(max_size=100)
    
    for i in range(50):
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            action=Action(data=np.random.randn(7).astype(np.float32)),
            reward=float(i),
            next_obs=Observation(),
            next_robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            done=False,
        )
        buf.add_transition(t)
    
    assert len(buf) == 50
    
    sampled = buf.sample_transitions(10)
    assert len(sampled) == 10
    
    print("✓ Buffer OK")


def test_env():
    """测试环境"""
    from env import DummyEnv, create_env
    from config import EnvConfig
    from data import Action
    import numpy as np
    
    # DummyEnv
    config = EnvConfig(state_dim=16, action_dim=4)
    env = DummyEnv(config, deterministic=True)
    
    out = env.reset()
    assert out.robot_state.raw_state.shape == (16,)
    
    action = Action(data=np.random.randn(4).astype(np.float32))
    out = env.step(action)
    assert isinstance(out.reward, float)
    
    # create_env
    env2 = create_env(config)
    assert isinstance(env2, DummyEnv)
    
    print("✓ Env OK")


def test_data_hub():
    """测试数据中心"""
    import numpy as np
    from data import DataHub, Transition, Observation, RobotState, Action
    from buffer import ReplayBuffer
    
    hub = DataHub(rollout_capacity=100)
    
    demo_buf = ReplayBuffer(100)
    for _ in range(20):
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            action=Action(data=np.random.randn(7).astype(np.float32)),
            reward=0.0,
            next_obs=Observation(),
            next_robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            done=False,
            source="demo",
        )
        demo_buf.add_transition(t)
    
    hub.register_buffer("demo", demo_buf)
    
    # 写入 rollout
    for _ in range(10):
        t = Transition(
            obs=Observation(),
            robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            action=Action(data=np.random.randn(7).astype(np.float32)),
            reward=1.0,
            next_obs=Observation(),
            next_robot_state=RobotState(raw_state=np.random.randn(16).astype(np.float32)),
            done=False,
            source="rollout",
        )
        hub.write(t, source="rollout")
    
    stats = hub.statistics()
    assert stats["demo_size"] == 20
    assert stats["rollout_size"] == 10
    
    # 采样
    batch = hub.sample(batch_size=8, strategy="demo_only")
    assert len(batch) == 8
    
    print("✓ DataHub OK")


def test_core():
    """测试核心组件"""
    import torch
    from core import ModelGroup, create_weight_sync
    from policy import MLPPolicy
    
    # ModelGroup
    group = ModelGroup()
    p1 = MLPPolicy(16, 7, [64])
    p2 = MLPPolicy(16, 7, [64])
    
    group.add("policy", p1)
    group.add("target", p2, frozen=True)
    
    assert group.is_frozen("target")
    assert not group.is_frozen("policy")
    assert "policy" in group
    
    # WeightSync
    sync = create_weight_sync("shared_memory")
    sync.push({"test": torch.randn(10)}, version=1)
    result = sync.pull()
    assert result is not None
    assert result[1] == 1
    
    print("✓ Core OK")


def test_transforms():
    """测试数据转换"""
    import numpy as np
    from data.transforms import Compose, ResizeImage, NormalizeImage, NormalizeAction
    
    # 图像转换
    data = {
        "images": {
            "cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
    }
    
    transform = Compose([
        ResizeImage(224),
        NormalizeImage(),
    ])
    
    out = transform(data)
    assert out["images"]["cam"].shape == (224, 224, 3)
    
    # 动作转换
    data2 = {"action": np.array([0.5, 0.5, 0.5])}
    norm = NormalizeAction(low=[0, 0, 0], high=[1, 1, 1])
    out2 = norm(data2)
    assert np.allclose(out2["action"], [0, 0, 0])
    
    print("✓ Transforms OK")


def test_robot():
    """测试机器人适配器"""
    from robot import BinocularAdapter, create_robot_adapter
    
    adapter = BinocularAdapter()
    assert adapter.state_dim == 15
    assert adapter.action_dim == 15
    assert len(adapter.camera_keys) == 4
    
    adapter2 = create_robot_adapter("binocular")
    assert adapter2.name == "binocular_single_arm"
    
    print("✓ Robot adapter OK")


def test_config():
    """测试配置"""
    from config import Config, EnvConfig, AlgorithmConfig
    
    config = Config()
    assert config.seed == 42
    
    config.env = EnvConfig(state_dim=15, action_dim=15)
    config.algorithm = AlgorithmConfig(name="bc", lr=1e-4)
    
    assert config.env.state_dim == 15
    assert config.algorithm.name == "bc"
    
    print("✓ Config OK")


def run_all_tests():
    """运行所有测试"""
    tests = [
        ("Data Types", test_data_types),
        ("Policy", test_policy),
        ("Network", test_network),
        ("Algorithm", test_algorithm),
        ("Buffer", test_buffer),
        ("Env", test_env),
        ("DataHub", test_data_hub),
        ("Core", test_core),
        ("Transforms", test_transforms),
        ("Robot", test_robot),
        ("Config", test_config),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 50)
    print("VLA-RL 框架验证")
    print("=" * 50)
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
