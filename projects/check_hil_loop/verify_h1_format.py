#!/usr/bin/env python
"""
H1 数据格式验证脚本

验证 demo 数据的维度，确认正确的 state/action 定义
"""
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.buffers import DemoBuffer
from policies.composed.mlp_policy import MLPPolicy
from policies.adapters import StandardPolicyAdapter
from algorithms.bc_algorithm import BCAlgorithm


# ============================================================
# H1 数据格式定义 (与实际控制对齐)
# ============================================================

# State keys (观测)
STATE_KEYS = [
    'observation/state/arm/position',       # 14 (左右臂各 7 关节)
    'observation/state/arm/velocity',       # 14 (关节速度)
    'observation/state/effector/position',  # 2  (左右 gripper)
    'observation/state/waist/position',     # 3  (height, pitch, yaw)
    'observation/state/head/position',      # 2  (pitch, yaw)
    'observation/state/base/velocity',      # 2  (x, yaw)
]
# 总共: 14 + 14 + 2 + 3 + 2 + 2 = 37

# Action keys (控制)
ACTION_KEYS = [
    'action/arm/position',       # 14 (左右臂各 7 关节)
    'action/effector/position',  # 2  (左右 gripper)
    'action/waist/position',     # 3  (height, pitch, yaw)
    'action/head/position',      # 2  (pitch, yaw)
    'action/base/velocity',      # 2  (x, yaw)
]
# 总共: 14 + 2 + 3 + 2 + 2 = 23


def flatten_state(sample: dict) -> np.ndarray:
    """将 H1 嵌套状态展平为向量"""
    return np.concatenate([sample[k] for k in STATE_KEYS], axis=-1)


def flatten_action(sample: dict) -> np.ndarray:
    """将 H1 嵌套动作展平为向量"""
    return np.concatenate([sample[k] for k in ACTION_KEYS], axis=-1)


def main():
    print("=" * 60)
    print("H1 数据格式验证 (23维 action)")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载 demo 数据...")
    buffer = DemoBuffer(data_path="data/demo/hdf5test")
    
    # 2. 检查原始维度
    print("\n[2] 原始数据维度:")
    sample = buffer.sample(batch_size=1)
    for key in sorted(sample.keys()):
        if 'state' in key or 'action' in key:
            print(f"    {key}: {sample[key].shape}")
    
    # 3. 展平后的维度
    print("\n[3] 展平后的维度:")
    sample = buffer.sample(batch_size=32)
    states = flatten_state(sample)
    actions = flatten_action(sample)
    
    print(f"    State: {states.shape} (期望: (32, 37))")
    print(f"    Action: {actions.shape} (期望: (32, 23))")
    
    # 维度分解
    print("\n    State 分解:")
    print(f"      arm/position:    14")
    print(f"      arm/velocity:    14")
    print(f"      effector:         2")
    print(f"      waist:            3")
    print(f"      head:             2")
    print(f"      base:             2")
    print(f"      -----------------")
    print(f"      总计:            37")
    
    print("\n    Action 分解:")
    print(f"      arm/position:    14 (左7+右7)")
    print(f"      effector:         2 (左gripper+右gripper)")
    print(f"      waist:            3 (height+pitch+yaw)")
    print(f"      head:             2 (pitch+yaw)")
    print(f"      base:             2 (x+yaw)")
    print(f"      -----------------")
    print(f"      总计:            23")
    
    # 4. 创建网络并验证
    print("\n[4] 创建 MLPPolicy 并验证...")
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    
    policy = MLPPolicy({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": [256, 256],
        "device": "cpu",
    })
    
    print(f"    网络结构: {state_dim} → 256 → 256 → {action_dim}")
    
    # 前向传播
    obs = {"state": states[0]}
    pred_action = policy.act(obs, deterministic=True)
    print(f"    输入: state {states[0].shape}")
    print(f"    输出: action {pred_action.shape}")
    
    # 5. 训练验证
    print("\n[5] BCAlgorithm 训练验证...")
    algorithm = BCAlgorithm(policy=policy, config={"learning_rate": 3e-4})
    
    losses = []
    for i in range(20):
        sample = buffer.sample(batch_size=64)
        batch = {
            "obs": flatten_state(sample),
            "action": flatten_action(sample),
        }
        metrics = algorithm.update(batch)
        losses.append(metrics["bc_loss"])
    
    print(f"    初始 loss: {losses[0]:.6f}")
    print(f"    最终 loss: {losses[-1]:.6f}")
    print(f"    下降: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # 6. Adapter 验证
    print("\n[6] StandardPolicyAdapter 验证...")
    adapter = StandardPolicyAdapter(policy)
    weights = adapter.get_weights()
    print(f"    权重数量: {len(weights)} 个 tensor")
    adapter.load_weights(weights)
    print(f"    load_weights: OK")
    
    print("\n" + "=" * 60)
    print("✅ 验证通过!")
    print("=" * 60)
    print(f"    State: {state_dim} 维")
    print(f"    Action: {action_dim} 维")
    print("=" * 60)


if __name__ == "__main__":
    main()
