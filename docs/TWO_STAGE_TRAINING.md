# Two-Stage Training: Offline → Online

## 概述

两阶段训练是 VLA/机器人强化学习的标准流程：

1. **Stage 1 (Offline)**: 用 Demo 数据学习基础策略
2. **Stage 2 (Online)**: 在环境中 finetune，超越 Demo 性能

---

## 为什么需要两阶段？

### VLA 不能从头 Online 训练

| 问题 | 原因 |
|------|------|
| **状态空间巨大** | 图像 224×224×3 ≈ 150K 维 |
| **随机探索无效** | 随机挥舞机械臂永远学不会抓取 |
| **样本效率极低** | 需要百万级交互才能收敛 |

### 两阶段方案

```
Demo 数据 → Offline Pretrain → Policy 学会模仿
                ↓
           Online Finetune → Policy 在环境反馈中优化，超越 Demo
```

---

## 数据流架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Offline TD3+BC                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  数据源: Demo (HDF5 或 fake)                                  │  │
│  │  Buffer: Demo 固定，Rollout 为空                              │  │
│  │  采样: demo_only                                              │  │
│  │  算法: TD3+BC (行为克隆 + Q 学习)                              │  │
│  │  输出: 训练好的 Policy + Q 网络                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓ 继承权重                              │
│  Stage 2: Online SAC                                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  数据源: Demo (固定) + Rollout (逐渐增加)                      │  │
│  │  Buffer: Demo 50 条，Rollout 0 → 2000 条                       │  │
│  │  采样: mixed (demo_ratio=0.25)                                │  │
│  │    - Rollout 为空时 → 自动退化为 demo_only                     │  │
│  │    - Rollout 增多后 → 25% Demo + 75% Rollout                  │  │
│  │  算法: SAC (最大化熵的 Actor-Critic)                           │  │
│  │  输出: 适应环境的 Policy                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 使用方法

### 快速开始

```bash
python scripts/train_two_stage_example.py
```

### 预期输出

```
============================================================
VLA-RL Two-Stage Training: Offline → Online
============================================================
[Setup] Environment: DummyEnv (state=16, action=4)
[Setup] Models: ['policy', 'q1', 'q2', 'target_q1', 'target_q2']
[Demo] Generated 50 demo transitions
[Setup] Demo buffer size: 50
[Setup] Rollout buffer size: 0 (should be 0)

============================================================
Stage 1: Offline TD3+BC (Demo Only)
============================================================
[Stage1] Step 100/500 | q_loss: 39.63, policy_loss: 0.00
[Stage1] Step 500/500 | q_loss: 0.34, policy_loss: 0.00
[Stage1] Offline training completed!

============================================================
Stage 2: Online SAC (Demo + Rollout Mixed)
============================================================
[Stage2] Step 100/2000 | Rollout: 100, Demo: 50 | q_loss: 659.47
[Stage2] Step 2000/2000 | Rollout: 2000, Demo: 50 | q_loss: 1.63
[Stage2] Online training completed!
[Stage2] Final avg reward (last 10 eps): 10.00

============================================================
Two-Stage Training Completed!
============================================================
```

---

## 关键设计点

### 1. MixedStrategy 自动退化

**实现**: [buffer/sample_strategy.py](../buffer/sample_strategy.py)

```python
class MixedStrategy:
    def sample(self, buffers, batch_size):
        # 关键：Rollout 为空时自动退化为纯 Demo
        if not rollout_available:
            if demo_available:
                return demo_buffer.sample_transitions(batch_size)
        
        # 正常情况按比例采样
        demo_size = int(batch_size * self.demo_ratio)  # 25%
        rollout_size = batch_size - demo_size          # 75%
```

**验证**:
```
Step 1: Rollout=0 → 采样 64 条全部来自 Demo
Step 100: Rollout=100 → 采样 16 Demo + 48 Rollout
Step 2000: Rollout=2000 → 采样比例稳定 25%/75%
```

### 2. Policy 类型兼容

**问题**: TD3+BC 和 SAC 需要不同 policy 类型
- TD3+BC: 确定性 policy 或 Gaussian policy 的均值
- SAC: Gaussian policy 的采样

**解决**: [algorithm/td3_bc.py](../algorithm/td3_bc.py)

```python
def _get_action(self, policy, state):
    output = policy.forward({}, state)
    if isinstance(output, tuple):
        # MLPGaussianPolicy 返回 (mean, log_std)
        mean, _ = output
        return torch.tanh(mean)
    else:
        # MLPPolicy 直接返回 action
        return output
```

### 3. 权重继承

**Stage 1 → Stage 2**: 直接复用同一个 `model_group` 对象

```python
# Stage 1 训练
model_group = run_stage1_offline(model_group, ...)

# Stage 2 直接使用，权重已在 model_group 中
run_stage2_online(model_group, ...)
```

---

## 自定义训练

### 使用真实 Demo 数据

```python
# 替换 generate_fake_demo() 为：
data_hub = DataHub(
    demo_paths=[
        "/path/to/demo1.hdf5",
        "/path/to/demo2.hdf5",
    ],
    camera_keys=["cam_high", "cam_left_wrist"],
    load_images=False,  # 先不加载图像
    rollout_capacity=50000,
)
```

### 调整训练参数

```python
# Stage 1: Offline TD3+BC
run_stage1_offline(
    model_group, data_hub, device,
    max_steps=5000,    # 增加训练步数
    logger=logger
)

# Stage 2: Online SAC
run_stage2_online(
    model_group, data_hub, env, device,
    max_steps=50000,   # 增加 online 训练
    logger=logger
)
```

### 修改采样比例

```python
# 在 run_stage2_online() 中修改
batch = data_hub.sample(
    batch_size=256,
    strategy="mixed",
    demo_ratio=0.1,  # 10% Demo + 90% Rollout
)
```

---

## 常见问题

### Q1: Stage 2 初始 q_loss 很大（>600）正常吗？

**正常**。因为 Stage 1 的 Q 网络只见过 Demo 数据，Stage 2 开始时 Rollout 数据分布不同，Q 值估计会偏离。随着训练会收敛。

### Q2: 为什么需要 demo_ratio=0.25？

**数据分布稳定**。纯 Rollout 训练容易遗忘 Demo 中的好行为，混合 Demo 可以：
- 保持策略不崩溃
- Demo 提供"安全基线"
- 符合 RLPD 等工作的经验

### Q3: 如何判断训练成功？

**Stage 1**: `q_loss` 降到 <1.0，`policy_loss` 接近 0

**Stage 2**: 
- `q_loss` 稳定在 <10
- `avg_reward` 稳定或上升
- Rollout 数据持续增加

### Q4: 可以跳过 Stage 1 直接 Online 训练吗？

**理论可行，但极不推荐**：
- 探索效率极低
- 收敛时间长 10-100 倍
- 容易陷入局部最优

---

## 相关工作

| 工作 | 方法 | 关键点 |
|------|------|--------|
| **RLPD** (2023) | Demo + Online SAC | 证明 demo 引导的重要性 |
| **Cal-QL** (2024) | Offline CQL → Online | 保守 Q 学习避免 OOD |
| **SERL** (2024) | 真机 Online RL | 简单任务可从头训，但需 reward shaping |
| **HIL-SERL** (2024) | Human-in-the-Loop | 人工干预提升探索效率 |

---

## 下一步

1. **接入真实环境**: 替换 DummyEnv 为真实机器人环境
2. **添加图像输入**: 修改 policy 支持 vision encoder
3. **Human-in-the-Loop**: 集成 InterventionBuffer
4. **多任务训练**: 使用语言条件的 policy

---

## 参考

- [ARCHITECTURE.md](../ARCHITECTURE.md) - 框架整体设计
- [README.md](../README.md) - 快速开始
- [algorithm/td3_bc.py](../algorithm/td3_bc.py) - TD3+BC 实现
- [algorithm/sac.py](../algorithm/sac.py) - SAC 实现
- [buffer/sample_strategy.py](../buffer/sample_strategy.py) - 采样策略
