# H1 HIL 项目

H1 人形机器人的 Human-in-the-Loop 训练项目。

## 更新历史

- 2026-01-23: 完成 gRPC 通信测试，验证 HILActorLoop/HILLearnerLoop
- 2026-01-23: 拆分测试组件到框架模块

## 架构

```
projects/h1_hil/
├── test_real_loop.py    # HIL Loop 测试脚本（使用真实 Loop）
├── config.yaml          # 项目配置
├── verify_h1_format.py  # H1 数据格式验证
└── README.md            # 本文件

使用的框架模块：
├── policies/adapters/simple_mlp_adapter.py  # SimpleMLPAdapter, SimpleMLPTrainer
├── env/dummy_env/dummy_env.py               # DummyEnv (支持 intervention_prob)
├── core/runtime/hil_loop.py                 # HILActorLoop, HILLearnerLoop
└── core/synchronization/grpc_impl.py        # gRPC 通信
```

## 测试 HIL Loop

使用真正的 HILActorLoop 和 HILLearnerLoop 进行 gRPC 通信测试：

```bash
# 终端 1 - Learner（先启动）
cd ~/rl_code/zerith/RL-unified-framework/projects/h1_hil
python test_real_loop.py --role learner --mode grpc --port 50051 --device cpu

# 终端 2 - Actor
python test_real_loop.py --role actor --mode grpc --learner-host localhost
```

### 参数说明

**通用参数**：
- `--role`: 角色 (learner 或 actor)
- `--mode`: 通信模式 (grpc 或 local)
- `--max-steps`: 最大步数

**Learner 参数**：
- `--port`: gRPC 监听端口
- `--device`: 训练设备 (cpu 或 cuda)
- `--training-starts`: 开始训练前收集的 transitions 数量
- `--weight-push-freq`: 权重发布频率

**Actor 参数**：
- `--learner-host`: Learner 地址
- `--learner-port`: Learner 端口
- `--intervention-prob`: 干预概率（DummyEnv）

## 正式训练

使用统一训练脚本：

```bash
# 终端 1 - Learner（GPU 服务器）
python scripts/train.py --config projects/h1_hil/config.yaml --role learner --device cuda

# 终端 2 - Actor（机器人端）
python scripts/train.py --config projects/h1_hil/config.yaml --role actor
```

## 验证结果

2026-01-23 测试通过：
- ✅ gRPC 通信正常
- ✅ 权重同步正常
- ✅ 数据分流正确（rollout / intervention）
- ✅ 训练 loss 下降
- ✅ HIL-SERL 加权采样工作
