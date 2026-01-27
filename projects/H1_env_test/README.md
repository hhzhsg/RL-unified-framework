# H1 实机 HIL 测试

## 更新历史
- 2026-01-26: 初始版本

## 目标
完整 Actor-Learner 分布式架构测试：
- gRPC 通信
- 权重同步
- HIL 干预检测（VR 按键接管）
- transition 数据流（rollout vs intervention）

因为没有好的 base policy，临时用 **QposEchoAdapter**（qpos → action），机器人保持静止。

## 使用方式

```bash
# Learner 端（先启动）
python run_h1_test.py --role learner --port 50060

# Actor 端（机器人上）
python run_h1_test.py --role actor --learner-host localhost
```

## 预期输出

**Actor 端**:
```
[QposEcho] qpos → action, range: [-0.123, 0.456]
```

**Learner 端**:
```
[Trainer] Step 1: batch=64, rollout=60, intervention=4
```

VR 按键接管时，`intervention` 数量会增加。

## 测试清单
- [ ] Learner 启动成功
- [ ] Actor 连接成功
- [ ] qpos 读取正常
- [ ] ZCM action 发布正常
- [ ] VR 接管时 is_intervention=True
