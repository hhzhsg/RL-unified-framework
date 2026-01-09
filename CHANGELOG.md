# 更新日志

本文档记录 VLA-RL Framework 的所有重要变更。

版本格式遵循 [语义化版本](https://semver.org/lang/zh-CN/)，
变更日志格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [0.2.0] - 2026-01-09

### 新增
- ✨ 多阶段训练支持 (`Stage`, `TrainingLoop` 自动管理阶段切换)
- ✨ `ModelGroup` 模型冻结/解冻控制
- ✨ TD3+BC 离线算法实现
- ✨ SAC 在线算法实现
- ✨ **Online RL 完整数据流验证通过** (DummyEnv → InferenceLoop → RolloutBuffer → SAC → WeightSync)
- ✨ **Reward 模块** - 统一奖励管理
  - `EnvReward`: 环境原始奖励（支持缩放/偏移）
  - `PotentialShapingReward`: 基于势函数的 Reward Shaping
  - `DistanceShapingReward`: 基于距离的奖励塑形
  - `RNDReward`: Random Network Distillation 内在奖励
  - `CompositeReward`: 多奖励组合
  - `RewardNormalizer`: 奖励归一化
  - `REWARD_REGISTRY` 注册表 + `create_reward()` 工厂
- ✨ **Logger 模块** - 统一日志管理
  - `ConsoleLogger`: 彩色控制台输出
  - `JSONLogger`/`CSVLogger`: 文件记录
  - `TensorBoardLogger`: TensorBoard 集成
  - `WandBLogger`: Weights & Biases 集成
  - `CompositeLogger`: 多后端组合
  - `MetricsTracker`: 指标追踪（滑动窗口、Episode 统计）
  - `create_logger()` 便捷工厂函数
- ✨ 配置工厂方法：`make_bc_config()`, `make_td3bc_config()`, `make_sac_config()`
- ✨ `algo_kwargs` 支持算法特定参数
- ✨ 完整的类型提示（type hints）
- ✨ `DummyEnv` 增强：支持可配置维度、确定性模式、自定义奖励函数
- ✨ `scripts/verify_online_flow.py` Online 数据流验证脚本
- ✨ `scripts/train_online.py` Online SAC 训练入口
- 📝 完善的文档：`ARCHITECTURE.md` 设计哲学文档
- 📝 `.github/copilot-instructions.md` AI 编码助手指南

### 改进
- ♻️ 重构 `AlgorithmConfig`，统一参数管理
- ♻️ 优化 `DataHub` 采样策略接口
- ♻️ 改进 `TrainingLoop` 算法实例缓存机制
- 🎨 统一代码风格和命名规范

### 修复
- 🐛 修复 `Batch.to(device)` 类型转换问题
- 🐛 修复 `ModelGroup.trainable_parameters()` 参数过滤逻辑
- 🐛 **修复 `InferenceLoop` 数据无法写入 DataHub 的严重 Bug**
  - **问题**: `collect_rollout()` 和 `_run_episode()` 中使用 `if self.data_hub:` 判断
  - **原因**: `DataHub.__len__()` 返回所有 buffer 总长度，空时为 0，导致 `bool(data_hub) == False`
  - **修复**: 改为 `if self.data_hub is not None:` 进行显式 None 检查
  - **影响文件**: `core/inference_loop.py` (第 180 行、第 279 行)

---

## [0.1.0] - 2025-12

### 新增
- 🎉 初始版本发布
- ✨ 基础框架搭建：模块化架构
- ✨ 注册表模式：`ALGORITHM_REGISTRY`, `POLICY_REGISTRY`, `ENV_REGISTRY`, `STRATEGY_REGISTRY`
- ✨ BC (Behavior Cloning) 算法实现
- ✨ `DataHub` 统一数据管理（Demo/Rollout/Intervention）
- ✨ `HDF5Buffer` 支持专家演示数据加载
- ✨ `RolloutBuffer` FIFO 环形缓冲区
- ✨ `MLPPolicy` / `MLPGaussianPolicy` 基础策略网络
- ✨ `DemoOnlyStrategy` / `RolloutOnlyStrategy` / `MixedStrategy` 采样策略
- ✨ YAML 配置系统
- ✨ `TrainingLoop` 训练循环基础实现

### 文档
- 📝 README.md 框架说明
- 📝 代码注释和类型提示

---

## [Unreleased] - 未来计划

### 计划新增
- [ ] CQL (Conservative Q-Learning) 算法
- [ ] IQL (Implicit Q-Learning) 算法
- [ ] AWR (Advantage Weighted Regression) 算法
- [ ] VF Regression 价值函数回归
- [ ] InferenceLoop 完整实现
- [ ] WeightSync 多进程同步优化
- [ ] TensorBoard / WandB 日志集成
- [ ] 单元测试套件
- [ ] 分布式训练支持

### 计划改进
- [ ] 配置验证和错误提示优化
- [ ] 性能优化：数据加载并行化
- [ ] 更多预设配置模板
- [ ] 可视化工具

---

## 版本说明

- **0.x.x** - 开发版本，API 可能变更
- **1.0.0** - 稳定版本发布（计划中）

---

## 如何升级

### 从 0.1.0 升级到 0.2.0

1. **配置文件更新**
```yaml
# 旧版本（0.1.0）
algorithm:
  name: "bc"
  lr: 1e-4

# 新版本（0.2.0）- 新增 algo_kwargs
algorithm:
  name: "td3_bc"
  lr: 1e-4
  algo_kwargs:
    bc_alpha: 2.5
    policy_noise: 0.2
```

2. **训练脚本更新**
```python
# 旧版本
trainer = TrainingLoop(model_group, data_hub, config)

# 新版本 - 需要传入 algo_config
trainer = TrainingLoop(
    model_group, 
    data_hub, 
    config.training,
    algo_config=config.algorithm
)
```

3. **依赖更新**
```bash
pip install --upgrade -r requirements.txt
```

---

## 贡献指南

欢迎提交 Issue 或 Pull Request！请遵循：
- 新功能：在 [Unreleased] 添加说明
- Bug 修复：在当前版本添加 `### 修复`
- 文档更新：在当前版本添加 `### 文档`
