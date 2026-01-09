# 📖 VLA-RL 文档导航

> 根据您的需求，快速找到对应文档

---

## 🚀 我想快速开始

**推荐阅读顺序**：
1. [README.md](README.md) - 5 分钟了解项目
2. [QUICKREF.md](QUICKREF.md) - 查看配置模板，复制粘贴开始实验
3. [示例代码](scripts/train.py) - 看看实际如何运行

**适合人群**：
- ✅ 第一次使用框架
- ✅ 想快速跑通一个实验
- ✅ 有一定 RL 基础

---

## 🧩 我想添加新算法

**推荐阅读顺序**：
1. [QUICKREF.md - 添加新算法速查](QUICKREF.md#-添加新算法速查) - 4 步添加算法
2. [.github/copilot-instructions.md](..github/copilot-instructions.md) - 详细的算法开发模式
3. [algorithm/bc.py](algorithm/bc.py) - 参考 BC 实现
4. [ARCHITECTURE.md - Algorithm 模块](ARCHITECTURE.md#2-algorithm-模块算法开发的标准化) - 理解设计原理

**适合人群**：
- ✅ 要实现自己的 RL 算法
- ✅ 想理解算法接口设计

---

## 🏗️ 我想理解框架设计

**推荐阅读顺序**：
1. [ARCHITECTURE.md](ARCHITECTURE.md) - **核心文档**，深入讲解设计哲学
2. [README.md - 设计理念](README.md#设计理念) - 快速概览
3. [.github/copilot-instructions.md](..github/copilot-instructions.md) - 实践指南

**适合人群**：
- ✅ 想深入理解框架为什么这样设计
- ✅ 准备贡献代码或大幅扩展框架
- ✅ 对软件架构感兴趣

**核心内容**：
- 设计原则（SOLID）
- 设计模式（注册表、策略、外观、模板方法）
- 模块化分析
- 扩展性分析

---

## 🔧 我想自定义某个模块

### 数据管理（Buffer）
- [ARCHITECTURE.md - Buffer 模块](ARCHITECTURE.md#1-buffer-模块数据管理的优雅抽象)
- [buffer/sample_strategy.py](buffer/sample_strategy.py) - 采样策略实现
- [buffer/data_hub.py](buffer/data_hub.py) - 数据中心实现

### 模型管理（ModelGroup）
- [ARCHITECTURE.md - ModelGroup 模块](ARCHITECTURE.md#3-modelgroup-模块模型生命周期管理)
- [model/model_group.py](model/model_group.py) - 源码
- [QUICKREF.md - ModelGroup 操作](QUICKREF.md#-modelgroup-常用操作)

### 训练流程（TrainingLoop）
- [ARCHITECTURE.md - TrainingLoop 模块](ARCHITECTURE.md#4-trainingloop-模块训练流程的编排器)
- [core/training_loop.py](core/training_loop.py) - 源码
- [core/stage.py](core/stage.py) - 多阶段训练

### 策略网络（Policy）
- [model/base_policy.py](model/base_policy.py) - 策略基类
- [model/mlp_policy.py](model/mlp_policy.py) - MLP 策略实现
- [QUICKREF.md - Policy 接口](QUICKREF.md#-policy-接口速查)

---

## 🐛 我遇到了问题

### 第一步：查看常见问题
- [QUICKREF.md - 常见问题](QUICKREF.md#-常见问题速查)

### 常见问题类型：
- **配置问题** → [QUICKREF.md - 配置模板](QUICKREF.md#-常用配置模板)
- **模型不训练** → [QUICKREF.md - Q1](QUICKREF.md#q1-模型不训练--loss-不下降)
- **找不到数据** → [QUICKREF.md - Q2](QUICKREF.md#q2-找不到数据--buffer-为空)
- **内存溢出** → [QUICKREF.md - Q3](QUICKREF.md#q3-cuda-out-of-memory)
- **自定义算法报错** → [QUICKREF.md - Q4](QUICKREF.md#q4-自定义算法找不到模型)

### 第二步：提交 Issue
- [GitHub Issues](https://github.com/hhzhsg/RL-unified-framework/issues)

---

## 📊 我想了解项目历史

**阅读**：
- [CHANGELOG.md](CHANGELOG.md) - 版本更新历史
- [README.md - 版本历史](README.md#-版本历史)

**了解**：
- 各版本新增特性
- 如何升级
- 未来计划

---

## 🤝 我想贡献代码

**推荐阅读顺序**：
1. [ARCHITECTURE.md](ARCHITECTURE.md) - 理解整体设计
2. [CHANGELOG.md - 贡献指南](CHANGELOG.md#贡献指南) - 提交规范
3. [README.md - 贡献](README.md#-贡献) - 流程说明

**贡献方式**：
- 🐛 修复 Bug
- ✨ 新增算法
- 📝 完善文档
- 💡 提出新特性建议

---

## 📚 完整文档列表

| 文档 | 类型 | 内容 | 适合人群 |
|-----|------|------|---------|
| [README.md](README.md) | 入门 | 项目概览、快速开始、使用示例 | 所有用户 |
| [QUICKREF.md](QUICKREF.md) | 速查 | 配置模板、API 速查、问题排查 | 日常使用 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 深度 | 设计哲学、模块分析、设计模式 | 深入理解 |
| [CHANGELOG.md](CHANGELOG.md) | 历史 | 版本更新、升级指南 | 版本管理 |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | AI | 注册表模式、扩展指南 | AI 辅助开发 |

---

## 🎯 按角色推荐

### 👨‍💻 算法研究员
想实现新的 RL 算法
```
README → QUICKREF(添加算法) → algorithm/bc.py → ARCHITECTURE(Algorithm模块)
```

### 🏗️ 框架开发者
想扩展框架核心功能
```
ARCHITECTURE → 源码阅读 → CHANGELOG(贡献指南)
```

### 🎓 学生/初学者
想学习 RL 框架设计
```
README → QUICKREF → ARCHITECTURE → 源码阅读
```

### 🚀 工程师
想快速应用到项目
```
README(快速开始) → QUICKREF(配置模板) → 开始实验
```

---

## 💡 阅读建议

### 🌟 新手路径（预计 30 分钟）
1. **README.md**（10 分钟）- 了解项目
2. **QUICKREF.md**（10 分钟）- 复制配置模板
3. **运行示例**（10 分钟）- 跑通第一个实验

### 🔥 进阶路径（预计 2 小时）
1. **README.md**（15 分钟）- 全面理解
2. **QUICKREF.md**（30 分钟）- 掌握常用操作
3. **ARCHITECTURE.md - 模块化设计**（45 分钟）- 理解设计
4. **源码阅读**（30 分钟）- 深入实现

### 🚀 专家路径（预计 4 小时）
1. **所有文档通读**（2 小时）
2. **源码全面阅读**（1.5 小时）
3. **实践：添加算法**（30 分钟）

---

## 🔗 外部资源

### RL 基础
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)

### Python 设计模式
- [Python Design Patterns](https://refactoring.guru/design-patterns/python)
- [Clean Architecture in Python](https://www.amazon.com/Clean-Architecture-Craftsmans-Software-Structure/dp/0134494164)

### 相关框架
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - 单文件 RL 实现
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) - 分布式 RL

---

**🎉 欢迎开始您的 VLA-RL 之旅！**

有问题？
- 💬 [提交 Issue](https://github.com/hhzhsg/RL-unified-framework/issues)
- 📧 联系作者：hhzhsg
