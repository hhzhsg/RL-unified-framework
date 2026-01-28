# H1_act - ACT++ HIL 推理

## 更新历史
- 2025-01-23: 初始版本，ACT++ 推理 + HIL 干预

## 功能
- 使用预训练 ACT++ 模型进行自动 rollout
- VR 操作员可随时介入控制
- 自动收集 rollout + intervention 数据
- 支持 standalone（本地收集）和分布式（连接 Learner）两种模式

## 文件结构
```
H1_act/
├── act_policy.py      # ACT++ 策略适配器
├── run_act_hil.py     # 运行脚本
├── config.yaml        # 配置文件
├── checkpoints/       # 预训练模型（需要复制）
│   ├── policy_last.ckpt
│   └── dataset_stats.pkl
└── collected_data/    # 收集的数据（自动创建）
```

## 环境依赖
```bash
# 在机器人上应该已经安装了这些
pip install torch torchvision
pip install einops  # ACT 需要

# 如果没有 detr.models，需要从原始 ACT 仓库安装
# git clone https://github.com/tonyzhaozh/act
# cd act && pip install -e .
```

## 使用方式

### 1. 准备 checkpoint
把训练好的 ACT++ 模型复制到 `checkpoints/` 目录：
```bash
mkdir -p checkpoints
cp /path/to/policy_last.ckpt checkpoints/
cp /path/to/dataset_stats.pkl checkpoints/
```

### 2. Standalone 模式（推荐先用这个测试）
```bash
# 完整模式
python run_act_hil.py --standalone

# DRY-RUN 模式（不下发动作，测试数据流）
python run_act_hil.py --standalone --dry-run

# 自定义 episode 长度
python run_act_hil.py --standalone --max-episode-steps 300
```

### 3. 分布式模式（连接 Learner 做在线学习）
```bash
# 先在 GPU 服务器启动 Learner
# python run_learner.py --port 50060

# 然后在机器人端运行 Actor
python run_act_hil.py --learner-host 192.168.1.100 --learner-port 50060
```

## 相机配置
ACT++ 使用 4 个相机图像（双目分割）：
- `v4l2/cam_high_0/color` + `v4l2/cam_high_1/color`（头部双目）
- `v4l2/cam_right_wrist_0/color` + `v4l2/cam_right_wrist_1/color`（手腕双目）

确保 `H1RobotEnv` 的 `split_stereo=True` 以正确分割双目图像。

## 动作维度
- ACT++ 输出 15 维动作（上半身）
- `act_policy.py` 会自动 padding 到 H1 需要的完整维度

## 常见问题

### Q: 找不到 `detr.models`？
A: 需要安装 ACT 依赖，见上方环境依赖。

### Q: 相机图像是黑的？
A: 检查 camera_grpc_target 配置，确保 camera server 正在运行。

### Q: 动作不对劲？
A: 检查 `dataset_stats.pkl` 是否匹配训练数据，归一化错误会导致动作异常。
