#!/usr/bin/env python
"""
统一训练脚本

支持多种训练模式，通过配置文件中的 hil.enabled 控制：
- 普通训练（Offline/Online）：hil.enabled = false 或未配置
- HIL 训练：hil.enabled = true

HIL 模式支持：
- local: 单进程调试（Actor/Learner 交替执行）
- distributed: 分布式训练（需要分别启动 Learner 和 Actor）

使用示例:
    # 普通训练
    python scripts/train.py --config projects/_template/config.yaml --steps 10000
    
    # HIL 本地调试模式（config 中 hil.enabled: true）
    python scripts/train.py --config projects/h1_hil/config.yaml --steps 10000
    
    # HIL 分布式模式 - Learner 端
    python scripts/train.py --config projects/h1_hil/config.yaml --role learner
    
    # HIL 分布式模式 - Actor 端（另一个终端）
    python scripts/train.py --config projects/h1_hil/config.yaml --role actor
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_yaml, Logger
from core.orchestration import SystemBuilder, REGISTRY
from core.runtime import TrainingLoop
from data import DataHub
from data.samplers import UniformSampler

# 导入模块以触发装饰器注册
import env 
import algorithms
import policies
import data.buffers
import data.samplers


def run_standard_training(components, config, args, logger):
    """普通训练流程（Offline/Online）"""
    # 设置DataHub
    data_hub = DataHub()
    for name, buf in components.buffers.items():
        data_hub.register_buffer(name, buf)
    
    # 获取采样器（优先使用配置中的，否则用默认的）
    sampler = components.sampler or UniformSampler()
    
    # 创建训练循环
    train_loop = TrainingLoop(
        algorithm=components.algorithm,
        data_hub=data_hub,
        sampler=sampler,
        config=config,
        weight_sync=components.weight_sync,
        device=args.device,
    )
    
    # 训练
    logger.info(f"Starting standard training for {args.steps} steps")
    results = train_loop.run(args.steps, log_freq=config.get("log_freq", 100))
    return results


def run_hil_training(components, config, args, logger):
    """HIL 训练流程"""
    from core.interfaces.adapters import StandardPolicyAdapter, AlgorithmAdapter
    from core.runtime import HILTrainer
    
    # 创建 Adapters
    actor_adapter = StandardPolicyAdapter(components.algorithm.get_policy())
    learner_adapter = AlgorithmAdapter(components.algorithm)
    
    # 加载 demo 数据（如果配置了）
    demo_buffer = components.buffers.get("demo")
    if demo_buffer:
        logger.info(f"Loaded demo buffer with {len(demo_buffer)} samples")
    
    # 加载 reward classifier（如果配置了）
    reward_classifier = None
    classifier_config = config.get("hil", {}).get("reward_classifier", {})
    classifier_path = classifier_config.get("checkpoint")
    if classifier_path:
        logger.info(f"Loading reward classifier from {classifier_path}")
        # TODO: 实现 classifier 加载
    
    # 确定 HIL 模式
    hil_mode = config.get("hil", {}).get("mode", "local")
    if args.role:
        hil_mode = "grpc"  # 指定 role 时自动切换到分布式
    
    # 创建 HILTrainer
    trainer = HILTrainer(
        actor_adapter=actor_adapter,
        learner_adapter=learner_adapter,
        env=components.env,
        config=config,
        demo_buffer=demo_buffer,
        reward_classifier=reward_classifier,
        mode=hil_mode,
    )
    
    # 运行
    if args.role:
        # 分布式模式
        logger.info(f"Starting distributed HIL training as {args.role}")
        results = trainer.run_distributed(role=args.role, num_steps=args.steps)
    else:
        # 本地模式
        logger.info(f"Starting local HIL training for {args.steps} steps")
        results = trainer.run_local(num_steps=args.steps, log_freq=config.get("log_freq", 100))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RL Framework Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--role", type=str, default=None, choices=["learner", "actor"],
                        help="Role for HIL distributed mode (optional)")
    args = parser.parse_args()
    
    # 加载配置
    config = load_yaml(args.config)
    
    # 命令行参数覆盖配置文件中的 device
    config["device"] = args.device
    if "policy" in config:
        config["policy"]["device"] = args.device
    if "algorithm" in config:
        config["algorithm"]["device"] = args.device
    
    # 初始化日志
    logger = Logger(log_dir=config.get("log_dir", "./logs"))
    logger.info(f"Loading config from {args.config}")
    
    # 构建系统
    builder = SystemBuilder(REGISTRY)
    components = builder.build_from_config(config)
    
    # 检查是否启用 HIL
    hil_enabled = config.get("hil", {}).get("enabled", False)
    
    if hil_enabled:
        logger.info("HIL mode enabled")
        results = run_hil_training(components, config, args, logger)
    else:
        logger.info("Standard training mode")
        results = run_standard_training(components, config, args, logger)
    
    logger.info(f"Training finished: {results}")
    logger.save_metrics()


if __name__ == "__main__":
    main()