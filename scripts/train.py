#!/usr/bin/env python
"""训练脚本"""
import argparse
from pathlib import Path

from utils import load_yaml, Logger
from core.orchestration import SystemBuilder, REGISTRY
from core.runtime import TrainingLoop
from data import DataHub
from data.samplers import UniformSampler


def main():
    parser = argparse.ArgumentParser(description="RL Framework Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    args = parser.parse_args()
    
    # 加载配置
    config = load_yaml(args.config)
    config["device"] = args.device
    
    # 初始化日志
    logger = Logger(log_dir=config.get("log_dir", "./logs"))
    logger.info(f"Loading config from {args.config}")
    
    # 构建系统
    builder = SystemBuilder(REGISTRY)
    components = builder.build_from_config(config)
    
    # 设置DataHub
    data_hub = DataHub()
    for name, buf in components.buffers.items():
        data_hub.add_buffer(name, buf)
    data_hub.set_sampler(components.sampler or UniformSampler())
    
    # 创建训练循环
    train_loop = TrainingLoop(
        algorithm=components.algorithm,
        data_hub=data_hub,
        sampler=data_hub._sampler,
        config=config,
        weight_sync=components.weight_sync,
        device=args.device,
    )
    
    # 训练
    logger.info(f"Starting training for {args.steps} steps")
    results = train_loop.run(args.steps, log_freq=config.get("log_freq", 100))
    
    logger.info(f"Training finished: {results}")
    logger.save_metrics()


if __name__ == "__main__":
    main()