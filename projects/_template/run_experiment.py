#!/usr/bin/env python
"""实验运行脚本模板"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import load_yaml, Logger
from core.orchestration import SystemBuilder, REGISTRY
from core.runtime import TrainingLoop
from data import DataHub
from data.samplers import UniformSampler


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(str(config_path))
    
    logger = Logger(log_dir=config["training"]["log_dir"])
    logger.info("Starting experiment")
    
    # 构建系统
    builder = SystemBuilder(REGISTRY)
    components = builder.build_from_config(config)
    
    # 设置DataHub
    data_hub = DataHub()
    for name, buf in components.buffers.items():
        data_hub.add_buffer(name, buf)
    data_hub.set_sampler(components.sampler or UniformSampler())
    
    # 训练
    train_loop = TrainingLoop(
        algorithm=components.algorithm,
        data_hub=data_hub,
        sampler=data_hub._sampler,
        config=config["training"],
        device=config["algorithm"]["device"],
    )
    
    results = train_loop.run(
        config["training"]["total_steps"],
        log_freq=config["training"]["log_freq"]
    )
    
    logger.info(f"Experiment finished: {results}")


if __name__ == "__main__":
    main()
