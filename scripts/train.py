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
from core.runtime import LearnerLoop
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
    train_loop = LearnerLoop(
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
    """HIL 分布式训练流程"""
    from policies.adapters import StandardPolicyAdapter, AlgorithmAdapter
    from core.runtime import HILActorLoop, HILActorConfig, HILLearnerLoop, HILLearnerConfig
    from core.synchronization.actor_learner import ActorLearnerConfig
    
    # 同步配置
    sync_config = ActorLearnerConfig(**config.get("sync", {}))
    hil_config = config.get("hil", {})
    mode = hil_config.get("mode", "local")
    
    if args.role == "learner":
        # ===== Learner 进程 =====
        logger.info("Starting HIL Learner...")
        
        learner_adapter = AlgorithmAdapter(components.algorithm)
        learner_config = HILLearnerConfig(**hil_config.get("learner", {}))
        
        # 加载 demo 数据
        demo_buffer = components.buffers.get("demo")
        if demo_buffer:
            logger.info(f"Loaded demo buffer: {len(demo_buffer)} samples")
        
        learner = HILLearnerLoop(
            trainable_adapter=learner_adapter,
            config=learner_config,
            sync_config=sync_config,
            demo_buffer=demo_buffer,
            mode=mode,
        )
        
        results = learner.run(args.steps, log_freq=config.get("log_freq", 100))
        learner.cleanup()
        
    elif args.role == "actor":
        # ===== Actor 进程 =====
        logger.info("Starting HIL Actor...")
        
        # 获取基础环境
        env = components.env
        
        # 根据配置添加 Wrapper
        wrapper_config = hil_config.get("wrapper", {})
        if wrapper_config.get("enabled", False):
            wrapper_type = wrapper_config.get("type", "vr")
            logger.info(f"Wrapping env with {wrapper_type} intervention wrapper")
            env = _wrap_env_with_intervention(env, wrapper_type, wrapper_config)
        
        actor_adapter = StandardPolicyAdapter(components.algorithm.get_policy())
        actor_config = HILActorConfig(**hil_config.get("actor", {}))
        
        # 加载 reward classifier
        reward_classifier = None
        classifier_path = hil_config.get("reward_classifier", {}).get("checkpoint")
        if classifier_path:
            logger.info(f"Loading reward classifier: {classifier_path}")
            # TODO: 实现 classifier 加载
        
        actor = HILActorLoop(
            policy_adapter=actor_adapter,
            env=env,
            config=actor_config,
            sync_config=sync_config,
            reward_classifier=reward_classifier,
            mode=mode,
        )
        
        results = actor.run(args.steps, log_freq=config.get("log_freq", 100))
        actor.cleanup()
    
    else:
        raise ValueError("HIL mode requires --role (learner or actor)")
    
    return results


def _wrap_env_with_intervention(env, wrapper_type: str, wrapper_config: dict):
    """
    根据配置包装环境
    
    Config 示例:
        hil:
          wrapper:
            enabled: true
            type: vr  # vr | keyboard | mock
            vr_device: quest
            action_scale: 1.0
    """
    if wrapper_type == "vr":
        from env.wrappers import VRWrapper
        return VRWrapper(
            env,
            vr_device=wrapper_config.get("vr_device", "quest"),
            action_scale=wrapper_config.get("action_scale", 1.0),
            trigger_button=wrapper_config.get("trigger_button", "grip"),
        )
    elif wrapper_type == "keyboard":
        # TODO: 实现键盘干预 wrapper
        raise NotImplementedError("Keyboard wrapper not implemented")
    elif wrapper_type == "mock":
        # Mock wrapper 用于测试，随机触发干预
        from env.wrappers import BaseInterventionWrapper
        # 直接返回原 env，不做包装（测试用）
        return env
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


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