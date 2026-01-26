#!/usr/bin/env python
"""策略评估脚本"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_yaml
from core.orchestration import SystemBuilder, REGISTRY
from core.runtime import EvaluatorLoop


def main():
    parser = argparse.ArgumentParser(description="RL Framework Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    builder = SystemBuilder(REGISTRY)
    components = builder.build_from_config(config)
    
    # 加载checkpoint
    components.algorithm.load(args.checkpoint)
    policy = components.algorithm.get_policy()
    
    # 评估循环
    evaluator = EvaluatorLoop(
        policy=policy,
        env=components.env,
        config={"render": False},
    )
    
    results = evaluator.evaluate(args.episodes)
    print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
