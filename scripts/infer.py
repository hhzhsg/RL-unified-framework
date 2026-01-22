#!/usr/bin/env python
"""推理脚本"""
import argparse
from utils import load_yaml
from core.orchestration import SystemBuilder, REGISTRY
from core.runtime import InferenceLoop


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
    
    # 推理循环
    infer_loop = InferenceLoop(
        policy=policy,
        env=components.env,
        config={"deterministic": True},
    )
    
    results = infer_loop.evaluate(args.episodes)
    print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
