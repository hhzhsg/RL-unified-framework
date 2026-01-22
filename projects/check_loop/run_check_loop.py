#!/usr/bin/env python3
"""Minimal online check: env -> inference -> DataHub -> sampler -> training

Run as a standalone script inside the repo root.
"""
import sys
import os
import time
import argparse
import numpy as np

# ensure repo on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.hub import DataHub
from data.samplers.uniform_sampler import UniformSampler
from core.runtime.inference_loop import InferenceLoop
from core.runtime.training_loop import TrainingLoop


class DummyEnv:
    def __init__(self, state_dim=8, action_dim=4, max_steps=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._step = 0
        self._max = max_steps

    def reset(self):
        self._step = 0
        obs = np.zeros(self.state_dim, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        next_obs = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        reward = float(-np.linalg.norm(next_obs))
        terminated = self._step >= self._max
        truncated = False
        info = {}
        return next_obs, reward, terminated, truncated, info


class DummyPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, obs, deterministic=False):
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

    def load_state_dict(self, sd):
        pass


class DummyAlgo:
    """A tiny algorithm that operates on numpy batches and can save a checkpoint file."""
    def __init__(self):
        self._train_step = 0

    def update(self, batch):
        self._train_step += 1
        obs = batch.get("obs")
        act = batch.get("action")
        loss = 0.0
        if obs is not None and act is not None:
            pred = obs[:, : act.shape[1]] * 0.01
            loss = float(((pred - act) ** 2).mean())
        return {"dummy_loss": loss}

    def get_policy(self):
        return DummyPolicy(action_dim=4)

    def save(self, path: str):
        # write a tiny file to simulate checkpoint
        with open(path, "wb") as f:
            f.write(b"dummy_checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./check_loop_checkpoints")
    args = parser.parse_args()

    # prepare
    hub = DataHub(rollout_capacity=10000, intervention_capacity=1000)
    env = DummyEnv(state_dim=8, action_dim=4, max_steps=1000)
    policy = DummyPolicy(action_dim=4)

    inf_cfg = {"deterministic": False, "sync_freq": 1000}
    inf_loop = InferenceLoop(policy=policy, env=env, config=inf_cfg, data_hub=hub, weight_sync=None)

    sampler = UniformSampler()
    train_cfg = {"batch_size": 16, "sync_freq": 1000, "checkpoint_freq": args.checkpoint_freq, "checkpoint_dir": args.checkpoint_dir}
    training_loop = TrainingLoop(algorithm=DummyAlgo(), data_hub=hub, sampler=sampler, config=train_cfg, weight_sync=None, device="cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Starting online check: steps=", args.steps)
    for i in range(args.steps):
        inf_info = inf_loop.step()
        try:
            train_metrics = training_loop.step()
        except Exception as e:
            train_metrics = {"error": str(e)}
        print(f"iter={i} inf_reward={inf_info.get('reward'):.4f} train_metrics={train_metrics} hub={hub}")
        time.sleep(0.01)

    print("Done. Checkpoints in:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
