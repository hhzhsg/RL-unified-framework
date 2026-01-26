"""
Evaluator Loop（用于策略评估，无训练）

业界标准命名：Actor-Learner-Evaluator 架构
- Actor: 与环境交互，收集数据
- Learner: 从数据学习，更新策略
- Evaluator: 评估策略性能

适用场景:
- 评估训练好的策略
- 对比不同 checkpoint 的性能
- 训练过程中定期评估
- 生成评估报告

与其他 Loop 的关系:
- 独立使用，不参与训练流程
- 只执行策略，不更新权重
- 不收集训练数据

职责:
- 在环境中评估策略（deterministic=True）
- 收集评估指标（reward, success_rate, episode_length）
- 支持多 episode 评估并统计
- 可选：视频录制、结果保存
"""
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import time

from .base_loop import BaseLoop
from ..interfaces import EnvInterface, PolicyInterface


class EvaluatorLoop(BaseLoop):
    """
    Evaluator Loop - 策略评估
    
    职责:
    - 在环境中评估策略（deterministic=True）
    - 收集评估指标
    - 支持多 episode 评估并统计
    
    使用示例:
        evaluator = EvaluatorLoop(policy, env, config)
        results = evaluator.evaluate(num_episodes=10)
        print(f"Avg reward: {results['avg_reward']}")
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        env: EnvInterface,
        config: Optional[Dict[str, Any]] = None,
        video_recorder: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.policy = policy
        self.env = env
        self.config = config or {}
        self.video_recorder = video_recorder
        
        # 评估结果
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_successes: List[bool] = []
        self._episode_infos: List[Dict[str, Any]] = []
        
        # 当前 episode 状态
        self._current_obs = None
        self._episode_reward = 0.0
        self._episode_length = 0
        
        # 配置
        self.render = self.config.get("render", False)
        self.max_episode_steps = self.config.get("max_episode_steps", 1000)
    
    def step(self) -> Dict[str, Any]:
        """执行单步评估"""
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()
        
        # 确定性动作（评估时不探索）
        action = self.policy.act(self._current_obs, deterministic=True)
        
        # 执行
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated or (self._episode_length >= self.max_episode_steps)
        
        # 渲染（可选）
        if self.render:
            self.env.render()
        
        self._episode_reward += reward
        self._episode_length += 1
        
        step_info = {"reward": reward, "done": done}
        
        if done:
            self._episode_rewards.append(self._episode_reward)
            self._episode_lengths.append(self._episode_length)
            self._episode_successes.append(info.get("success", False))
            self._episode_infos.append(info)
            
            step_info.update({
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_length,
                "episode_success": info.get("success", False),
            })
            
            # 重置
            self._episode_reward = 0.0
            self._episode_length = 0
            self._current_obs, _ = self.env.reset()
        else:
            self._current_obs = next_obs
        
        return step_info
    
    def evaluate(self, num_episodes: int, verbose: bool = True) -> Dict[str, float]:
        """
        评估指定数量的 episode
        
        Args:
            num_episodes: 评估 episode 数
            verbose: 是否打印进度
            
        Returns:
            评估结果字典
        """
        self.reset_statistics()
        start_time = time.time()
        
        while len(self._episode_rewards) < num_episodes:
            self.step()
            self._step_count += 1
            
            if verbose and len(self._episode_rewards) > 0:
                # 每完成一个 episode 打印进度
                if len(self._episode_rewards) != getattr(self, '_last_printed', 0):
                    self._last_printed = len(self._episode_rewards)
                    elapsed = time.time() - start_time
                    print(f"[Evaluator] Episode {len(self._episode_rewards)}/{num_episodes}, "
                          f"reward={self._episode_rewards[-1]:.2f}, "
                          f"elapsed={elapsed:.1f}s")
        
        results = self.get_statistics()
        
        if verbose:
            print(f"\n[Evaluator] Evaluation complete:")
            print(f"  Episodes: {num_episodes}")
            print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Success Rate: {results['success_rate']*100:.1f}%")
            print(f"  Avg Length: {results['avg_length']:.1f}")
        
        return results
    
    def get_statistics(self) -> Dict[str, float]:
        """获取评估统计"""
        if not self._episode_rewards:
            return {
                "num_episodes": 0,
                "avg_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "avg_length": 0.0,
                "success_rate": 0.0,
            }
        
        return {
            "num_episodes": len(self._episode_rewards),
            "avg_reward": float(np.mean(self._episode_rewards)),
            "std_reward": float(np.std(self._episode_rewards)),
            "min_reward": float(np.min(self._episode_rewards)),
            "max_reward": float(np.max(self._episode_rewards)),
            "avg_length": float(np.mean(self._episode_lengths)),
            "success_rate": float(np.mean(self._episode_successes)),
        }
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_successes = []
        self._episode_infos = []
        self._last_printed = 0
    
    def get_all_episode_infos(self) -> List[Dict[str, Any]]:
        """获取所有 episode 的详细信息"""
        return self._episode_infos.copy()


# 向后兼容别名
EvaluationLoop = EvaluatorLoop
