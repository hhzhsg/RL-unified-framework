"""
Metrics 追踪器
"""
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodeMetrics:
    """单个 Episode 的指标"""
    total_reward: float = 0.0
    length: int = 0
    success: bool = False
    task_id: str = ""
    
    # 额外指标
    extra: Dict[str, float] = field(default_factory=dict)


class MetricsTracker:
    """
    训练指标追踪器
    
    功能：
    - 累积和平均指标
    - Episode 统计
    - 滑动窗口统计
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        
        # 累积值（用于计算平均）
        self._cumulative: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        
        # 滑动窗口
        self._windows: Dict[str, List[float]] = defaultdict(list)
        
        # Episode 历史
        self._episodes: List[EpisodeMetrics] = []
        self._current_episode = EpisodeMetrics()
        
        # 全局统计
        self._total_steps = 0
        self._total_episodes = 0
    
    def add(self, tag: str, value: float):
        """添加单个值"""
        self._cumulative[tag] += value
        self._counts[tag] += 1
        
        # 更新滑动窗口
        window = self._windows[tag]
        window.append(value)
        if len(window) > self.window_size:
            window.pop(0)
    
    def add_dict(self, metrics: Dict[str, float]):
        """批量添加"""
        for tag, value in metrics.items():
            self.add(tag, value)
    
    def get_mean(self, tag: str) -> float:
        """获取累积平均"""
        if self._counts[tag] == 0:
            return 0.0
        return self._cumulative[tag] / self._counts[tag]
    
    def get_window_mean(self, tag: str) -> float:
        """获取滑动窗口平均"""
        window = self._windows[tag]
        if not window:
            return 0.0
        return np.mean(window)
    
    def get_window_std(self, tag: str) -> float:
        """获取滑动窗口标准差"""
        window = self._windows[tag]
        if len(window) < 2:
            return 0.0
        return np.std(window)
    
    def get_all_means(self) -> Dict[str, float]:
        """获取所有累积平均"""
        return {tag: self.get_mean(tag) for tag in self._cumulative}
    
    def get_all_window_means(self) -> Dict[str, float]:
        """获取所有滑动窗口平均"""
        return {tag: self.get_window_mean(tag) for tag in self._windows}
    
    # ========== Episode 追踪 ==========
    
    def add_episode_step(self, reward: float, done: bool = False):
        """添加 episode 内的一步"""
        self._current_episode.total_reward += reward
        self._current_episode.length += 1
        self._total_steps += 1
        
        if done:
            self.end_episode()
    
    def end_episode(self, success: bool = False, task_id: str = "", extra: Dict[str, float] = None):
        """结束当前 episode"""
        self._current_episode.success = success
        self._current_episode.task_id = task_id
        if extra:
            self._current_episode.extra = extra
        
        self._episodes.append(self._current_episode)
        self._total_episodes += 1
        
        # 添加到指标追踪
        self.add("episode/reward", self._current_episode.total_reward)
        self.add("episode/length", self._current_episode.length)
        self.add("episode/success", float(success))
        
        # 重置
        self._current_episode = EpisodeMetrics()
        
        # 保持历史窗口
        if len(self._episodes) > self.window_size * 10:
            self._episodes = self._episodes[-self.window_size * 10:]
    
    def get_episode_stats(self) -> Dict[str, float]:
        """获取 episode 统计"""
        if not self._episodes:
            return {}
        
        recent = self._episodes[-self.window_size:]
        
        return {
            "episode/reward_mean": np.mean([e.total_reward for e in recent]),
            "episode/reward_std": np.std([e.total_reward for e in recent]),
            "episode/length_mean": np.mean([e.length for e in recent]),
            "episode/success_rate": np.mean([float(e.success) for e in recent]),
            "episode/total": self._total_episodes,
        }
    
    def reset(self):
        """重置所有统计"""
        self._cumulative.clear()
        self._counts.clear()
        self._windows.clear()
        self._episodes.clear()
        self._current_episode = EpisodeMetrics()
        self._total_steps = 0
        self._total_episodes = 0
    
    def reset_cumulative(self):
        """只重置累积统计（保留 episode 历史）"""
        self._cumulative.clear()
        self._counts.clear()
    
    @property
    def total_steps(self) -> int:
        return self._total_steps
    
    @property
    def total_episodes(self) -> int:
        return self._total_episodes


class TrainingMetrics:
    """
    训练过程专用指标
    
    跟踪:
    - Loss
    - Learning rate
    - Gradient norm
    - Q-values (for RL)
    """
    
    def __init__(self, window_size: int = 100):
        self.tracker = MetricsTracker(window_size)
        
        # 记录最佳值
        self._best: Dict[str, float] = {}
    
    def log_train_step(self, metrics: Dict[str, float], step: int):
        """记录训练步指标"""
        self.tracker.add_dict(metrics)
        
        # 更新最佳值
        for tag, value in metrics.items():
            if "loss" in tag.lower():
                if tag not in self._best or value < self._best[tag]:
                    self._best[tag] = value
    
    def get_summary(self) -> Dict[str, float]:
        """获取汇总"""
        summary = self.tracker.get_all_window_means()
        
        # 添加最佳值
        for tag, value in self._best.items():
            summary[f"best/{tag}"] = value
        
        return summary
