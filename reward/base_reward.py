"""
VLA-RL Reward 基类
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from data import Transition, Batch


class BaseReward(ABC):
    """
    奖励函数基类
    
    所有自定义奖励需要继承此类并实现:
    - compute(): 计算单步奖励
    - transform_batch() (可选): 批量转换奖励
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self._call_count = 0
    
    @abstractmethod
    def compute(self, 
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        """
        计算单步奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            env_reward: 环境原始奖励
            done: 是否结束
            info: 额外信息
            
        Returns:
            计算后的奖励
        """
        pass
    
    def compute_from_transition(self, transition: Transition) -> float:
        """从 Transition 计算奖励"""
        return self.compute(
            state=transition.robot_state.to_array(),
            action=transition.action.data,
            next_state=transition.next_robot_state.to_array(),
            env_reward=transition.reward,
            done=transition.done,
            info=None,
        )
    
    def transform_batch(self, batch: Batch) -> Batch:
        """
        批量转换奖励
        
        默认实现：逐个计算。子类可以重写以利用向量化加速。
        
        Args:
            batch: 原始批次
            
        Returns:
            奖励转换后的批次
        """
        new_rewards = []
        for i in range(len(batch)):
            reward = self.compute(
                state=batch.robot_state[i] if hasattr(batch.robot_state, '__getitem__') else batch.robot_state,
                action=batch.action[i] if hasattr(batch.action, '__getitem__') else batch.action,
                next_state=batch.next_robot_state[i] if hasattr(batch.next_robot_state, '__getitem__') else batch.next_robot_state,
                env_reward=float(batch.reward[i]) if hasattr(batch.reward, '__getitem__') else float(batch.reward),
                done=bool(batch.done[i]) if hasattr(batch.done, '__getitem__') else bool(batch.done),
                info=None,
            )
            new_rewards.append(reward)
        
        # 创建新的 batch，只替换 reward
        import copy
        new_batch = copy.copy(batch)
        new_batch.reward = np.array(new_rewards, dtype=np.float32)
        return new_batch
    
    def reset(self):
        """重置内部状态（如果有的话）"""
        pass
    
    def update(self, batch: Batch):
        """
        更新奖励模型（用于学习型奖励如 RND）
        
        Args:
            batch: 训练批次
        """
        pass
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {"call_count": self._call_count}
    
    def __call__(self, *args, **kwargs) -> float:
        self._call_count += 1
        return self.compute(*args, **kwargs)


class RewardWrapper(BaseReward):
    """
    奖励包装器
    
    用于组合多个奖励或添加后处理
    """
    
    def __init__(self, base_reward: BaseReward, name: str = "wrapper"):
        super().__init__(name)
        self.base_reward = base_reward
    
    def compute(self, 
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        return self.base_reward.compute(state, action, next_state, env_reward, done, info)
    
    def reset(self):
        self.base_reward.reset()
    
    def update(self, batch: Batch):
        self.base_reward.update(batch)
