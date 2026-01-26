"""
Actor Loop（适用于 Online RL，无人类干预）

业界标准命名：Actor-Learner 架构
- Actor: 与环境交互，收集数据
- Learner: 从数据学习，更新策略
- Evaluator: 评估策略性能

适用场景:
- Online RL: 配合 LearnerLoop 使用（ActorLoop 收集数据 → LearnerLoop 训练）
- 纯推理部署: 加载 checkpoint 执行策略

与 HIL 的关系:
- HIL 场景请使用 HILActorLoop，它额外负责:
  - 从 Wrapper 获取干预信息（VR/SpaceMouse）
  - 区分 intervention 和 rollout 数据
  - 与 Learner 通信（发送 transitions / 接收权重）
  - 使用奖励分类器预测奖励（可选）

职责:
- 在环境中执行策略
- 收集 rollout 数据到 DataHub
- 可选：从 Learner 同步权重
"""
from typing import Dict, Any, Optional
import numpy as np

from .base_loop import BaseLoop
from ..interfaces import EnvInterface, PolicyInterface, SyncInterface


class ActorLoop(BaseLoop):
    """
    Actor Loop - 环境交互与数据收集
    
    职责:
    - 在环境中执行策略
    - 收集 rollout 数据到 DataHub
    - 从 Learner 同步权重
    
    使用示例:
        actor = ActorLoop(policy, env, config, data_hub, weight_sync)
        actor.run(num_steps=10000)
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        env: EnvInterface,
        config: Dict[str, Any],
        data_hub: Optional[Any] = None,
        weight_sync: Optional[SyncInterface] = None,
    ):
        super().__init__()
        
        self.policy = policy
        self.env = env
        self.config = config
        self.data_hub = data_hub
        self.weight_sync = weight_sync
        
        # 状态
        self._current_obs = None
        self._current_info = None
        self._episode_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        
        # 配置
        self.deterministic = config.get("deterministic", False)
        self.sync_freq = config.get("sync_freq", 10)
    
    def step(self) -> Dict[str, Any]:
        """执行单步交互"""
        # 0. 尝试同步权重
        if self.weight_sync and self._step_count % self.sync_freq == 0:
            self._try_sync_weights()
        
        # 1. 确保环境已重置
        if self._current_obs is None:
            self._current_obs, self._current_info = self.env.reset()
        
        # 2. 获取动作
        action = self.policy.act(self._current_obs, deterministic=self.deterministic)
        
        # 3. 执行动作
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # 4. 记录数据
        if self.data_hub is not None:
            transition = {
                "obs": self._current_obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": info,
            }
            self.data_hub.add(transition, source="rollout")
        
        # 5. 更新状态
        self._episode_reward += reward
        self._episode_length += 1
        
        step_info = {
            "reward": reward,
            "done": done,
        }
        
        if done:
            step_info["episode_reward"] = self._episode_reward
            step_info["episode_length"] = self._episode_length
            self._episode_count += 1
            self._episode_reward = 0.0
            self._episode_length = 0
            self._current_obs, self._current_info = self.env.reset()
        else:
            self._current_obs = next_obs
            self._current_info = info
        
        return step_info
    
    def _try_sync_weights(self) -> None:
        """尝试从 Learner 同步权重"""
        result = self.weight_sync.pull(tag="policy_weights")
        if result is not None:
            if "policy" in result:
                self.policy.load_state_dict(result["policy"])
    
    @property
    def episode_count(self) -> int:
        return self._episode_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "episode_count": self._episode_count,
            "total_steps": self._step_count,
        }


# 向后兼容别名
InferenceLoop = ActorLoop
