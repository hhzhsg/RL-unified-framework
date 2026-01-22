"""
HIL-SERL Actor Loop

负责:
1. 在环境中执行策略
2. 检测并处理人类干预
3. 使用奖励分类器预测奖励
4. 发送transitions到Learner
5. 同步最新权重
"""
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import time
import numpy as np
import torch

from core.runtime.base_loop import BaseLoop
from core.interfaces import EnvInterface, PolicyInterface


@dataclass
class HILSerlActorConfig:
    """HIL-SERL Actor配置"""
    # 干预相关
    intervention_key: str = "intervene_action"
    intervention_source: str = "gamepad"  # "gamepad" | "spacemouse" | "leader_arm"
    
    # 奖励分类器
    use_reward_classifier: bool = False
    reward_classifier_threshold: float = 0.5
    camera_keys: List[str] = field(default_factory=lambda: ["images.front", "images.side"])
    
    # 同步
    weight_sync_freq: int = 1  # 每N步尝试同步权重
    transition_batch_size: int = 1  # 每次发送的transition数量
    
    # 策略
    deterministic: bool = False
    
    # Episode
    max_episode_steps: int = 200


class HILSerlActorLoop(BaseLoop):
    """
    HIL-SERL Actor循环
    
    核心职责:
    - 执行策略推理
    - 检测人类干预（SpaceMouse/Gamepad）
    - 使用奖励分类器
    - 发送数据到Learner
    - 同步权重
    """
    
    def __init__(
        self,
        policy: PolicyInterface,
        env: EnvInterface,
        actor_client,  # ActorClientInterface
        config: HILSerlActorConfig,
        reward_classifier: Optional[Any] = None,
    ):
        super().__init__()
        
        self.policy = policy
        self.env = env
        self.client = actor_client
        self.config = config
        self.reward_classifier = reward_classifier
        
        # 状态
        self._current_obs = None
        self._current_info = None
        self._episode_step = 0
        self._episode_count = 0
        self._episode_reward = 0.0
        
        # 统计
        self._intervention_count = 0
        self._total_actions = 0
        
        # Transition缓冲
        self._transition_buffer: List[Dict[str, Any]] = []
        
        # 标记是否已同步初始权重
        self._initial_weights_synced = False
    
    def wait_for_initial_weights(self, timeout: float = 60.0) -> bool:
        """
        等待Learner发送初始权重
        
        Args:
            timeout: 超时时间（秒）
        Returns:
            是否成功同步
        """
        print("[Actor] Waiting for initial weights from Learner...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            weights = self.client.recv_weights(block=True, timeout=5.0)
            if weights is not None:
                self.policy.load_state_dict(weights)
                self._initial_weights_synced = True
                print("[Actor] Initial weights received!")
                return True
        
        print("[Actor] Timeout waiting for initial weights")
        return False
    
    def step(self) -> Dict[str, Any]:
        """执行单步Actor逻辑"""
        # 0. 确保已同步初始权重
        if not self._initial_weights_synced:
            raise RuntimeError("Must call wait_for_initial_weights() first")
        
        # 1. 尝试同步最新权重（非阻塞）
        if self._step_count % self.config.weight_sync_freq == 0:
            self._try_sync_weights()
        
        # 2. 确保环境已重置
        if self._current_obs is None:
            self._current_obs, self._current_info = self.env.reset()
            self._episode_step = 0
            self._episode_reward = 0.0
        
        # 3. 获取动作（人类干预优先）
        action, is_intervention = self._get_action(self._current_obs, self._current_info)
        
        # 4. 执行动作
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # 5. 计算奖励（可能使用奖励分类器）
        reward = self._compute_reward(env_reward, self._current_obs, info)
        
        # 6. 构建并缓冲transition
        transition = self._build_transition(
            obs=self._current_obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            is_intervention=is_intervention,
            info=info,
        )
        self._buffer_transition(transition)
        
        # 7. 更新统计
        self._episode_reward += reward
        self._episode_step += 1
        self._total_actions += 1
        if is_intervention:
            self._intervention_count += 1
        
        # 8. 构建返回信息
        step_info = {
            "reward": reward,
            "done": done,
            "is_intervention": is_intervention,
            "episode_step": self._episode_step,
        }
        
        # 9. Episode结束处理
        if done:
            step_info["episode_reward"] = self._episode_reward
            step_info["episode_length"] = self._episode_step
            step_info["intervention_rate"] = self._intervention_count / max(self._total_actions, 1)
            
            self._episode_count += 1
            self._current_obs, self._current_info = self.env.reset()
            self._episode_step = 0
            self._episode_reward = 0.0
        else:
            self._current_obs = next_obs
            self._current_info = info
        
        return step_info
    
    def _get_action(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        获取动作
        
        优先检查人类干预，否则使用策略
        
        Returns:
            (action, is_intervention)
        """
        # 检查人类干预
        if self.config.intervention_key in info:
            intervention_action = info[self.config.intervention_key]
            if intervention_action is not None:
                return np.array(intervention_action), True
        
        # 使用策略
        action = self.policy.act(obs, deterministic=self.config.deterministic)
        return action, False
    
    def _compute_reward(self, env_reward: float, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        计算奖励
        
        可使用奖励分类器替代/补充环境奖励
        """
        if not self.config.use_reward_classifier or self.reward_classifier is None:
            return env_reward
        
        # 收集相机图像
        images = []
        for key in self.config.camera_keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                images.append(img.unsqueeze(0))  # 添加batch维度
        
        if not images:
            return env_reward
        
        # 使用奖励分类器预测
        with torch.no_grad():
            predicted_reward = self.reward_classifier.predict_reward(
                images, 
                threshold=self.config.reward_classifier_threshold
            )
        
        return float(predicted_reward)
    
    def _build_transition(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
        is_intervention: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建transition字典"""
        # 提取state向量
        state = obs.get("state", obs.get("observation", None))
        next_state = next_obs.get("state", next_obs.get("observation", None))
        
        return {
            "obs": state if isinstance(state, np.ndarray) else np.array(state),
            "action": action,
            "reward": reward,
            "next_obs": next_state if isinstance(next_state, np.ndarray) else np.array(next_state),
            "done": done,
            "is_intervention": is_intervention,
            "source": "intervention" if is_intervention else "rollout",
        }
    
    def _buffer_transition(self, transition: Dict[str, Any]) -> None:
        """缓冲transition并在达到批量大小时发送"""
        self._transition_buffer.append(transition)
        
        if len(self._transition_buffer) >= self.config.transition_batch_size:
            self._flush_transitions()
    
    def _flush_transitions(self) -> None:
        """发送所有缓冲的transitions到Learner"""
        if self._transition_buffer:
            self.client.send_transitions(self._transition_buffer)
            self._transition_buffer = []
    
    def _try_sync_weights(self) -> bool:
        """尝试同步最新权重（非阻塞）"""
        weights = self.client.recv_weights(block=False)
        if weights is not None:
            self.policy.load_state_dict(weights)
            return True
        return False
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            "episode_count": self._episode_count,
            "total_steps": self._step_count,
            "intervention_rate": self._intervention_count / max(self._total_actions, 1),
            "total_interventions": self._intervention_count,
        }
    
    def stop(self) -> None:
        """停止Actor循环"""
        super().stop()
        self._flush_transitions()  # 发送剩余数据
