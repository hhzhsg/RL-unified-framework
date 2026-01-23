"""
HIL Actor Loop

Human-in-the-Loop Actor 循环，与具体模型和干预设备解耦

设计原则：
1. 干预设备检测由 Wrapper 负责（VRWrapper / SpaceMouseWrapper 等）
2. Actor 只管调用 policy.act() 和 env.step()
3. 通过 info 获取干预信息
4. 模型通过 PolicyAdapter 接口接入

负责:
1. 在环境中执行策略
2. 从 info 读取干预信息（由 Wrapper 提供）
3. 使用奖励分类器预测奖励（可选）
4. 发送 transitions 到 Learner
5. 同步最新权重
"""
from typing import Dict, Any, Optional, Tuple, List, Union, Protocol
from dataclasses import dataclass, field
import time
import numpy as np
import torch

from core.runtime.base_loop import BaseLoop
from core.interfaces import EnvInterface


# ============ 协议定义（用于类型提示，不强制继承） ============

class PolicyAdapterProtocol(Protocol):
    """策略适配器协议"""
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any: ...
    def get_weights(self) -> Dict[str, torch.Tensor]: ...
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None: ...


class ActorClientProtocol(Protocol):
    """Actor 客户端协议"""
    def send_transitions(self, transitions: List[Dict[str, Any]]) -> None: ...
    def recv_weights(self, block: bool = True, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]: ...


class RewardClassifierProtocol(Protocol):
    """奖励分类器协议"""
    def predict_reward(self, images: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor: ...


# ============ 配置 ============

@dataclass
class HILActorConfig:
    """HIL Actor 配置"""
    # 奖励分类器
    use_reward_classifier: bool = False
    reward_classifier_threshold: float = 0.5
    camera_keys: List[str] = field(default_factory=lambda: ["images.front", "images.side"])
    
    # 同步
    weight_sync_freq: int = 1  # 每 N 步尝试同步权重
    transition_batch_size: int = 1  # 每次发送的 transition 数量
    
    # 策略
    deterministic: bool = False
    
    # Episode
    max_episode_steps: int = 200
    
    # 是否需要等待初始权重（对于预训练模型可设为 False）
    require_initial_weights: bool = True


# ============ HIL Actor Loop ============

class HILActorLoop(BaseLoop):
    """
    Human-in-the-Loop Actor 循环
    
    设计特点：
    1. 干预检测交给 Wrapper（VRWrapper/SpaceMouseWrapper）
    2. Actor 始终调用 policy.act()，始终传给 env.step()
    3. Wrapper 决定实际执行哪个动作，通过 info 返回
    4. 模型通过 PolicyAdapter 解耦
    
    使用示例：
        # 1. 用 Wrapper 包装环境
        from env.wrappers import VRWrapper
        base_env = H1RobotEnv()
        env = VRWrapper(base_env, vr_device="quest")
        
        # 2. 创建 Actor
        adapter = StandardPolicyAdapter(sac_policy)
        actor = HILActorLoop(adapter, env, client, config)
        
        # 3. 运行
        while not done:
            info = actor.step()
            # Wrapper 已经处理了干预，info 中包含干预信息
    """
    
    def __init__(
        self,
        policy_adapter: PolicyAdapterProtocol,
        env: EnvInterface,
        actor_client: ActorClientProtocol,
        config: HILActorConfig,
        reward_classifier: Optional[RewardClassifierProtocol] = None,
    ):
        super().__init__()
        
        self.policy = policy_adapter
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
        self._weight_sync_count = 0
        
        # Transition 缓冲
        self._transition_buffer: List[Dict[str, Any]] = []
        
        # 初始权重同步状态
        self._initial_weights_synced = not config.require_initial_weights
    
    def wait_for_initial_weights(self, timeout: float = 60.0) -> bool:
        """
        等待 Learner 发送初始权重
        
        Args:
            timeout: 超时时间（秒）
        Returns:
            是否成功同步
        """
        if not self.config.require_initial_weights:
            self._initial_weights_synced = True
            return True
        
        print("[HIL-Actor] Waiting for initial weights...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            weights = self.client.recv_weights(block=True, timeout=5.0)
            if weights is not None:
                self.policy.load_weights(weights)
                self._initial_weights_synced = True
                self._weight_sync_count += 1
                print("[HIL-Actor] Initial weights received!")
                return True
        
        print("[HIL-Actor] Timeout waiting for initial weights")
        return False
    
    def skip_initial_weights(self) -> None:
        """跳过初始权重等待（用于已有预训练权重的情况）"""
        self._initial_weights_synced = True
        print("[HIL-Actor] Skipped initial weight sync (using existing weights)")
    
    def step(self) -> Dict[str, Any]:
        """
        执行单步 Actor 逻辑
        
        核心流程：
        1. 总是调用 policy.act() 获取策略动作
        2. 总是将策略动作传给 env.step()
        3. Wrapper 在内部决定实际执行哪个动作
        4. 从 info 读取干预信息
        """
        # 0. 检查初始化
        if not self._initial_weights_synced:
            raise RuntimeError("Must call wait_for_initial_weights() or skip_initial_weights() first")
        
        # 1. 尝试同步最新权重（非阻塞）
        if self._step_count % self.config.weight_sync_freq == 0:
            self._try_sync_weights()
        
        # 2. 确保环境已重置
        if self._current_obs is None:
            self._reset_episode()
        
        # 3. 获取策略动作（始终调用策略）
        policy_action = self.policy.act(self._current_obs, deterministic=self.config.deterministic)
        policy_action = np.asarray(policy_action)
        
        # 4. 执行动作（传入策略动作，Wrapper 决定实际执行哪个）
        next_obs, env_reward, terminated, truncated, info = self.env.step(policy_action)
        done = terminated or truncated or (self._episode_step >= self.config.max_episode_steps)
        
        # 5. 从 info 读取干预信息（由 Wrapper 提供）
        is_intervention = info.get("is_intervention", False)
        actual_action = info.get("intervene_action", policy_action) if is_intervention else policy_action
        
        # 6. 计算奖励
        reward = self._compute_reward(env_reward, self._current_obs, next_obs, info)
        
        # 7. 构建并缓冲 transition
        transition = self._build_transition(
            obs=self._current_obs,
            action=actual_action,  # 使用实际执行的动作
            policy_action=policy_action,  # 也保存策略动作（用于分析）
            reward=reward,
            next_obs=next_obs,
            done=done,
            is_intervention=is_intervention,
            info=info,
        )
        self._buffer_transition(transition)
        
        # 8. 更新统计
        self._episode_reward += reward
        self._episode_step += 1
        self._total_actions += 1
        if is_intervention:
            self._intervention_count += 1
        
        # 9. 构建返回信息
        step_info = {
            "reward": reward,
            "done": done,
            "is_intervention": is_intervention,
            "episode_step": self._episode_step,
        }
        
        # 10. Episode 结束处理
        if done:
            step_info.update({
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_step,
                "episode_intervention_rate": self._get_episode_intervention_rate(),
            })
            self._episode_count += 1
            self._reset_episode()
        else:
            self._current_obs = next_obs
            self._current_info = info
        
        return step_info
    
    def _reset_episode(self) -> None:
        """重置 episode 状态"""
        self._current_obs, self._current_info = self.env.reset()
        self._episode_step = 0
        self._episode_reward = 0.0
        
        # 重置策略状态（对于有状态的策略）
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
    
    def _compute_reward(
        self,
        env_reward: float,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """计算奖励（可使用奖励分类器）"""
        if not self.config.use_reward_classifier or self.reward_classifier is None:
            return env_reward
        
        # 收集相机图像
        images = self._collect_images(next_obs)
        if not images:
            return env_reward
        
        # 使用奖励分类器预测
        with torch.no_grad():
            predicted = self.reward_classifier.predict_reward(
                images,
                threshold=self.config.reward_classifier_threshold
            )
        
        return float(predicted)
    
    def _collect_images(self, obs: Dict[str, Any]) -> List[torch.Tensor]:
        """从观测中收集图像"""
        images = []
        for key in self.config.camera_keys:
            img = obs.get(key)
            if img is not None:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # 添加 batch 维度
                images.append(img)
        return images
    
    def _build_transition(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        policy_action: np.ndarray,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
        is_intervention: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建 transition 字典"""
        return {
            "obs": self._extract_state(obs),
            "action": action,
            "policy_action": policy_action,  # 策略原始输出
            "reward": reward,
            "next_obs": self._extract_state(next_obs),
            "done": done,
            "is_intervention": is_intervention,
            "source": "intervention" if is_intervention else "rollout",
        }
    
    def _extract_state(self, obs: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
        """从观测中提取状态向量"""
        if isinstance(obs, np.ndarray):
            return obs
        
        # 尝试常见的键
        for key in ["state", "observation", "proprio", "qpos"]:
            if key in obs:
                val = obs[key]
                return np.asarray(val) if not isinstance(val, np.ndarray) else val
        
        # 如果没有找到，尝试拼接所有数值数据
        arrays = []
        for v in obs.values():
            if isinstance(v, (np.ndarray, list)) and not isinstance(v, str):
                arr = np.asarray(v).flatten()
                if arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                    arrays.append(arr)
        
        if arrays:
            return np.concatenate(arrays)
        
        raise ValueError(f"Cannot extract state from observation: {obs.keys()}")
    
    def _buffer_transition(self, transition: Dict[str, Any]) -> None:
        """缓冲 transition 并在达到批量大小时发送"""
        self._transition_buffer.append(transition)
        
        if len(self._transition_buffer) >= self.config.transition_batch_size:
            self._flush_transitions()
    
    def _flush_transitions(self) -> None:
        """发送所有缓冲的 transitions 到 Learner"""
        if self._transition_buffer:
            self.client.send_transitions(self._transition_buffer)
            self._transition_buffer = []
    
    def _try_sync_weights(self) -> bool:
        """尝试同步最新权重（非阻塞）"""
        weights = self.client.recv_weights(block=False)
        if weights is not None:
            self.policy.load_weights(weights)
            self._weight_sync_count += 1
            return True
        return False
    
    def _get_episode_intervention_rate(self) -> float:
        """获取当前 episode 的干预率"""
        if self._episode_step == 0:
            return 0.0
        # 这里简化计算，实际可以更精确地追踪每个 episode
        return self._intervention_count / max(self._total_actions, 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "episode_count": self._episode_count,
            "total_steps": self._step_count,
            "total_interventions": self._intervention_count,
            "intervention_rate": self._intervention_count / max(self._total_actions, 1),
            "weight_sync_count": self._weight_sync_count,
        }
    
    def stop(self) -> None:
        """停止 Actor 循环"""
        super().stop()
        self._flush_transitions()


# ============ 向后兼容别名 ============
HILSerlActorLoop = HILActorLoop
HILSerlActorConfig = HILActorConfig
