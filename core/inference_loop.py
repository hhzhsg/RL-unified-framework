"""
VLA-RL 推理循环

支持:
- 权重热更新
- Action Chunking
- 历史上下文
"""
from typing import Optional, Deque
from collections import deque
import copy
import numpy as np

from .weight_sync import BaseWeightSync
from model import BasePolicy
from env import BaseEnv
from buffer import DataHub
from data import Observation, RobotState, Action, Transition, Episode, EnvOutput
from config import InferenceConfig


class HistoryBuffer:
    """
    历史缓冲区
    维护推理所需的历史上下文
    """
    
    def __init__(self, max_len: int = 10):
        self.max_len = max_len
        self.observations: Deque[Observation] = deque(maxlen=max_len)
        self.robot_states: Deque[RobotState] = deque(maxlen=max_len)
        self.actions: Deque[Action] = deque(maxlen=max_len)
    
    def push(self, obs: Observation, robot_state: RobotState, action: Optional[Action] = None):
        """添加一步"""
        self.observations.append(obs)
        self.robot_states.append(robot_state)
        if action is not None:
            self.actions.append(action)
    
    def get_context(self, window_size: Optional[int] = None) -> dict:
        """获取历史上下文"""
        if window_size is None:
            window_size = self.max_len
        
        obs_list = list(self.observations)[-window_size:]
        state_list = list(self.robot_states)[-window_size:]
        action_list = list(self.actions)[-window_size:]
        
        return {
            "observations": obs_list,
            "robot_states": state_list,
            "actions": action_list,
        }
    
    def clear(self):
        """清空"""
        self.observations.clear()
        self.robot_states.clear()
        self.actions.clear()


class InferenceLoop:
    """
    推理循环
    
    用于在环境中执行策略，收集数据
    """
    
    def __init__(self,
                 policy: BasePolicy,
                 env: BaseEnv,
                 config: InferenceConfig,
                 data_hub: Optional[DataHub] = None,
                 weight_sync: Optional[BaseWeightSync] = None):
        """
        Args:
            policy: 策略模型
            env: 环境
            config: 推理配置
            data_hub: 数据中心 (用于写入 rollout 数据)
            weight_sync: 权重同步器 (用于接收训练权重)
        """
        # 复制策略用于推理
        self.policy = copy.deepcopy(policy)
        self.policy.to(config.device)
        self.policy.eval()
        
        self.env = env
        self.config = config
        self.data_hub = data_hub
        self.weight_sync = weight_sync
        
        # Action Chunking
        self.action_horizon = config.action_horizon
        self.execute_steps = config.execute_steps
        self._action_chunk: Optional[np.ndarray] = None
        self._chunk_index: int = 0
        
        # 历史缓冲区
        self.history_buffer: Optional[HistoryBuffer] = None
        if config.history_len > 1:
            self.history_buffer = HistoryBuffer(max_len=config.history_len)
        
        # 运行状态
        self._running = False
        self._weight_version = 0
        self._episode_count = 0
        self._step_count = 0
    
    def run(self, max_episodes: Optional[int] = None, 
            collect_data: bool = True,
            source: str = "rollout"):
        """
        运行推理循环
        
        Args:
            max_episodes: 最大 episode 数，None 表示无限
            collect_data: 是否收集数据到 data_hub
            source: 数据来源标记 ("rollout" | "intervention")
        """
        self._running = True
        
        while self._running:
            # 运行一个 episode
            episode = self._run_episode(collect_data, source)
            self._episode_count += 1
            
            print(f"[Inference] Episode {self._episode_count} finished, "
                  f"length={len(episode)}, success={episode.success}")
            
            if max_episodes and self._episode_count >= max_episodes:
                break
        
        self._running = False
    
    def _run_episode(self, collect_data: bool, source: str) -> Episode:
        """运行单个 episode"""
        env_output = self.env.reset()
        episode = Episode(task_id=env_output.info.get("task_id", ""))
        
        # 清空历史和 action chunk
        if self.history_buffer:
            self.history_buffer.clear()
        self._action_chunk = None
        self._chunk_index = 0
        
        done = False
        
        while not done and self._running:
            # 检查权重更新
            self._check_weight_update()
            
            # 获取动作
            action = self._get_action(env_output.obs, env_output.robot_state)
            
            # 保存当前状态
            prev_obs = env_output.obs
            prev_robot_state = env_output.robot_state
            
            # 执行动作
            env_output = self.env.step(action)
            done = env_output.done
            self._step_count += 1
            
            # 创建 transition
            transition = Transition(
                obs=prev_obs,
                robot_state=prev_robot_state,
                action=action,
                reward=env_output.reward,
                next_obs=env_output.obs,
                next_robot_state=env_output.robot_state,
                done=done,
                source=source,
            )
            episode.add(transition)
            
            # 写入数据
            if collect_data and self.data_hub is not None:
                self.data_hub.write(transition, source=source)
            
            # 更新历史
            if self.history_buffer:
                self.history_buffer.push(env_output.obs, env_output.robot_state, action)
        
        episode.success = env_output.info.get("success", False)
        
        return episode
    
    def _get_action(self, obs: Observation, robot_state: RobotState) -> Action:
        """获取动作 (支持 Action Chunking)"""
        # 如果有未执行完的 chunk
        if self._action_chunk is not None and self._chunk_index < len(self._action_chunk):
            action_data = self._action_chunk[self._chunk_index]
            self._chunk_index += 1
            
            # 检查是否需要重新预测
            if self._chunk_index >= self.execute_steps:
                self._action_chunk = None
                self._chunk_index = 0
            
            return Action(data=action_data, space=self.policy.action_space)
        
        # 需要重新预测
        action = self.policy.act(obs, robot_state, deterministic=self.config.deterministic)
        
        # 如果是 action chunk
        if self.action_horizon > 1 and len(action.data.shape) > 1:
            self._action_chunk = action.data
            self._chunk_index = 1
            return Action(data=action.data[0], space=action.space)
        
        return action
    
    def _check_weight_update(self):
        """检查是否有新权重"""
        if self.weight_sync is None:
            return
        
        result = self.weight_sync.pull()
        if result is not None:
            state_dict, version = result
            if version > self._weight_version:
                # 只加载 policy 的权重
                if "policy" in state_dict:
                    self.policy.load_state_dict(state_dict["policy"])
                else:
                    self.policy.load_state_dict(state_dict)
                self._weight_version = version
                print(f"[Inference] Updated to weights v{version}")
    
    def stop(self):
        """停止推理"""
        self._running = False
    
    def collect_rollout(self, num_steps: int, source: str = "rollout") -> int:
        """
        收集指定步数的数据
        
        Args:
            num_steps: 要收集的步数
            source: 数据来源标记
            
        Returns:
            实际收集的步数
        """
        collected = 0
        env_output = self.env.reset()
        
        while collected < num_steps:
            # 检查权重更新
            self._check_weight_update()
            
            # 获取动作
            action = self._get_action(env_output.obs, env_output.robot_state)
            
            # 保存当前状态
            prev_obs = env_output.obs
            prev_robot_state = env_output.robot_state
            
            # 执行动作
            env_output = self.env.step(action)
            collected += 1
            
            # 创建 transition
            transition = Transition(
                obs=prev_obs,
                robot_state=prev_robot_state,
                action=action,
                reward=env_output.reward,
                next_obs=env_output.obs,
                next_robot_state=env_output.robot_state,
                done=env_output.done,
                source=source,
            )
            
            # 写入数据
            if self.data_hub is not None:
                self.data_hub.write(transition, source=source)
            
            # 如果 episode 结束，重置环境
            if env_output.done:
                env_output = self.env.reset()
        
        return collected
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        评估策略
        
        Args:
            num_episodes: 评估 episode 数
            
        Returns:
            评估指标
        """
        successes = 0
        total_rewards = []
        episode_lengths = []
        
        for i in range(num_episodes):
            episode = self._run_episode(collect_data=False, source="eval")
            
            if episode.success:
                successes += 1
            
            total_reward = sum(t.reward for t in episode.transitions)
            total_rewards.append(total_reward)
            episode_lengths.append(len(episode))
        
        return {
            "success_rate": successes / num_episodes,
            "avg_reward": np.mean(total_rewards),
            "avg_length": np.mean(episode_lengths),
            "num_episodes": num_episodes,
        }
