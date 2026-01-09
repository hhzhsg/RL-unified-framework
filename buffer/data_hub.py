"""
VLA-RL DataHub: 统一数据管理

管理三种数据源:
- Demo: 专家演示数据 (只读)
- Rollout: 策略采集数据 (FIFO)
- Intervention: 人工干预数据 (带持久化)
"""
from typing import Optional, Dict, Any, List, Union
import numpy as np

from .base_buffer import BaseBuffer
from .rollout_buffer import RolloutBuffer
from .sample_strategy import BaseSampleStrategy, create_strategy
from data import Transition, Episode, Batch, Observation, RobotState


class SimpleBuffer(BaseBuffer):
    """简单的内存 Buffer，用于测试"""
    
    def __init__(self, max_size: int = 100000):
        super().__init__(max_size if max_size > 0 else 100000)
        self._transitions: List[Transition] = []
    
    def add_transition(self, transition: Transition):
        if self.max_size > 0 and len(self._transitions) >= self.max_size:
            self._transitions.pop(0)
        self._transitions.append(transition)
    
    def add_episode(self, episode: Episode):
        for t in episode.transitions:
            self.add_transition(t)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        import random
        if len(self._transitions) == 0:
            return []
        return random.choices(self._transitions, k=batch_size)
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return 0
    
    def _get_save_data(self):
        return self._transitions
    
    def _load_from_data(self, data):
        self._transitions = data
    
    def clear(self):
        self._transitions = []


class DataHub:
    """
    数据中心
    
    统一管理 Demo/Rollout/Intervention 三种数据源
    """
    
    def __init__(self,
                 # Demo 配置
                 demo_paths: Optional[List[str]] = None,
                 demo_format: str = "hdf5",
                 camera_keys: Optional[List[str]] = None,
                 load_images: bool = True,
                 # Rollout 配置
                 rollout_capacity: int = 100000,
                 # Intervention 配置
                 intervention_capacity: int = 50000,
                 intervention_save_dir: Optional[str] = None,
                 auto_save_intervention: bool = True):
        """
        Args:
            demo_paths: Demo 文件路径
            demo_format: Demo 格式 "hdf5" 或 "pkl"
            camera_keys: 要加载的相机 keys
            load_images: 是否加载图像
            rollout_capacity: Rollout buffer 容量
            intervention_capacity: Intervention buffer 容量
            intervention_save_dir: Intervention 落盘目录
            auto_save_intervention: 是否自动保存 intervention
        """
        # Demo Buffer
        self.demo_buffer: BaseBuffer = self._create_demo_buffer(
            demo_paths, demo_format, camera_keys, load_images
        )
        
        # Rollout Buffer
        self.rollout_buffer = RolloutBuffer(max_size=rollout_capacity)
        
        # Intervention Buffer (简化版)
        self.intervention_buffer = SimpleBuffer(max_size=intervention_capacity)
    
    def _create_demo_buffer(self, demo_paths, demo_format, camera_keys, load_images) -> BaseBuffer:
        """创建 Demo Buffer"""
        if not demo_paths:
            return SimpleBuffer(max_size=0)
        
        # 尝试使用 HDF5 Buffer
        if demo_format == "hdf5":
            try:
                from .hdf5_buffer import HDF5DemoBuffer
                return HDF5DemoBuffer(
                    demo_paths=demo_paths,
                    camera_keys=camera_keys,
                    load_images=load_images,
                )
            except ImportError:
                print("[Warning] HDF5 not available, using SimpleBuffer")
        
        # 回退到简单 Buffer
        return SimpleBuffer()
    
    @property
    def buffers(self) -> Dict[str, BaseBuffer]:
        """获取所有 buffer"""
        return {
            "demo": self.demo_buffer,
            "rollout": self.rollout_buffer,
            "intervention": self.intervention_buffer,
        }
    
    def write(self, data: Union[Episode, Transition], source: str):
        """
        写入数据
        
        Args:
            data: Episode 或 Transition
            source: 数据来源 ("rollout" | "intervention")
        """
        if source == "demo":
            raise ValueError("Demo buffer is read-only")
        
        buffer = self.buffers[source]
        
        if isinstance(data, Episode):
            for t in data.transitions:
                t.source = source
            buffer.add_episode(data)
        elif isinstance(data, Transition):
            data.source = source
            buffer.add_transition(data)
        else:
            raise ValueError(f"Unknown data type: {type(data)}")
    
    def sample(self, batch_size: int, 
               strategy: Union[str, BaseSampleStrategy] = "demo_only",
               **kwargs) -> Batch:
        """
        采样数据
        
        Args:
            batch_size: 批次大小
            strategy: 采样策略
            **kwargs: 策略参数
        """
        if isinstance(strategy, str):
            strategy = create_strategy(strategy, **kwargs)
        
        transitions = strategy.sample(self.buffers, batch_size)
        
        if len(transitions) == 0:
            raise ValueError("No data available in buffers")
        
        return self._transitions_to_batch(transitions)
    
    def _transitions_to_batch(self, transitions: List[Transition]) -> Batch:
        """将 transitions 转换为 Batch"""
        obs_list = []
        robot_state_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        next_robot_state_list = []
        done_list = []
        source_list = []
        
        for t in transitions:
            obs_list.append(t.obs.to_dict())
            robot_state_list.append(t.robot_state.to_array())
            action_list.append(t.action.data)
            reward_list.append(t.reward)
            next_obs_list.append(t.next_obs.to_dict())
            next_robot_state_list.append(t.next_robot_state.to_array())
            done_list.append(float(t.done))
            source_list.append(t.source)
        
        obs_batch = self._merge_obs_dicts(obs_list)
        next_obs_batch = self._merge_obs_dicts(next_obs_list)
        
        return Batch(
            obs=obs_batch,
            robot_state=np.stack(robot_state_list),
            action=np.stack(action_list),
            reward=np.array(reward_list, dtype=np.float32),
            next_obs=next_obs_batch,
            next_robot_state=np.stack(next_robot_state_list),
            done=np.array(done_list, dtype=np.float32),
            source=source_list,
        )
    
    def _merge_obs_dicts(self, obs_dicts: List[dict]) -> dict:
        """合并观测字典"""
        if len(obs_dicts) == 0:
            return {}
        
        result = {}
        keys = obs_dicts[0].keys()
        
        for key in keys:
            values = [d.get(key) for d in obs_dicts if d.get(key) is not None]
            if len(values) == 0:
                continue
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            else:
                result[key] = values
        
        return result
    
    def __len__(self) -> int:
        return len(self.demo_buffer) + len(self.rollout_buffer) + len(self.intervention_buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "demo": self.demo_buffer.get_statistics(),
            "rollout": self.rollout_buffer.get_statistics(),
            "intervention": self.intervention_buffer.get_statistics(),
            "total_transitions": len(self),
        }
