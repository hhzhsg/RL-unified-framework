"""
VLA-RL DataHub: 支持 HDF5 和专用 Buffer 类型
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Literal, Union
import os
import numpy as np

from .base_buffer import BaseBuffer
from .rollout_buffer import RolloutBuffer
from .intervention_buffer import InterventionBuffer
from .sample_strategy import BaseSampleStrategy, create_strategy
from data import Transition, Episode, Batch, Observation, RobotState

# 可选 HDF5 支持
try:
    from .hdf5_buffer import HDF5DemoBuffer
    HAS_HDF5 = True
except ImportError:
    HDF5DemoBuffer = None
    HAS_HDF5 = False


SourceType = Literal["demo", "rollout", "intervention"]


class DataHub:
    """
    数据中心 V2
    
    针对真实机器人场景优化:
    - Demo: HDF5 lazy loading，支持大规模图像数据
    - Rollout: 内存环形缓冲，FIFO
    - Intervention: 内存 + 异步落盘
    """
    
    def __init__(self,
                 # Demo 配置
                 demo_paths: Optional[List[str]] = None,
                 demo_format: str = "hdf5",  # "hdf5" | "pkl"
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
            demo_paths: Demo 文件路径，支持 glob 模式
            demo_format: Demo 格式 "hdf5" 或 "pkl"
            camera_keys: 要加载的相机 keys
            load_images: 是否加载图像
            rollout_capacity: Rollout buffer 容量
            intervention_capacity: Intervention buffer 容量
            intervention_save_dir: Intervention 落盘目录
            auto_save_intervention: 是否自动保存 intervention
        """
        # Demo Buffer
        if demo_format == "hdf5" and HAS_HDF5:
            self.demo_buffer = HDF5DemoBuffer(
                demo_paths=demo_paths,
                camera_keys=camera_keys,
                load_images=load_images,
            )
        else:
            # 回退到简单 ReplayBuffer (pkl 格式)
            from .simple_replay_buffer import SimpleReplayBuffer
            self.demo_buffer = SimpleReplayBuffer()
            if demo_paths:
                for path in demo_paths:
                    if os.path.exists(path):
                        self.demo_buffer.load(path)
        
        # Rollout Buffer (环形)
        self.rollout_buffer = RolloutBuffer(max_size=rollout_capacity)
        
        # Intervention Buffer (带落盘)
        self.intervention_buffer = InterventionBuffer(
            max_size=intervention_capacity,
            save_dir=intervention_save_dir,
            auto_save=auto_save_intervention,
        )
    
    @property
    def buffers(self) -> Dict[str, BaseBuffer]:
        """获取所有 buffer"""
        return {
            "demo": self.demo_buffer,
            "rollout": self.rollout_buffer,
            "intervention": self.intervention_buffer,
        }
    
    def write(self, data: Union[Episode, Transition], source: SourceType):
        """
        写入数据
        
        Args:
            data: Episode 或 Transition
            source: 数据来源
        """
        if source == "demo":
            raise ValueError("Demo buffer is read-only in V2")
        
        buffer = self._get_buffer(source)
        
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
    
    def _get_buffer(self, source: SourceType) -> BaseBuffer:
        """根据 source 获取对应 buffer"""
        return self.buffers[source]
    
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
        """总 transition 数量"""
        return len(self.demo_buffer) + len(self.rollout_buffer) + len(self.intervention_buffer)
    
    def __bool__(self) -> bool:
        """始终为真"""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "demo": self.demo_buffer.get_statistics(),
            "rollout": self.rollout_buffer.get_statistics(),
            "intervention": self.intervention_buffer.get_statistics(),
            "total_transitions": len(self),
        }
    
    # ========== Intervention 特殊操作 ==========
    
    def start_intervention_episode(self, task_id: str = ""):
        """开始收集 intervention episode"""
        self.intervention_buffer.start_episode(task_id)
    
    def end_intervention_episode(self, success: bool = False):
        """结束 intervention episode"""
        self.intervention_buffer.end_episode(success)
    
    def load_intervention_history(self, load_dir: Optional[str] = None):
        """加载历史 intervention 数据"""
        self.intervention_buffer.load_from_disk(load_dir)
    
    def flush_intervention(self):
        """强制落盘 intervention"""
        self.intervention_buffer.flush()

