import numpy as np
from env.base_env import BaseEnv

class H1BinocularEnv(BaseEnv):
    """
    双目机器人环境，标准化 RL 环境接口
    """
    def __init__(self, config, reward_fn=None):
        super().__init__(config)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.reward_fn = reward_fn
        # 仿真/占位数据结构，后续可用真实数据填充
        self._current_state = np.zeros(self.state_dim)
        self._target_state = np.zeros(self.state_dim)
        self._step_count = 0
        self.max_steps = getattr(config, 'max_steps', 200)
        # 多模态观测字段
        self.qpos = np.zeros(14)
        self.qvel = np.zeros(14)
        self.torque = np.zeros(14)
        self.eef = np.zeros(14)
        self.images = {}
        self.buttons = np.zeros(4)

    def reset(self, task_id=None):
        self._current_state = np.random.randn(self.state_dim) * 0.1
        self._target_state = np.zeros(self.state_dim)
        self._step_count = 0
        # 初始化多模态观测（可用采集/仿真数据填充）
        self.qpos = np.random.randn(14)
        self.qvel = np.random.randn(14)
        self.torque = np.random.randn(14)
        self.eef = np.random.randn(14)
        self.images = {f'cam_{i}': np.zeros((224,224,3)) for i in range(4)}
        self.buttons = np.zeros(4)
        obs = self._get_obs()
        return {
            'obs': obs,
            'robot_state': self._current_state.copy(),
            'reward': 0.0,
            'done': False,
            'info': {'task_id': task_id}
        }

    def step(self, action):
        self._step_count += 1
        # 状态转移: s' = s + a * scale + noise
        self._current_state += action[:self.state_dim] * 0.1 + np.random.randn(self.state_dim) * 0.01
        # 多模态观测更新（可用采集/仿真数据填充）
        self.qpos += action[:14] * 0.05
        self.qvel = np.random.randn(14)
        self.torque = np.random.randn(14)
        self.eef = np.random.randn(14)
        self.images = {f'cam_{i}': np.random.randint(0,255,(224,224,3),dtype=np.uint8) for i in range(4)}
        self.buttons = np.random.randint(0,2,4)
        dist = np.linalg.norm(self._current_state - self._target_state)
        reward = -dist * 0.1
        success = dist < 0.5
        if success:
            reward = 10.0
        if self.reward_fn:
            reward = self.reward_fn(self._current_state, action, self._target_state)
        done = self._step_count >= self.max_steps or success
        obs = self._get_obs()
        return {
            'obs': obs,
            'robot_state': self._current_state.copy(),
            'reward': reward,
            'done': done,
            'info': {'success': success}
        }

    def _get_obs(self):
        # 多模态观测结构，对齐 real_env.py
        return {
            'state': self._current_state.copy(),
            'qpos': self.qpos.copy(),
            'qvel': self.qvel.copy(),
            'torque': self.torque.copy(),
            'eef': self.eef.copy(),
            'images': self.images.copy(),
            'buttons': self.buttons.copy()
        }
