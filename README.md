
## 目录结构

```
RL-unified-framework/
├── algorithm/                  # 算法模块
│   ├── base_algorithm.py      # BaseAlgorithm 基类
│   ├── bc.py                  # Behavior Cloning
│   ├── sac.py                 # Soft Actor-Critic
│   └── td3_bc.py              # TD3+BC 离线算法
│
├── buffer/                     # 数据缓冲模块
│   ├── base_buffer.py         # BaseBuffer 基类
│   ├── data_hub.py            # DataHub 统一数据管理
│   ├── hdf5_buffer.py         # Demo 数据只读缓冲
│   ├── rollout_buffer.py      # Rollout FIFO 缓冲
│   └── sample_strategy.py     # 采样策略 (DemoOnly/Mixed)
│
├── core/                       # 核心训练循环
│   ├── training_loop.py       # 训练循环
│   ├── inference_loop.py      # 推理循环
│   ├── stage.py               # 多阶段训练管理
│   └── weight_sync.py         # 权重同步
│
├── data/                       # 数据类型定义
│   └── types.py               # Transition/Batch/Action/EnvOutput
│
├── env/                        # 环境模块
│   ├── base_env.py            # BaseEnv 基类
│   └── dummy_env.py           # DummyEnv 测试环境
│
├── model/                      # 模型模块
│   ├── base_policy.py         # BasePolicy 基类
│   ├── mlp_policy.py          # MLP 策略网络
│   ├── q_network.py           # Q/V 网络
│   └── model_group.py         # 模型管理器
│
├── reward/                     # 奖励模块
│   ├── base_reward.py         # BaseReward 基类
│   ├── env_reward.py          # 环境奖励
│   ├── shaped_reward.py       # 奖励塑形
│   └── composite_reward.py    # 奖励组合
│
├── logger/                     # 日志模块
│   ├── base_logger.py         # BaseLogger 基类
│   ├── console_logger.py      # 控制台日志
│   └── tensorboard_logger.py  # TensorBoard 集成
│
├── config/                     # 配置模块
│   ├── config.py              # 配置类定义
│   └── train_config.yaml      # 训练配置文件
│
└── scripts/                    # 训练脚本
    ├── train.py               # 离线训练
    ├── train_online.py        # 在线训练
    └── train_two_stage_example.py  # 双阶段训练示例
```

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/hhzhsg/RL-unified-framework.git
cd RL-unified-framework

# 2. 创建环境
conda create -n vla_rl python=3.10
conda activate vla_rl

# 3. 安装依赖
pip install -r requirements.txt
```

**依赖**: `torch>=2.0.0`, `numpy>=1.24.0`, `h5py>=3.8.0`, `pyyaml>=6.0`

---

## 使用方式

### 1. 离线训练（BC / TD3+BC）

```bash
# 行为克隆
python scripts/train.py --config config/train_config.yaml --name offline_bc

# TD3+BC
python scripts/train.py --config config/train_config.yaml --name offline_td3bc
```

### 2. 在线训练（SAC）

```bash
python scripts/train_online.py --config config/train_config.yaml --name online_sac_dummy --warmup 1000
```

### 3. 双阶段训练（Offline → Online）

```bash
python scripts/train_two_stage_example.py
```

---

## 搭建算法：双阶段 Online RL 实战

基于 **scripts/train_two_stage_example.py**，展示如何通过继承子类搭建 Offline TD3+BC → Online SAC 训练流程。

### 整体架构

```
Stage 1 (Offline TD3+BC)          Stage 2 (Online SAC)
        ↓                                 ↓
   ┌─────────┐                       ┌─────────┐
   │ DummyEnv│ 生成 Demo              │ DummyEnv│ 真实交互
   └────┬────┘                       └────┬────┘
        │                                 │
        ▼                                 ▼
 ┌─────────────┐                   ┌─────────────┐
 │SimpleBuffer │ Demo Only          │RolloutBuffer│ Online 数据
 │   (Demo)    │                   │  + Demo     │
 └──────┬──────┘                   └──────┬──────┘
        │                                 │
        ▼                                 ▼
 ┌─────────────┐                   ┌─────────────┐
 │DemoOnlyStrat│ 只采样 Demo        │MixedStrategy│ Demo 25% + Rollout 75%
 └──────┬──────┘                   └──────┬──────┘
        │                                 │
        ▼                                 ▼
 ┌─────────────┐                   ┌─────────────┐
 │   TD3+BC    │ Offline 训练      │     SAC     │ Online 训练
 │  Algorithm  │                   │  Algorithm  │
 └──────┬──────┘                   └──────┬──────┘
        │                                 │
        ▼                                 ▼
 ┌─────────────────────┐           ┌─────────────────────┐
 │MLPGaussianPolicy    │ 权重继承  │MLPGaussianPolicy    │
 │QNetwork × 4         │─────────▶│QNetwork × 4         │
 └─────────────────────┘           └─────────────────────┘
```

---

## 数据流转：模块间接口详解

### Stage 1: Offline TD3+BC (Demo Only)

#### 步骤 1: 生成 Demo 数据

**涉及模块**: `DummyEnv` (继承 BaseEnv) → `SimpleBuffer` (继承 BaseBuffer)

```python
# env/dummy_env.py
class DummyEnv(BaseEnv):
    def reset(self) -> EnvOutput:
        # 返回: EnvOutput(obs, robot_state, reward=0, done=False)
        
    def step(self, action: Action) -> EnvOutput:
        # 输入: Action(data=np.array)
        # 返回: EnvOutput(obs', robot_state', reward, done)
```

**接口**: `EnvOutput` → `Transition`

```python
# scripts/train_two_stage_example.py: generate_fake_demo()
for ep in range(num_episodes):
    env_output = env.reset()  # EnvOutput
    while not done:
        action = expert_policy(env_output.robot_state)  # Action
        next_env_output = env.step(action)              # EnvOutput
        
        # 构造 Transition
        transition = Transition(
            obs=env_output.obs,
            robot_state=env_output.robot_state,
            action=action,
            reward=next_env_output.reward,
            next_obs=next_env_output.obs,
            next_robot_state=next_env_output.robot_state,
            done=next_env_output.done,
            source="demo"  # 标记数据来源
        )
        
        # 写入 Demo Buffer
        data_hub.demo_buffer.add_transition(transition)
```

**数据类型转换**:
```
BaseEnv.step() → EnvOutput
                    ↓
         构造 Transition (source="demo")
                    ↓
         BaseBuffer.add_transition()
```

---

#### 步骤 2: 采样训练数据

**涉及模块**: `DataHub` + `DemoOnlyStrategy` (继承 BaseSampleStrategy) → `Batch`

```python
# buffer/sample_strategy.py
class DemoOnlyStrategy(BaseSampleStrategy):
    def sample(self, buffers: Dict[str, BaseBuffer], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        return demo_buffer.sample_transitions(batch_size)
```

**接口**: `List[Transition]` → `Batch`

```python
# buffer/data_hub.py
def sample(self, batch_size: int, strategy: str) -> Batch:
    strategy_obj = create_strategy(strategy)  # DemoOnlyStrategy
    transitions = strategy_obj.sample(self.buffers, batch_size)  # List[Transition]
    return Batch.from_transitions(transitions)  # Batch
```

**Batch 数据结构**:
```python
@dataclass
class Batch:
    robot_state: Tensor    # [B, state_dim]
    action: Tensor         # [B, action_dim]
    reward: Tensor         # [B]
    next_robot_state: Tensor  # [B, state_dim]
    done: Tensor           # [B]
    # obs: Optional[Dict[str, Tensor]]  # 可选的图像数据
```

**数据类型转换**:
```
BaseSampleStrategy.sample() → List[Transition]
                                    ↓
                Batch.from_transitions() (数据堆叠)
                                    ↓
                    Batch (tensor 格式)
```

---

#### 步骤 3: 算法训练

**涉及模块**: `TD3_BC` (继承 BaseAlgorithm) + `ModelGroup`

```python
# algorithm/td3_bc.py
class TD3_BC(BaseAlgorithm):
    def train_step(self, batch: Batch) -> Dict[str, float]:
        # 输入: Batch (已转为 GPU tensor)
        
        # 1. 从 ModelGroup 获取模型
        policy = self.model_group.get("policy")       # MLPGaussianPolicy
        q1 = self.model_group.get("q1")               # QNetwork
        q2 = self.model_group.get("q2")               # QNetwork
        target_q1 = self.model_group.get("target_q1") # QNetwork (冻结)
        
        # 2. 计算 Q-loss
        current_q1 = q1(batch.robot_state, batch.action)
        target_q = compute_target(target_q1, batch)
        q_loss = F.mse_loss(current_q1, target_q)
        
        # 3. 计算 Policy-loss (BC + TD3)
        pred_action = policy.act(batch.robot_state)
        bc_loss = F.mse_loss(pred_action, batch.action)  # 行为克隆
        q_value = q1(batch.robot_state, pred_action)
        policy_loss = bc_loss - 0.1 * q_value.mean()  # TD3 部分
        
        # 4. 反向传播
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 5. 软更新 Target 网络
        self._soft_update_target(tau=0.005)
        
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}
```

**接口**: `Batch` → `Dict[str, float]` (metrics)

**数据类型转换**:
```
TrainingLoop.run()
    ↓
batch = data_hub.sample(...)  # Batch
    ↓
batch = batch.to(device)      # 转移到 GPU
    ↓
metrics = algorithm.train_step(batch)  # Dict[str, float]
    ↓
logger.log(metrics)
```

---

### Stage 2: Online SAC (Demo + Rollout Mixed)

#### 步骤 1: 环境交互（推理循环）

**涉及模块**: `MLPGaussianPolicy` (继承 BasePolicy) → `DummyEnv` → `RolloutBuffer`

```python
# model/mlp_policy.py
class MLPGaussianPolicy(BasePolicy):
    def sample(self, obs: Dict, robot_state: Tensor) -> Tuple[Tensor, Tensor]:
        # 输入: robot_state [1, state_dim]
        # 输出: (action [1, action_dim], log_prob [1])
        mean, log_std = self.network(robot_state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()  # 重参数化采样
        log_prob = dist.log_prob(action).sum(-1)
        return torch.tanh(action), log_prob  # 限制到 [-1, 1]
```

**推理交互流程**:

```python
# scripts/train_two_stage_example.py: run_stage2_online()
inference_policy = copy.deepcopy(policy)
inference_policy.eval()

env_output = env.reset()

for step in range(max_steps):
    # 1. 推理得到动作
    with torch.no_grad():
        state_tensor = torch.FloatTensor(env_output.robot_state.raw_state).unsqueeze(0)
        action_data, _ = inference_policy.sample({}, state_tensor)  # Tensor [1, action_dim]
        action_data = action_data.squeeze(0).numpy()  # numpy [action_dim]
    
    action = Action(data=action_data)
    
    # 2. 执行动作
    prev_robot_state = env_output.robot_state
    env_output = env.step(action)  # EnvOutput
    
    # 3. 构造 Transition
    transition = Transition(
        robot_state=prev_robot_state,
        action=action,
        reward=env_output.reward,
        next_robot_state=env_output.robot_state,
        done=env_output.done,
        source="rollout"  # 标记为在线数据
    )
    
    # 4. 写入 Rollout Buffer
    data_hub.write(transition, source="rollout")
```

**接口**: `BasePolicy.sample()` → `Action` → `EnvOutput` → `Transition`

**数据类型转换**:
```
BasePolicy.sample() → Tensor (action)
                        ↓
            转换为 Action(data=numpy)
                        ↓
            BaseEnv.step(Action) → EnvOutput
                        ↓
            构造 Transition (source="rollout")
                        ↓
            RolloutBuffer.add_transition()
```

---

#### 步骤 2: 混合采样（Demo + Rollout）

**涉及模块**: `MixedStrategy` (继承 BaseSampleStrategy)

```python
# buffer/sample_strategy.py
class MixedStrategy(BaseSampleStrategy):
    def __init__(self, demo_ratio: float = 0.25):
        self.demo_ratio = demo_ratio
        self.rollout_ratio = 1.0 - demo_ratio
    
    def sample(self, buffers: Dict[str, BaseBuffer], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        rollout_buffer = buffers.get("rollout")
        
        # 关键逻辑: Rollout 为空时自动退化为纯 Demo
        if len(rollout_buffer) == 0:
            return demo_buffer.sample_transitions(batch_size)
        
        # 混合采样
        demo_size = int(batch_size * self.demo_ratio)      # 25% Demo
        rollout_size = batch_size - demo_size              # 75% Rollout
        
        transitions = []
        transitions.extend(demo_buffer.sample_transitions(demo_size))
        transitions.extend(rollout_buffer.sample_transitions(rollout_size))
        return transitions
```

**接口**: `Dict[str, BaseBuffer]` → `List[Transition]`

**数据流转过程**:
```
初期 (Rollout=0):
    MixedStrategy.sample() → 自动退化为纯 Demo
    返回 64 条 Demo Transition

中期 (Rollout 增长):
    MixedStrategy.sample() → 混合采样
    返回 16 条 Demo + 48 条 Rollout

后期 (Rollout 充足):
    MixedStrategy.sample() → 稳定混合比例
    返回 16 条 Demo (25%) + 48 条 Rollout (75%)
```

---

#### 步骤 3: SAC 训练

**涉及模块**: `SAC` (继承 BaseAlgorithm)

```python
# algorithm/sac.py
class SAC(BaseAlgorithm):
    def train_step(self, batch: Batch) -> Dict[str, float]:
        # 输入: Batch (混合了 Demo 和 Rollout 数据)
        
        # 1. 计算 Q-loss
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample({}, batch.next_robot_state)
            target_q1 = self.target_q1(batch.next_robot_state, next_action)
            target_q2 = self.target_q2(batch.next_robot_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = batch.reward + self.gamma * (1 - batch.done) * target_q
        
        current_q1 = self.q1(batch.robot_state, batch.action)
        current_q2 = self.q2(batch.robot_state, batch.action)
        q_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        # 2. 计算 Policy-loss (带熵正则)
        new_action, log_prob = self.policy.sample({}, batch.robot_state)
        q_value = torch.min(
            self.q1(batch.robot_state, new_action),
            self.q2(batch.robot_state, new_action)
        )
        policy_loss = (self.alpha * log_prob - q_value).mean()
        
        # 3. 反向传播
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 4. 软更新 Target
        self._soft_update_target(tau=0.005)
        
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}
```

**接口**: `Batch` → `Dict[str, float]`

**关键特性**: SAC 不关心数据来源（Demo 或 Rollout），统一处理混合 Batch。

---

#### 步骤 4: 权重同步（训练 → 推理）

**涉及模块**: `SharedMemorySync` (继承 BaseWeightSync)

```python
# core/weight_sync.py
class SharedMemorySync(BaseWeightSync):
    def push(self, state_dict: Dict[str, Tensor], version: int):
        # 训练进程调用: 将权重推送到共享内存
        
    def pull(self) -> Tuple[Optional[Dict], int]:
        # 推理进程调用: 从共享内存拉取最新权重
```

**权重同步流程**:

```python
# 训练循环中
for step in range(max_steps):
    # ... 环境交互 ...
    
    # 训练
    batch = data_hub.sample(batch_size=64, strategy="mixed", demo_ratio=0.25)
    metrics = algorithm.train_step(batch)
    
    # 每 10 步同步一次权重
    if step % 10 == 0:
        inference_policy.load_state_dict(policy.state_dict())
```

**接口**: `state_dict` (Dict) → 推理 Policy 更新

**数据类型转换**:
```
训练 Policy.state_dict() → Dict[str, Tensor]
                            ↓
    inference_policy.load_state_dict(...)
                            ↓
    推理 Policy 权重更新 → 下一轮推理使用新策略
```

---

## 完整数据流转总结

### Stage 1: Offline TD3+BC

```
DummyEnv.reset()
    → EnvOutput
        → 构造 Transition (source="demo")
            → SimpleBuffer.add_transition()
                → DemoOnlyStrategy.sample()
                    → List[Transition]
                        → Batch.from_transitions()
                            → Batch (GPU)
                                → TD3_BC.train_step()
                                    → ModelGroup (Policy, Q1, Q2, Target Q)
                                        → 梯度更新
                                            → 权重保存在 ModelGroup
```

### Stage 2: Online SAC

```
推理循环:
    MLPGaussianPolicy.sample()
        → Tensor (action)
            → Action(data=numpy)
                → DummyEnv.step()
                    → EnvOutput
                        → Transition (source="rollout")
                            → RolloutBuffer.add_transition()

训练循环:
    MixedStrategy.sample()
        → Demo Buffer (25%) + Rollout Buffer (75%)
            → List[Transition]
                → Batch (GPU)
                    → SAC.train_step()
                        → ModelGroup (继承 Stage 1 权重)
                            → 梯度更新
                                → 权重同步
                                    → inference_policy.load_state_dict()
```

---

## 关键接口说明

### 1. BaseEnv 接口

```python
class BaseEnv(ABC):
    @abstractmethod
    def reset(self, task_id=None) -> EnvOutput:
        """返回初始观测"""
        
    @abstractmethod
    def step(self, action: Action) -> EnvOutput:
        """执行动作，返回下一状态"""
```

**输入输出**:
- 输入: `Action(data=np.array)`
- 输出: `EnvOutput(obs, robot_state, reward, done, info)`

---

### 2. BaseBuffer 接口

```python
class BaseBuffer(ABC):
    @abstractmethod
    def add_transition(self, transition: Transition):
        """添加单条 transition"""
        
    @abstractmethod
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样 batch_size 条数据"""
```

**输入输出**:
- 输入: `Transition` 或 `Episode`
- 输出: `List[Transition]`

---

### 3. BaseSampleStrategy 接口

```python
class BaseSampleStrategy(ABC):
    @abstractmethod
    def sample(self, buffers: Dict[str, BaseBuffer], batch_size: int) -> List[Transition]:
        """从多个 buffer 中按策略采样"""
```

**输入输出**:
- 输入: `Dict[str, BaseBuffer]` (demo, rollout, intervention)
- 输出: `List[Transition]`

---

### 4. BaseAlgorithm 接口

```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """单步训练，返回 metrics"""
```

**输入输出**:
- 输入: `Batch` (GPU tensor)
- 输出: `Dict[str, float]` (loss 和其他指标)

---

### 5. BasePolicy 接口

```python
class BasePolicy(ABC):
    @abstractmethod
    def forward(self, obs: Dict, robot_state: Tensor) -> Tensor:
        """前向传播，返回动作分布参数"""
        
    @abstractmethod
    def act(self, obs: Dict, robot_state: Tensor, deterministic: bool = False) -> Action:
        """采样动作"""
```

**输入输出**:
- 输入: `robot_state: Tensor [B, state_dim]`
- 输出: `Action(data=np.array)` 或 `Tensor`

---

## 扩展到大模型训练（Physical Intelligence π₀ 为例）

Physical Intelligence 的 π₀ 模型使用 **VLA (Vision-Language-Action)** 架构，结合视觉、语言和动作预测。在本框架中可以这样实现：

### 1. 实现 VLA Policy

```python
# model/vla_policy.py
class VLAPolicy(BasePolicy):
    def __init__(self, vision_encoder, language_encoder, action_decoder):
        self.vision_encoder = vision_encoder      # 预训练的视觉编码器 (如 CLIP)
        self.language_encoder = language_encoder  # 预训练的语言编码器 (如 T5)
        self.action_decoder = action_decoder      # 动作解码器 (Transformer)
    
    def forward(self, obs: Dict, robot_state: Tensor) -> Tensor:
        # 1. 编码视觉输入
        image_embeds = self.vision_encoder(obs["images"])  # [B, D_vision]
        
        # 2. 编码语言指令
        text_embeds = self.language_encoder(obs["language"])  # [B, D_text]
        
        # 3. 拼接状态
        state_embeds = torch.cat([image_embeds, text_embeds, robot_state], dim=-1)
        
        # 4. 解码动作
        action_params = self.action_decoder(state_embeds)  # [B, action_dim]
        return action_params
    
    def act(self, obs: Dict, robot_state: Tensor, deterministic: bool = False) -> Action:
        action_params = self.forward(obs, robot_state)
        if deterministic:
            action_data = action_params  # 确定性动作
        else:
            # 添加探索噪声
            noise = torch.randn_like(action_params) * 0.1
            action_data = action_params + noise
        return Action(data=action_data.cpu().numpy())
```

**注册到框架**:

```python
# model/__init__.py
from .vla_policy import VLAPolicy
POLICY_REGISTRY["vla"] = VLAPolicy
```

---

### 2. 配置 π₀ 训练

```yaml
# config/train_config.yaml
configs:
  pi0_bc:
    exp_name: "pi0_bc_training"
    device: "cuda"
    
    env:
      name: "real_robot"  # 实际机器人环境
      state_dim: 65       # 关节位置 + 速度
      action_dim: 37      # 关节控制
    
    data:
      type: "hdf5"
      demo_paths:
        - "/data/robot_demos/task1/*.hdf5"
        - "/data/robot_demos/task2/*.hdf5"
      load_images: true    # 加载图像数据
      camera_keys: ["wrist_cam", "third_person_cam"]
    
    model:
      policy_type: "vla"
      vision_encoder: "clip_vitb32"    # CLIP ViT-B/32
      language_encoder: "t5_base"      # T5-base
      action_decoder: "transformer"    # Transformer decoder
      hidden_dims: [1024, 512]
    
    algorithm:
      name: "bc"
      lr: 1.0e-4
      batch_size: 32       # VLA 模型较大，batch size 较小
      weight_decay: 0.01
    
    training:
      stages:
        - name: "bc_pretrain"
          max_steps: 100000
          sample_strategy: "demo_only"
          active_models: ["policy"]
      checkpoint_dir: "./checkpoints/pi0_bc"
```

---

### 3. 三阶段训练流程（模仿 π₀）

```python
# scripts/train_pi0_three_stage.py

# Stage 1: 预训练 VLA (BC on Demo)
config_stage1 = load_config_from_yaml("config/train_config.yaml", "pi0_bc")
model_group = create_vla_model_group(config_stage1)
data_hub = DataHub(demo_paths=config_stage1.data.demo_paths, load_images=True)

trainer_stage1 = TrainingLoop(
    model_group=model_group,
    data_hub=data_hub,
    config=config_stage1.training,
    algo_config=config_stage1.algorithm
)
trainer_stage1.run()  # 训练 100k 步

# Stage 2: 价值函数训练 (VF Regression)
vf_network = VNetwork(state_dim=65, hidden_dims=[512, 512])
model_group.add("vf", vf_network, frozen=False)
model_group.freeze("policy")  # 冻结 policy

algo_config_stage2 = AlgorithmConfig(name="vf_regression", lr=3e-4)
trainer_stage2 = TrainingLoop(
    model_group=model_group,
    data_hub=data_hub,
    config=config_stage1.training,
    algo_config=algo_config_stage2
)
trainer_stage2.run()

# Stage 3: 策略微调 (AWR with VF)
model_group.unfreeze("policy")
model_group.freeze("vf")  # 冻结价值函数

algo_config_stage3 = AlgorithmConfig(name="awr", lr=1e-4)
trainer_stage3 = TrainingLoop(
    model_group=model_group,
    data_hub=data_hub,
    config=config_stage1.training,
    algo_config=algo_config_stage3
)
trainer_stage3.run()
```

---

### 4. 数据流转（VLA 场景）

```
HDF5 文件 (包含图像 + 语言 + 动作)
    ↓
HDF5DemoBuffer.sample_transitions()
    ↓
List[Transition] (每个 Transition 包含 obs["images"], obs["language"])
    ↓
Batch.from_transitions()
    ↓
Batch (GPU tensor):
    - obs["images"]: [B, C, H, W]
    - obs["language"]: [B, max_seq_len]
    - robot_state: [B, 65]
    - action: [B, 37]
    ↓
VLAPolicy.forward()
    - vision_encoder(images) → [B, 512]
    - language_encoder(text) → [B, 768]
    - cat([vision, language, state]) → [B, 1345]
    - action_decoder → [B, 37]
    ↓
BC.train_step()
    - loss = MSE(predicted_action, batch.action)
    - backward() → 更新 VLA 权重
```

---

### 5. 与本框架的适配点

| π₀ 组件 | 本框架对应模块 | 实现方式 |
|---------|---------------|----------|
| VLA 模型 | `VLAPolicy` (继承 BasePolicy) | 新增 vla_policy.py |
| Vision Encoder | `vision_encoder` (CLIP) | 预训练模型加载 |
| Language Encoder | `language_encoder` (T5) | 预训练模型加载 |
| Action Decoder | `action_decoder` (Transformer) | 自定义 Transformer |
| 行为克隆 | `BC` (已实现) | 直接复用 |
| 价值函数训练 | `VFRegression` (新增) | 继承 BaseAlgorithm |
| AWR 算法 | `AWR` (新增) | 继承 BaseAlgorithm |
| HDF5 数据 | `HDF5DemoBuffer` (已实现) | 支持图像 lazy load |
| 多阶段训练 | `Stage` (已实现) | 直接复用 |

**新增算法示例**:

```python
# algorithm/vf_regression.py
class VFRegression(BaseAlgorithm):
    def train_step(self, batch: Batch) -> Dict[str, float]:
        vf = self.model_group.get("vf")
        target_value = batch.reward + self.gamma * vf(batch.next_robot_state) * (1 - batch.done)
        pred_value = vf(batch.robot_state)
        loss = F.mse_loss(pred_value, target_value.detach())
        loss.backward()
        self.optimizer.step()
        return {"vf_loss": loss.item()}

# algorithm/awr.py
class AWR(BaseAlgorithm):
    def train_step(self, batch: Batch) -> Dict[str, float]:
        vf = self.model_group.get("vf")
        policy = self.model_group.get("policy")
        
        # 计算优势函数
        with torch.no_grad():
            value = vf(batch.robot_state)
            advantage = batch.reward + self.gamma * vf(batch.next_robot_state) - value
            weight = torch.exp(advantage / self.beta).clamp(max=20.0)
        
        # 加权行为克隆
        pred_action = policy.act(batch.obs, batch.robot_state)
        loss = (weight * F.mse_loss(pred_action, batch.action, reduction='none')).mean()
        loss.backward()
        self.optimizer.step()
        return {"awr_loss": loss.item()}
```
