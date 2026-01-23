"""
HIL Trainer

封装 Actor 和 Learner 的启动与协调
支持本地调试和分布式训练
"""
from typing import Dict, Any, Optional

from .hil_actor_loop import HILActorLoop, HILActorConfig
from .hil_learner_loop import HILLearnerLoop, HILLearnerConfig
from core.synchronization.actor_learner import (
    ActorLearnerConfig,
    create_learner_server,
    create_actor_client,
)


class HILTrainer:
    """
    HIL 完整训练器
    
    封装 Actor 和 Learner 的启动与协调
    
    使用示例（本地模式）：
        trainer = HILTrainer(
            actor_adapter=policy_adapter,
            learner_adapter=algorithm_adapter,
            env=env,
            config=config,
            mode="local",
        )
        results = trainer.run_local(num_steps=10000)
    
    使用示例（分布式模式）：
        # Learner 进程
        trainer = HILTrainer(...)
        trainer.run_distributed("learner", num_steps=10000)
        
        # Actor 进程
        trainer = HILTrainer(...)
        trainer.run_distributed("actor", num_steps=10000)
    """
    
    def __init__(
        self,
        actor_adapter,  # PolicyAdapterProtocol
        learner_adapter,  # TrainableAdapterProtocol
        env,  # EnvInterface
        config: Dict[str, Any],
        demo_buffer=None,
        reward_classifier=None,
        mode: str = "local",  # "local" | "grpc"
    ):
        self.actor_adapter = actor_adapter
        self.learner_adapter = learner_adapter
        self.env = env
        self.config = config
        self.demo_buffer = demo_buffer
        self.reward_classifier = reward_classifier
        self.mode = mode
        
        # 创建通信组件
        self.sync_config = ActorLearnerConfig(**config.get("sync", {}))
        self.server = create_learner_server(mode, self.sync_config)
        
        if mode == "local":
            self.client = create_actor_client(mode, server=self.server)
        else:
            self.client = create_actor_client(mode, config=self.sync_config)
    
    def run_local(self, num_steps: int, log_freq: int = 100) -> Dict[str, Any]:
        """
        本地模式运行（单进程，用于调试）
        
        交替执行 Actor 和 Learner 步骤
        """
        # 解析配置
        actor_config = HILActorConfig(**self.config.get("actor", {}))
        learner_config = HILLearnerConfig(**self.config.get("learner", {}))
        
        # 创建循环
        actor_loop = HILActorLoop(
            policy_adapter=self.actor_adapter,
            env=self.env,
            actor_client=self.client,
            config=actor_config,
            reward_classifier=self.reward_classifier,
        )
        
        learner_loop = HILLearnerLoop(
            trainable_adapter=self.learner_adapter,
            learner_server=self.server,
            config=learner_config,
            demo_buffer=self.demo_buffer,
        )
        
        # 启动通信
        self.server.start()
        self.client.connect()
        
        # 发布初始权重
        learner_loop.publish_initial_weights()
        
        # Actor 同步初始权重
        actor_loop.wait_for_initial_weights(timeout=5.0)
        
        # 收集初始数据
        print(f"[HIL-Trainer] Collecting initial data ({learner_config.training_starts} transitions)...")
        for i in range(learner_config.training_starts):
            actor_loop.step()
            if (i + 1) % 100 == 0:
                print(f"[HIL-Trainer] Collected {i + 1}/{learner_config.training_starts}")
        
        learner_loop.start_training()
        
        # 主训练循环
        print(f"[HIL-Trainer] Starting training for {num_steps} steps...")
        for step in range(num_steps):
            # Actor 执行
            actor_info = actor_loop.step()
            
            # Learner 更新
            learner_info = learner_loop.step()
            
            # 日志
            if (step + 1) % log_freq == 0:
                reward = actor_info.get('reward', 0)
                loss = learner_info.get('critic_loss', learner_info.get('loss', 0))
                intv_rate = actor_loop.get_statistics()['intervention_rate']
                print(f"[Step {step + 1}] reward={reward:.3f}, loss={loss:.4f}, intervention_rate={intv_rate:.2%}")
        
        # 清理
        self.server.stop()
        self.client.disconnect()
        
        return {
            "actor_stats": actor_loop.get_statistics(),
            "learner_stats": learner_loop.get_statistics(),
        }
    
    def run_distributed(self, role: str, num_steps: int) -> Dict[str, Any]:
        """
        分布式模式运行
        
        Args:
            role: "actor" | "learner"
            num_steps: 运行步数
        """
        if role == "learner":
            return self._run_learner(num_steps)
        elif role == "actor":
            return self._run_actor(num_steps)
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def _run_learner(self, num_steps: int) -> Dict[str, Any]:
        """运行 Learner 进程"""
        learner_config = HILLearnerConfig(**self.config.get("learner", {}))
        
        learner_loop = HILLearnerLoop(
            trainable_adapter=self.learner_adapter,
            learner_server=self.server,
            config=learner_config,
            demo_buffer=self.demo_buffer,
        )
        
        self.server.start()
        
        # 等待数据
        if not learner_loop.wait_for_minimum_data():
            raise RuntimeError("Failed to collect minimum data")
        
        # 发布初始权重
        learner_loop.publish_initial_weights()
        
        # 训练
        results = learner_loop.run(num_steps, log_freq=100)
        
        self.server.stop()
        return results
    
    def _run_actor(self, num_steps: int) -> Dict[str, Any]:
        """运行 Actor 进程"""
        actor_config = HILActorConfig(**self.config.get("actor", {}))
        
        actor_loop = HILActorLoop(
            policy_adapter=self.actor_adapter,
            env=self.env,
            actor_client=self.client,
            config=actor_config,
            reward_classifier=self.reward_classifier,
        )
        
        self.client.connect()
        
        # 等待初始权重
        if not actor_loop.wait_for_initial_weights():
            raise RuntimeError("Failed to receive initial weights")
        
        # 执行
        results = actor_loop.run(num_steps, log_freq=100)
        
        self.client.disconnect()
        return results


# ============ 向后兼容别名 ============
HILSerlTrainer = HILTrainer
