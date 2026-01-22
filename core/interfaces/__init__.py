"""
Core Interfaces

所有组件必须遵循的抽象协议
"""
from .env_interface import EnvInterface
from .buffer_interface import BufferInterface
from .sampler_interface import SamplerInterface, WeightedSamplerInterface
from .policy_interface import PolicyInterface, ActorInterface, CriticInterface
from .network_interface import NetworkInterface, EncoderInterface, MLPInterface, ImageEncoderInterface
from .transform_interface import TransformInterface, ComposableTransform, ReversibleTransform
from .algorithm_interface import AlgorithmInterface, OffPolicyAlgorithmInterface, OnPolicyAlgorithmInterface
from .system_interface import SystemInterface, LoopInterface, SyncInterface, ComponentCapability

__all__ = [
    # Env
    "EnvInterface",
    # Buffer
    "BufferInterface",
    # Sampler
    "SamplerInterface",
    "WeightedSamplerInterface",
    # Policy
    "PolicyInterface",
    "ActorInterface",
    "CriticInterface",
    # Network
    "NetworkInterface",
    "EncoderInterface",
    "MLPInterface",
    "ImageEncoderInterface",
    # Transform
    "TransformInterface",
    "ComposableTransform",
    "ReversibleTransform",
    # Algorithm
    "AlgorithmInterface",
    "OffPolicyAlgorithmInterface",
    "OnPolicyAlgorithmInterface",
    # System
    "SystemInterface",
    "LoopInterface",
    "SyncInterface",
    "ComponentCapability",
]
