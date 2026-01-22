"""
Orchestration模块

系统组装与校验
"""
from .system_builder import SystemBuilder, SystemComponents
from .capability_checker import CapabilityChecker, CompatibilityIssue
from .component_registry import (
    ComponentRegistry, 
    REGISTRY,
    register_env,
    register_buffer,
    register_sampler,
    register_policy,
    register_algorithm,
    register_sync,
    register_transform,
)

__all__ = [
    "SystemBuilder",
    "SystemComponents",
    "CapabilityChecker",
    "CompatibilityIssue",
    "ComponentRegistry",
    "REGISTRY",
    "register_env",
    "register_buffer",
    "register_sampler",
    "register_policy",
    "register_algorithm",
    "register_sync",
    "register_transform",
]
