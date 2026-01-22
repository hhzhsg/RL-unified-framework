"""
能力检查器

校验组件间的兼容性
"""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from ..interfaces import ComponentCapability


@dataclass
class CompatibilityIssue:
    """兼容性问题"""
    severity: str  # "error" | "warning"
    component1: str
    component2: str
    message: str


class CapabilityChecker:
    """
    能力检查器
    
    检查组件间的兼容性
    """
    
    def __init__(self):
        self._capabilities: Dict[str, ComponentCapability] = {}
    
    def register_capability(self, component_name: str, capability: ComponentCapability) -> None:
        """注册组件能力"""
        self._capabilities[component_name] = capability
    
    def check_compatibility(self) -> Tuple[bool, List[CompatibilityIssue]]:
        """
        检查所有组件间的兼容性
        
        Returns:
            is_compatible: 是否兼容
            issues: 问题列表
        """
        issues = []
        
        # 检查Policy和Algorithm的兼容性
        if "policy" in self._capabilities and "algorithm" in self._capabilities:
            policy_cap = self._capabilities["policy"]
            algo_cap = self._capabilities["algorithm"]
            
            # 检查action_space
            if policy_cap.action_space != algo_cap.action_space:
                issues.append(CompatibilityIssue(
                    severity="error",
                    component1="policy",
                    component2="algorithm",
                    message=f"Action space mismatch: policy={policy_cap.action_space}, algorithm={algo_cap.action_space}"
                ))
            
            # 检查target_network
            if algo_cap.requires_target_network and not hasattr(policy_cap, "supports_target"):
                issues.append(CompatibilityIssue(
                    severity="warning",
                    component1="policy",
                    component2="algorithm",
                    message="Algorithm requires target network but policy may not support it"
                ))
        
        # 检查Env和Policy的兼容性
        if "env" in self._capabilities and "policy" in self._capabilities:
            env_cap = self._capabilities["env"]
            policy_cap = self._capabilities["policy"]
            
            # 检查图像支持
            if env_cap.supports_images and not policy_cap.supports_images:
                issues.append(CompatibilityIssue(
                    severity="error",
                    component1="env",
                    component2="policy",
                    message="Environment provides images but policy doesn't support them"
                ))
        
        # 检查Algorithm和Buffer的兼容性
        if "algorithm" in self._capabilities and "buffer" in self._capabilities:
            algo_cap = self._capabilities["algorithm"]
            buffer_cap = self._capabilities["buffer"]
            
            if algo_cap.requires_replay_buffer and not buffer_cap:
                issues.append(CompatibilityIssue(
                    severity="error",
                    component1="algorithm",
                    component2="buffer",
                    message="Algorithm requires replay buffer but none provided"
                ))
        
        is_compatible = not any(i.severity == "error" for i in issues)
        return is_compatible, issues
    
    def print_report(self, issues: List[CompatibilityIssue]) -> None:
        """打印兼容性报告"""
        if not issues:
            print("✓ All components are compatible")
            return
        
        print("Compatibility Issues:")
        for issue in issues:
            icon = "✗" if issue.severity == "error" else "⚠"
            print(f"  {icon} [{issue.severity.upper()}] {issue.component1} <-> {issue.component2}")
            print(f"    {issue.message}")
