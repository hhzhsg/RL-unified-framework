"""
IO 工具

文件读写相关功能
"""
import os
import json
from typing import Dict, Any, Optional
import yaml


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str):
    """保存 JSON 文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], path: str):
    """保存 YAML 文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def glob_files(pattern: str) -> list:
    """匹配文件"""
    import glob
    return sorted(glob.glob(pattern))
