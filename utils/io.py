"""IO工具"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
