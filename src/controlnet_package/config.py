"""Configuration handling for ControlNet."""

import json
from pathlib import Path
from typing import Dict, Any


def validate_config(func):
    """Decorator to validate configuration parameters."""
    def wrapper(config_path: Path) -> Dict[str, Any]:
        config = func(config_path)
        required_keys = [
            "pretrained_model", "controlnet_model", "train_data_root",
            "image_size", "batch_size", "epochs", "lr",
            "ckpt_dir"
        ]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Required configuration key '{key}' not found")
        for path_key in ["train_data_root", "ckpt_dir"]:
            if not Path(config[path_key]).exists():
                Path(config[path_key]).mkdir(parents=True, exist_ok=True)
        return config
    return wrapper


@validate_config
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file with validation.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open('r') as f:
        config = json.load(f)
    return config
