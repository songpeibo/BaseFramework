"""Configuration loading utilities."""

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_name: str) -> dict[str, Any]:
    """Load and deep-merge base config with a dataset-specific config.

    Args:
        config_name: Name of the config file (e.g. ``paviau.yaml``).

    Returns:
        Merged configuration dictionary.
    """
    base_path = CONFIG_DIR / "base.yaml"
    dataset_path = CONFIG_DIR / config_name

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Config file not found: {dataset_path}")

    with open(base_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    with open(dataset_path, encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f) or {}

    return deep_update(base_config, dataset_config)


def save_config(config: dict[str, Any], save_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
