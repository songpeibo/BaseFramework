"""Inspect merged configuration."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config

REQUIRED_KEYS = [
    ("model", "name"),
    ("experiment", "save_dir"),
    ("experiment", "exp_name"),
]


def validate_config(config: dict) -> None:
    """Ensure critical keys exist after deep merge."""
    missing = []
    for section, key in REQUIRED_KEYS:
        if section not in config or key not in config[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise KeyError(
            "Missing required config keys after merge: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect merged config")
    parser.add_argument("--config", type=str, default="paviau.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    print(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    print("\nValidation passed:")
    print(f"  model.name          = {config['model']['name']}")
    print(f"  experiment.save_dir = {config['experiment']['save_dir']}")
    print(f"  experiment.exp_name = {config['experiment']['exp_name']}")
    print(f"  train.epochs        = {config['train']['epochs']}")


if __name__ == "__main__":
    main()
