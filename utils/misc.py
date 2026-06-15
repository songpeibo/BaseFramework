"""Miscellaneous utilities."""

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> torch.device:
    """Return torch device, falling back to CPU if CUDA unavailable."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_exp_dir(config: dict[str, Any]) -> Path:
    """Build experiment directory path: experiments/<exp_name>."""
    exp_cfg = config["experiment"]
    return Path(exp_cfg["save_dir"]) / exp_cfg["exp_name"]


def ensure_exp_subdirs(exp_dir: str | Path) -> dict[str, Path]:
    """Create standard experiment subdirectories."""
    exp_dir = Path(exp_dir)
    subdirs = {}
    for name in ("checkpoints", "logs", "results", "figures", "tables"):
        path = exp_dir / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_command(save_path: str | Path) -> None:
    """Save the current command line to a text file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    command = f"{sys.executable} {' '.join(sys.argv)}"
    save_path.write_text(command + "\n", encoding="utf-8")


def save_env_info(save_path: str | Path) -> None:
    """Save Python and PyTorch environment information."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"Python: {sys.version.replace(chr(10), ' ')}"]
    try:
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA version: {torch.version.cuda}")
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        lines.append(f"PyTorch: not available ({exc})")

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one JSON record as a line to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    save_path: str | Path,
) -> None:
    """Save model checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
