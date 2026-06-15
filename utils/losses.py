"""Loss functions for the project demo.

This is a stable project utility file. When migrating to a specific paper
project, replace or extend task-specific losses here.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Build loss function from configuration."""
    name = config["loss"]["name"]
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss()
    raise ValueError(f"Unknown loss: {name}")
