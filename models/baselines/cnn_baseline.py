"""CNN baseline for hyperspectral image classification."""

import torch
import torch.nn as nn

from models.components import ConvBlock


class CNNBaseline(nn.Module):
    """Simple 3-layer CNN baseline."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            nn.MaxPool2d(2),
            ConvBlock(hidden_dim, hidden_dim * 2),
            nn.MaxPool2d(2),
            ConvBlock(hidden_dim * 2, hidden_dim * 4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
