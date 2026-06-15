"""Main model network and factory."""

from typing import Any

import torch
import torch.nn as nn

from models.baselines.cnn_baseline import CNNBaseline
from models.baselines.transformer_baseline import TransformerBaseline
from models.components import ConvBlock, ResidualBlock, SpatialAttention, SpectralAttention


class HSINet(nn.Module):
    """Main hyperspectral image classification network."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.spectral_attn = SpectralAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()
        self.backbone = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.spectral_attn(x)
        x = self.spatial_attn(x)
        x = self.backbone(x)
        return self.head(x)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from configuration.

    Supported model names: ``hsi_net``, ``cnn_baseline``, ``transformer_baseline``.
    """
    model_cfg = config["model"]
    name = model_cfg["name"]
    kwargs = {
        "in_channels": model_cfg["in_channels"],
        "num_classes": model_cfg["num_classes"],
    }

    if name == "hsi_net":
        return HSINet(
            **kwargs,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )
    if name == "cnn_baseline":
        return CNNBaseline(**kwargs, hidden_dim=model_cfg.get("hidden_dim", 64))
    if name == "transformer_baseline":
        return TransformerBaseline(
            **kwargs,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )

    raise ValueError(f"Unknown model: {name}")
