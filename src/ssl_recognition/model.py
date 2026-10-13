"""Sequence classifiers over landmark features."""

from __future__ import annotations

import torch
from torch import nn

from .landmarks import FEATURE_DIM


class GRUClassifier(nn.Module):
    """LayerNorm -> bidirectional GRU -> (mean + last) pooling -> linear."""

    def __init__(
        self,
        num_classes: int,
        in_dim: int = FEATURE_DIM,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        h = torch.relu(self.proj(self.norm(x)))
        out, _ = self.gru(h)  # (B, T, 2H)
        pooled = torch.cat([out.mean(dim=1), out[:, -1]], dim=1)  # (B, 4H)
        return self.head(self.drop(pooled))


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    from .features import feature_dim

    if str(cfg.get("model", "gru")).lower() == "transformer":
        from .model_transformer import TransformerClassifier

        return TransformerClassifier(
            num_classes=num_classes,
            in_dim=feature_dim(cfg),
            dim=int(cfg.get("hidden", 128)),
            heads=int(cfg.get("heads", 4)),
            layers=int(cfg.get("layers", 2)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    return GRUClassifier(
        num_classes=num_classes,
        in_dim=feature_dim(cfg),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
