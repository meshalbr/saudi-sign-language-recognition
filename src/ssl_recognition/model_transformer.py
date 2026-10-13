"""Small Transformer encoder classifier over landmark sequences (comparison model).

Selected with ``model: transformer`` in the config. Kept deliberately small so it trains on
the laptop CPU in the same time budget as the GRU.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """LayerNorm -> linear -> [CLS] + positional encoding -> encoder -> CLS head."""

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = PositionalEncoding(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, in_dim)
        h = self.proj(self.norm(x))
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)
        h = self.encoder(self.pos(h))
        return self.head(self.drop(h[:, 0]))
