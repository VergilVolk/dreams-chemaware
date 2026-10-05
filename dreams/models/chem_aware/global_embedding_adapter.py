"""Candidate-free post-adapter for a shared normalized DreaMS embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalResidualEmbeddingAdapter(nn.Module):
    """Identity-initialized residual MLP acting on one 1024-D spectrum vector."""

    def __init__(self, dimension: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if dimension <= 0 or hidden_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid global embedding adapter configuration")
        self.norm = nn.LayerNorm(dimension)
        self.fc1 = nn.Linear(dimension, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dimension)
        self.dropout = float(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise RuntimeError("global embedding adapter expects a matrix")
        delta = self.fc2(
            F.dropout(F.gelu(self.fc1(self.norm(value))), self.dropout, self.training)
        )
        return F.normalize(value + delta, dim=-1)


class RuleConditionedResidualEmbeddingAdapter(nn.Module):
    """Bounded shared map whose route is visible in one clean spectrum.

    ``rule_feature`` is a spectrum-local vector of responses to registered
    fragment/neutral-loss masses.  It contains no candidate identity,
    structure, score, rank, or label.  The zero-initialized output projection
    makes the initial map numerically equivalent to the normalized supplied
    official embedding.
    """

    def __init__(
        self, dimension: int, rule_dimension: int, hidden_dim: int,
        residual_strength: float = 0.25,
    ):
        super().__init__()
        if min(dimension, rule_dimension, hidden_dim) <= 0 or not 0 < residual_strength <= 1:
            raise ValueError("invalid rule-conditioned adapter configuration")
        self.residual_strength = float(residual_strength)
        self.embedding_norm = nn.LayerNorm(dimension)
        self.rule_norm = nn.LayerNorm(rule_dimension)
        self.embedding_down = nn.Linear(dimension, hidden_dim, bias=False)
        self.rule_down = nn.Linear(rule_dimension, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, value: torch.Tensor, rule_feature: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 2 or rule_feature.ndim != 2 or len(value) != len(rule_feature):
            raise RuntimeError("embedding and rule-feature batches must align")
        hidden = F.gelu(
            self.embedding_down(self.embedding_norm(value))
            + self.rule_down(self.rule_norm(rule_feature))
        )
        residual = self.up(hidden)
        residual = residual / torch.linalg.vector_norm(
            residual, dim=-1, keepdim=True,
        ).clamp_min(1.0)
        return F.normalize(value + self.residual_strength * residual, dim=-1)


class SharedEmbeddingPostAdapter(nn.Module):
    """Raw-spectrum deployment wrapper; no structure or candidate input exists."""

    def __init__(self, official_model: nn.Module, adapter: GlobalResidualEmbeddingAdapter):
        super().__init__()
        self.official_model = official_model
        self.adapter = adapter

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.official_model(spectra))
