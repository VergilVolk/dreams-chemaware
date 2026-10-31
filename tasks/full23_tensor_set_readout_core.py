"""Information-preserving representation readout for Full23.

The previous v1.1 readout projected each 1024-dimensional embedding to 16
dimensions *before* forming query/reference interactions.  It then reduced the
reference axis with mean/max pooling.  This module delays dimensionality
reduction until after the full ``abs(q-r)`` and ``q*r`` tensors have been
formed, retains all registered reference tokens, and lets the four frozen
encoders exchange candidate-level information.

The Full23 score remains an explicit anchor.  This module changes neither an
encoder nor any WSE/P2b/MSG/V2 upstream artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from full23_multireference_readout_core import ENCODER_NAMES, TOP_K


SCHEMA = "full23_anchored_tensor_set_readout_v1"


@dataclass(frozen=True)
class TensorSetSettings:
    encoder_names: tuple[str, ...] = ENCODER_NAMES
    embedding_dimension: int = 1024
    pair_dimension: int = 64
    encoder_dimension: int = 64
    attention_heads: int = 4
    fusion_hidden: int = 128
    top_k: int = TOP_K


class Full23TensorSetReadout(nn.Module):
    """Bounded Full23 residual over full pair tensors and reference sets."""

    def __init__(self, settings: TensorSetSettings, residual_bound: float):
        super().__init__()
        if tuple(settings.encoder_names) != ENCODER_NAMES:
            raise ValueError("encoder registry drifted")
        if settings.embedding_dimension < 1 or settings.top_k != TOP_K:
            raise ValueError("invalid tensor-set dimensions")
        if settings.encoder_dimension % settings.attention_heads:
            raise ValueError("encoder dimension must be divisible by attention heads")
        if not np.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("residual bound must be finite and positive")
        self.settings = settings
        self.residual_bound = float(residual_bound)

        # Crucially, q/r are not projected separately.  Every original
        # dimension can contribute to both difference and multiplicative
        # interactions before the learned bottleneck.
        self.pair_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * settings.embedding_dimension, settings.pair_dimension),
                nn.GELU(),
                nn.LayerNorm(settings.pair_dimension),
            )
            for name in settings.encoder_names
        })
        self.reference_attention = nn.ModuleDict({
            name: nn.Linear(settings.pair_dimension, 1, bias=False)
            for name in settings.encoder_names
        })
        # Attention-weighted mean and standard deviation preserve consensus
        # and dispersion.  Reference count is explicit rather than leaking via
        # max-pooling opportunity.
        self.encoder_summaries = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * settings.pair_dimension + 1, settings.encoder_dimension),
                nn.GELU(),
                nn.LayerNorm(settings.encoder_dimension),
            )
            for name in settings.encoder_names
        })
        self.cross_encoder_attention = nn.MultiheadAttention(
            settings.encoder_dimension,
            settings.attention_heads,
            batch_first=True,
        )
        self.cross_encoder_norm = nn.LayerNorm(settings.encoder_dimension)
        self.cross_encoder_ffn = nn.Sequential(
            nn.Linear(settings.encoder_dimension, 2 * settings.encoder_dimension),
            nn.GELU(),
            nn.Linear(2 * settings.encoder_dimension, settings.encoder_dimension),
        )
        self.cross_encoder_ffn_norm = nn.LayerNorm(settings.encoder_dimension)
        self.residual_head = nn.Sequential(
            nn.Linear(len(settings.encoder_names) * settings.encoder_dimension,
                      settings.fusion_hidden),
            nn.GELU(),
            nn.Linear(settings.fusion_hidden, 1),
        )
        # Exact Full23 identity at initialization.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _encoder_token(
        self,
        name: str,
        query: torch.Tensor,
        references: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.pair_encoders[name][0].weight.dtype
        query = F.normalize(query.to(dtype=dtype), dim=-1)
        references = F.normalize(references.to(dtype=dtype), dim=-1)
        expanded = query[:, None, :].expand_as(references)
        interaction = torch.cat((
            torch.abs(expanded - references), expanded * references,
        ), dim=-1)
        pair = self.pair_encoders[name](interaction)
        attention_logits = self.reference_attention[name](pair).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=1)
        mean = torch.sum(attention[:, :, None] * pair, dim=1)
        centered = pair - mean[:, None, :]
        variance = torch.sum(attention[:, :, None] * centered.square(), dim=1)
        deviation = torch.sqrt(variance.clamp_min(1e-8))
        count = mask.sum(dim=1, keepdim=True).to(dtype=dtype) / float(self.settings.top_k)
        return self.encoder_summaries[name](torch.cat((mean, deviation, count), dim=-1))

    def raw_residual(
        self,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        count = query_positions.shape[0]
        encoder_count = len(self.settings.encoder_names)
        if query_positions.shape != (count, encoder_count):
            raise ValueError("query-position axis mismatch")
        expected = (count, encoder_count, self.settings.top_k)
        if reference_positions.shape != expected or reference_mask.shape != expected:
            raise ValueError("reference tensor axis mismatch")
        if not torch.all(reference_mask[:, :, 0]):
            raise ValueError("every encoder/candidate requires one reference")
        if set(banks) != set(self.settings.encoder_names):
            raise ValueError("embedding bank registry drifted")

        encoder_tokens = []
        for encoder_index, name in enumerate(self.settings.encoder_names):
            bank = banks[name]
            query = bank[query_positions[:, encoder_index]]
            references = bank[reference_positions[:, encoder_index]]
            encoder_tokens.append(self._encoder_token(
                name, query, references, reference_mask[:, encoder_index],
            ))
        tokens = torch.stack(encoder_tokens, dim=1)
        attended, _ = self.cross_encoder_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.cross_encoder_norm(tokens + attended)
        tokens = self.cross_encoder_ffn_norm(tokens + self.cross_encoder_ffn(tokens))
        return self.residual_head(tokens.flatten(1)).squeeze(-1)

    def residual(
        self,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raw = self.raw_residual(query_positions, reference_positions, reference_mask, banks)
        return self.residual_bound * torch.tanh(raw / self.residual_bound)

    def forward(
        self,
        anchor: torch.Tensor,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return anchor + self.residual(
            query_positions, reference_positions, reference_mask, banks,
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weighted_anchor_safe_ranking_loss(
    logits: torch.Tensor,
    anchor: torch.Tensor,
    ptr: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    pairwise_margin: float = 0.10,
    pairwise_weight: float = 0.50,
    anchor_correct_multiplier: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Listwise loss plus hardest-negative and asymmetric anchor protection.

    All terms are query-balanced.  Anchor-correct queries receive extra
    hardest-negative weight; no truth-dependent feature is passed to inference.
    """
    per_listwise = []
    per_pairwise = []
    masses = []
    protected = []
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        local_labels = np.asarray(labels[left:right], dtype=bool)
        positives = np.flatnonzero(local_labels)
        if len(positives) != 1 or right - left < 2:
            raise ValueError("each training query needs one positive and one negative")
        positive = left + int(positives[0])
        negative_local = torch.cat((logits[left:positive], logits[positive + 1:right]))
        hardest_negative = torch.max(negative_local)
        listwise = F.cross_entropy(
            logits[left:right][None, :],
            torch.as_tensor([int(positives[0])], device=logits.device),
            reduction="none",
        )[0]
        pairwise = F.softplus(pairwise_margin - (logits[positive] - hardest_negative))
        anchor_local = anchor[left:right]
        anchor_correct = int(torch.argmax(anchor_local).item()) == int(positives[0])
        multiplier = anchor_correct_multiplier if anchor_correct else 1.0
        per_listwise.append(listwise)
        per_pairwise.append(pairwise * multiplier)
        masses.append(float(weights[query]))
        protected.append(float(anchor_correct))
    mass = torch.as_tensor(masses, dtype=logits.dtype, device=logits.device)
    mass = mass / mass.sum().clamp_min(torch.finfo(logits.dtype).eps)
    listwise_loss = torch.sum(torch.stack(per_listwise) * mass)
    pairwise_loss = torch.sum(torch.stack(per_pairwise) * mass)
    return listwise_loss + pairwise_weight * pairwise_loss, {
        "listwise": listwise_loss,
        "hardest_pairwise": pairwise_loss,
        "anchor_correct_fraction": torch.as_tensor(
            np.average(protected, weights=masses), dtype=logits.dtype,
            device=logits.device,
        ),
    }


def model_from_payload(
    payload: dict,
    device: torch.device | str = "cpu",
) -> Full23TensorSetReadout:
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("unsupported tensor-set checkpoint schema")
    raw = dict(payload.get("settings", {}))
    raw["encoder_names"] = tuple(raw.get("encoder_names", ENCODER_NAMES))
    settings = TensorSetSettings(**raw)
    model = Full23TensorSetReadout(settings, float(payload["residual_bound"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
