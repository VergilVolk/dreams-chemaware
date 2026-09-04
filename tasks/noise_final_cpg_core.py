"""Core math for counterfactual peak-gated (CPG) noise fine-tuning.

This module deliberately has no dataset-specific action selection.  It defines
the loss-preserving pieces that are unit tested before any GPU job is allowed:
full-candidate molecular aggregation, paired residual targets, an exactly
identity-initialized clean-spectrum peak gate, bounded hierarchical sampling,
and clean-safe gradient projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def molecule_max_scores(pair_scores: torch.Tensor, molecule_ptr: Sequence[int]) -> torch.Tensor:
    """Aggregate spectrum-pair scores with the official per-molecule maximum."""
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if pair_scores.ndim != 1 or ptr.ndim != 1 or len(ptr) < 3:
        raise ValueError("pair scores and molecule pointer have invalid dimensions")
    if ptr[0] != 0 or ptr[-1] != pair_scores.numel() or np.any(np.diff(ptr) < 1):
        raise ValueError("molecule pointer does not span non-empty molecule blocks")
    return torch.stack([
        torch.max(pair_scores[int(left):int(right)])
        for left, right in zip(ptr[:-1], ptr[1:])
    ])


def positive_candidate_margins(molecule_scores: torch.Tensor) -> torch.Tensor:
    """Return positive-minus-each-negative margins; positive is unique and first."""
    if molecule_scores.ndim != 1 or molecule_scores.numel() < 2:
        raise ValueError("at least one positive and one negative molecule are required")
    return molecule_scores[0] - molecule_scores[1:]


def candidate_margin_vector(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    molecule_ptr: Sequence[int],
) -> torch.Tensor:
    """Differentiable full-candidate positive-vs-negative margin vector."""
    if query_embedding.ndim != 1 or candidate_embeddings.ndim != 2:
        raise ValueError("query/candidate embedding dimensions are invalid")
    if candidate_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("query/candidate embedding dimensions disagree")
    pair_scores = candidate_embeddings @ query_embedding
    return positive_candidate_margins(molecule_max_scores(pair_scores, molecule_ptr))


def paired_candidate_residual(
    target_query: torch.Tensor,
    control_queries: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    molecule_ptr: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target-minus-control candidate-margin residual without scalar collapse.

    The function is mathematical rather than a trainable loss.  Callers building
    a frozen teacher use inference tensors; student code receives the resulting
    residual as a fixed target.  One or more controls are supported.
    """
    if control_queries.ndim != 2 or control_queries.shape[0] < 1:
        raise ValueError("at least one control query embedding is required")
    target = candidate_margin_vector(target_query, candidate_embeddings, molecule_ptr)
    controls = torch.stack([
        candidate_margin_vector(control, candidate_embeddings, molecule_ptr)
        for control in control_queries
    ])
    mean_control = controls.mean(dim=0)
    return target - mean_control, target, mean_control


def bounded_residual_target(residual: torch.Tensor, bound: float) -> torch.Tensor:
    """Smoothly bound a signed teacher residual while preserving zero and sign."""
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("residual bound must be positive")
    return float(bound) * torch.tanh(residual / float(bound))


def clean_candidate_residual_loss(
    current_query: torch.Tensor,
    current_candidates: torch.Tensor,
    initial_query: torch.Tensor,
    initial_candidates: torch.Tensor,
    molecule_ptr: Sequence[int],
    teacher_residual: torch.Tensor,
    *,
    residual_bound: float = 0.10,
    candidate_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Fit clean-space margin changes to a fixed paired counterfactual target.

    Both current query and current reference embeddings receive gradient.  The
    initialization and teacher are fixed anchors.  This closes the old gap where
    only the perturbed action view learned to rank.
    """
    current = candidate_margin_vector(current_query, current_candidates, molecule_ptr)
    with torch.no_grad():
        initial = candidate_margin_vector(initial_query, initial_candidates, molecule_ptr)
        target = bounded_residual_target(teacher_residual, residual_bound)
    if target.shape != current.shape:
        raise ValueError("teacher residual does not match the candidate margin vector")
    prediction = current - initial
    element = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.01)
    if candidate_weight is not None:
        if candidate_weight.shape != element.shape or torch.any(candidate_weight < 0):
            raise ValueError("candidate weights are invalid")
        weight = candidate_weight / candidate_weight.sum().clamp_min(1e-12)
        loss = torch.sum(weight * element)
    else:
        loss = element.mean()
    return loss, {
        "predicted_residual": prediction,
        "bounded_teacher_residual": target,
        "absolute_error": torch.abs(prediction - target),
    }


class ContextualPeakGate(nn.Module):
    """Exactly identity-initialized reweighting of observed contextual peaks."""

    def __init__(self, dimension: int, hidden_dimension: int = 128, strength: float = 0.25):
        super().__init__()
        if dimension < 2 or hidden_dimension < 2 or not 0 < strength <= 1:
            raise ValueError("invalid contextual peak-gate dimensions or strength")
        self.dimension = int(dimension)
        self.strength = float(strength)
        self.norm = nn.LayerNorm(dimension)
        self.scorer = nn.Sequential(
            nn.Linear(dimension + 2, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        sequence: torch.Tensor,
        spectra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or spectra.ndim != 3 or sequence.shape[:2] != spectra.shape[:2]:
            raise ValueError("contextual sequence and spectrum shapes disagree")
        if sequence.shape[2] != self.dimension or spectra.shape[2] < 2:
            raise ValueError("contextual sequence dimension is invalid")
        fragment = sequence[:, 1:, :]
        mz = spectra[:, 1:, 0]
        intensity = spectra[:, 1:, 1]
        mask = (mz > 0) & (intensity > 0)
        auxiliary = torch.stack((mz / 1000.0, intensity), dim=-1).to(fragment.dtype)
        features = torch.cat((self.norm(fragment), auxiliary), dim=-1)
        logits = self.scorer(features).squeeze(-1)
        # 2*sigmoid(0)=1 exactly. Padding is forced to zero and never contributes.
        gate = (2.0 * torch.sigmoid(logits)) * mask.to(logits.dtype)
        base_weight = intensity.to(fragment.dtype) * mask.to(fragment.dtype)
        gated_weight = base_weight * gate
        base_pool = torch.sum(base_weight.unsqueeze(-1) * fragment, dim=1) / (
            base_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        gated_pool = torch.sum(gated_weight.unsqueeze(-1) * fragment, dim=1) / (
            gated_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        precursor = sequence[:, 0, :] + self.strength * (gated_pool - base_pool)
        return precursor, gate


class PeakGatedSharedEncoder(nn.Module):
    """DreaMS backbone/head wrapper that changes the shared embedding itself."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        dimension: int,
        gate_hidden_dimension: int = 128,
        gate_strength: float = 0.25,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.peak_gate = ContextualPeakGate(dimension, gate_hidden_dimension, gate_strength)

    def forward(self, spectra: torch.Tensor, *, return_gate: bool = False):
        sequence = self.backbone(spectra, None)
        precursor, gate = self.peak_gate(sequence, spectra)
        embedding = F.normalize(self.head(precursor), p=2, dim=-1)
        return (embedding, gate) if return_gate else embedding


@dataclass(frozen=True)
class ActionKey:
    identity: str
    query: int
    mechanism: str
    action: int


def no_replacement_hierarchical_sample(
    actions: Iterable[ActionKey],
    rng: np.random.Generator,
    *,
    maximum_queries_per_identity: int = 1,
) -> list[ActionKey]:
    """Sample identity -> query -> mechanism/action without within-epoch reuse."""
    if maximum_queries_per_identity < 1:
        raise ValueError("maximum queries per identity must be positive")
    by_identity: dict[str, dict[int, list[ActionKey]]] = {}
    seen_actions: set[int] = set()
    for item in actions:
        if item.action in seen_actions:
            raise ValueError(f"duplicate action id {item.action}")
        seen_actions.add(item.action)
        by_identity.setdefault(item.identity, {}).setdefault(item.query, []).append(item)
    output: list[ActionKey] = []
    identities = sorted(by_identity)
    rng.shuffle(identities)
    for identity in identities:
        queries = sorted(by_identity[identity])
        rng.shuffle(queries)
        for query in queries[:maximum_queries_per_identity]:
            candidates = by_identity[identity][query]
            mechanisms: dict[str, list[ActionKey]] = {}
            for item in candidates:
                mechanisms.setdefault(item.mechanism, []).append(item)
            names = sorted(mechanisms)
            mechanism = names[int(rng.integers(0, len(names)))]
            choices = mechanisms[mechanism]
            output.append(choices[int(rng.integers(0, len(choices)))])
    if len({item.action for item in output}) != len(output):
        raise RuntimeError("hierarchical sampler reused an action")
    return output


def project_auxiliary_against_reference(
    auxiliary: Sequence[torch.Tensor | None],
    reference: Sequence[torch.Tensor | None],
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """PCGrad-style projection only when an auxiliary branch opposes safety."""
    if len(auxiliary) != len(reference) or epsilon <= 0:
        raise ValueError("gradient lists or epsilon are invalid")
    dot = None
    reference_norm = None
    for aux, ref in zip(auxiliary, reference):
        if aux is None or ref is None:
            continue
        value = torch.sum(aux * ref)
        norm = torch.sum(ref * ref)
        dot = value if dot is None else dot + value
        reference_norm = norm if reference_norm is None else reference_norm + norm
    if dot is None or reference_norm is None:
        return list(auxiliary), {"dot": 0.0, "projected": 0.0, "coefficient": 0.0}
    coefficient = torch.clamp(-dot / reference_norm.clamp_min(epsilon), min=0.0)
    projected: list[torch.Tensor | None] = []
    for aux, ref in zip(auxiliary, reference):
        if aux is None:
            projected.append(None)
        elif ref is None:
            projected.append(aux)
        else:
            projected.append(aux + coefficient * ref)
    return projected, {
        "dot": float(dot.detach().cpu()),
        "projected": float((coefficient > 0).detach().cpu()),
        "coefficient": float(coefficient.detach().cpu()),
    }

