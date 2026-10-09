"""Joint evidence model for the evidence-qualified DreaMS system.

Modules are retained in the registry, but only evidence-qualified modules are
eligible for a production score.  Candidate-level applicability is explicit:
an unavailable candidate/module cell is neutral rather than a zero score.
Reliability weights may become exactly zero, which permits a harmful or
out-of-domain expert to abstain without deleting its audit trail.

Expected modules include the shared/condition encoders (official, Noise,
ChemAware), full-spectrum evidence (WSE), both Noise rerankers, the P2b local
fragment views, ChemAware candidate evidence, and BioAware event evidence.
Missing evidence is represented by ``availability=0``; it is never confused
with negative evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-6


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Ascending fractional ranks in [0, 1], averaging exact ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / max(len(values) - 1, 1)


def rank_logit_evidence(
    scores: np.ndarray,
    query_ptr: np.ndarray,
    availability: np.ndarray | None = None,
    clip: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert incomparable module scores into query-local log-odds evidence.

    Parameters
    ----------
    scores:
        ``(modules, candidates)`` raw scores. Larger must mean better.
    query_ptr:
        Candidate offsets for each query.
    availability:
        ``(modules, candidates)`` candidate-level applicability.  The legacy
        ``(queries, modules)`` shape is accepted and expanded for compatibility.

    Returns
    -------
    evidence, summaries:
        Rank-logit candidate evidence and label-free query summaries.  The
        latter contains, per module, availability, normalized top gap, score
        dispersion and top-candidate agreement with the other modules.
    """
    scores = np.asarray(scores, dtype=np.float64)
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    if scores.ndim != 2 or query_ptr.ndim != 1 or query_ptr[0] != 0:
        raise ValueError("invalid score matrix or query_ptr")
    if int(query_ptr[-1]) != scores.shape[1]:
        raise ValueError("query_ptr does not span candidate score matrix")
    q_count = len(query_ptr) - 1
    m_count = scores.shape[0]
    if availability is None:
        availability = np.ones_like(scores, dtype=np.float64)
    availability = np.asarray(availability, dtype=np.float64)
    if availability.shape == (q_count, m_count):
        expanded = np.zeros_like(scores, dtype=np.float64)
        for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
            expanded[:, int(left):int(right)] = availability[query, :, None]
        availability = expanded
    if availability.shape != scores.shape:
        raise ValueError("availability must have shape (modules, candidates)")
    if np.any((availability < 0) | (availability > 1)):
        raise ValueError("availability values must be in [0, 1]")
    if np.any(np.sum(availability > 0, axis=0) == 0):
        raise ValueError("every candidate must have at least one applicable module")

    evidence = np.zeros_like(scores, dtype=np.float32)
    summaries = np.zeros((q_count, m_count, 4), dtype=np.float32)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        if right <= left:
            continue
        winners = np.full(m_count, -1, dtype=np.int64)
        for module in range(m_count):
            present = availability[module, left:right] > 0
            summaries[query, module, 0] = float(np.mean(present))
            if not np.any(present):
                continue
            block = scores[module, left:right]
            if not np.all(np.isfinite(block[present])):
                raise ValueError(f"non-finite module scores at query={query}, module={module}")
            observed = block[present]
            percentile = np.clip(_average_rank(observed), 1e-3, 1.0 - 1e-3)
            local = np.log(percentile / (1.0 - percentile))
            local -= np.mean(local)
            target = evidence[module, left:right]
            target[present] = np.clip(local, -clip, clip)
            ordered = np.sort(observed)[::-1]
            spread = float(ordered[0] - ordered[-1])
            summaries[query, module, 1] = (
                float((ordered[0] - ordered[1]) / spread)
                if len(ordered) > 1 and spread > 0 else 0.0
            )
            # Unavailable cells are placeholders, not observations.  Including
            # them here would let arbitrary fill values alter every module's
            # query-level reliability even though their candidate weights are
            # gated to zero.
            summaries[query, module, 2] = float(np.std(observed))
            top = np.flatnonzero(present & (block == np.max(observed)))
            winners[module] = int(top[0]) if len(top) == 1 else -1
        valid = winners >= 0
        for module in range(m_count):
            if winners[module] < 0 or not np.any(valid):
                continue
            summaries[query, module, 3] = float(
                np.mean(winners[valid] == winners[module])
            )
    return evidence, summaries.reshape(q_count, -1)


@dataclass(frozen=True)
class UnifiedEvidenceOutput:
    logits: torch.Tensor
    module_weights: torch.Tensor
    module_contributions: torch.Tensor
    consensus_penalty: torch.Tensor


class AllModuleEvidenceModel(nn.Module):
    """Candidate-aware product-of-evidence model with evidence abstention."""

    def __init__(
        self,
        module_names: tuple[str, ...],
        summary_features_per_module: int = 4,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if len(module_names) < 2 or len(set(module_names)) != len(module_names):
            raise ValueError("module names must be unique and contain at least two modules")
        self.module_names = module_names
        self.module_count = len(module_names)
        summary_dim = self.module_count * summary_features_per_module
        self.reliability = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.module_count),
        )
        self.raw_scale = nn.Parameter(torch.zeros(self.module_count))
        self.raw_consensus = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        evidence: torch.Tensor,
        query_ptr: torch.Tensor,
        summaries: torch.Tensor,
        availability: torch.Tensor,
    ) -> UnifiedEvidenceOutput:
        """Fuse all evidence into one candidate logit per query.

        ``evidence`` and ``availability`` are ``(M, C)``. Reliability is learned
        per query, then gated per candidate. ReLU permits exact zero weights.
        """
        if evidence.ndim != 2 or evidence.shape[0] != self.module_count:
            raise ValueError("evidence must have shape (modules, candidates)")
        if availability.shape != evidence.shape:
            raise ValueError("availability shape mismatch")
        query_precision = F.relu(self.reliability(summaries) + 1.0)
        scale = F.softplus(self.raw_scale) + EPS

        logits = torch.zeros(evidence.shape[1], dtype=evidence.dtype, device=evidence.device)
        contributions = torch.zeros_like(evidence)
        candidate_weights = torch.zeros_like(evidence)
        penalties = torch.zeros_like(logits)
        consensus_strength = F.softplus(self.raw_consensus)
        for query in range(len(query_ptr) - 1):
            left, right = int(query_ptr[query]), int(query_ptr[query + 1])
            if right <= left:
                continue
            block = evidence[:, left:right] * scale[:, None]
            present = availability[:, left:right]
            precision = query_precision[query, :, None] * present
            denom = precision.sum(dim=0, keepdim=True)
            fallback = present / present.sum(dim=0, keepdim=True).clamp_min(EPS)
            w = torch.where(denom > EPS, precision / denom.clamp_min(EPS), fallback)
            candidate_weights[:, left:right] = w
            contributions[:, left:right] = w * block
            mean = torch.sum(w * block, dim=0)
            # Candidate-specific disagreement: a candidate supported by only
            # one evidence family cannot look identical to multi-view support.
            variance = torch.sum(w * torch.square(block - mean[None, :]), dim=0)
            penalties[left:right] = consensus_strength * variance
            logits[left:right] = mean - penalties[left:right]
        return UnifiedEvidenceOutput(logits, candidate_weights, contributions, penalties)


def listwise_loss(
    logits: torch.Tensor,
    query_ptr: torch.Tensor,
    labels: torch.Tensor,
    module_weights: torch.Tensor,
    availability: torch.Tensor,
    balance_strength: float = 0.0,
) -> torch.Tensor:
    """Query-listwise NLL; optional entropy term is disabled by default."""
    losses = []
    for query in range(len(query_ptr) - 1):
        left, right = int(query_ptr[query]), int(query_ptr[query + 1])
        if right <= left:
            continue
        target = torch.nonzero(labels[left:right] > 0, as_tuple=False).flatten()
        if len(target) != 1:
            raise ValueError(f"query {query} must contain exactly one positive candidate")
        losses.append(F.cross_entropy(logits[left:right][None, :], target[:1]))
    if not losses:
        raise ValueError("no non-empty query lists")
    nll = torch.stack(losses).mean()
    if balance_strength <= 0:
        return nll
    present = availability > 0
    active_weights = module_weights[present]
    if not len(active_weights):
        return nll
    entropy_penalty = torch.sum(active_weights * torch.log(active_weights.clamp_min(EPS))) / len(active_weights)
    return nll + float(balance_strength) * entropy_penalty


def strict_ranks(logits: np.ndarray, query_ptr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Strict positive ranks; ties count against the positive."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    ranks = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        positive = np.flatnonzero(labels[left:right] > 0)
        if len(positive) != 1:
            raise ValueError("each query must have exactly one positive")
        p = float(logits[left + positive[0]])
        ranks.append(1 + int(np.sum(logits[left:right] >= p)) - 1)
    return np.asarray(ranks, dtype=np.int32)
