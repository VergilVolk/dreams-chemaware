"""Core primitives for conditional-null candidate energy ranking.

This module deliberately does not implement expert routing.  One scalar score
is learned for every candidate molecule, while the official/Noise spectral
score remains an explicit offset.  Auxiliary evidence may change a ranking
only through an evidence vector that can also be reconstructed under each
registered, nuisance-matched null arm.

The core is kept independent of repository artefacts so its invariants can be
tested on CPU before any Slurm job is submitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn


CHEM_ARMS = ("zero", "reversed", "rotated")
FEATURE_NAMES = (
    "spectral_primary_percentile",
    "spectral_primary_gap_from_v1",
    "neutral_loss_percentile",
    "chem_centered_mean_advantage",
    "chem_advantage_min",
    "chem_advantage_max",
    "chem_advantage_positive_fraction",
)


def validate_query_ledger(query_ptr: np.ndarray, molecule_count: int) -> np.ndarray:
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        query_ptr.ndim != 1
        or len(query_ptr) < 2
        or int(query_ptr[0]) != 0
        or int(query_ptr[-1]) != int(molecule_count)
        or np.any(np.diff(query_ptr) < 2)
    ):
        raise ValueError("query_ptr must span all molecules with >=2 candidates per query")
    return query_ptr


def percentile_by_query(values: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    """Return [0,1] within-query percentiles with exact ties averaged."""
    values = np.asarray(values, dtype=np.float64)
    query_ptr = validate_query_ledger(query_ptr, len(values))
    output = np.empty(len(values), dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        block = values[left:right]
        if not np.all(np.isfinite(block)):
            raise ValueError("non-finite candidate score")
        order = np.argsort(block, kind="stable")
        ranks = np.empty(len(block), dtype=np.float64)
        ranks[order] = np.arange(len(block), dtype=np.float64)
        sorted_values = block[order]
        start = 0
        while start < len(block):
            stop = start + 1
            while stop < len(block) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = np.mean(ranks[order[start:stop]])
            start = stop
        output[left:right] = ranks / max(len(block) - 1, 1)
    return output


def deterministic_keyed_derangements(
    values: np.ndarray,
    candidate_keys: np.ndarray,
    query_keys: np.ndarray,
    query_ptr: np.ndarray,
    seeds: tuple[int, ...] = (20261004, 20261005, 20261006),
) -> np.ndarray:
    """Candidate-order-invariant, label-blind within-query derangements.

    Candidates are first put into a canonical order using deployment-visible
    stable identifiers.  A query-keyed non-zero cyclic shift then reassigns
    the score multiset.  Reordering an input candidate block and undoing that
    reorder therefore produces bitwise-identical null evidence.
    """
    values = np.asarray(values, dtype=np.float64)
    candidate_keys = np.asarray(candidate_keys, dtype=str)
    query_keys = np.asarray(query_keys, dtype=str)
    query_ptr = validate_query_ledger(query_ptr, len(values))
    if candidate_keys.shape != values.shape or query_keys.shape != (len(query_ptr) - 1,):
        raise ValueError("candidate/query keys do not align with the ledger")
    if not seeds:
        raise ValueError("at least one derangement seed is required")
    output = np.empty((len(seeds), len(values)), dtype=np.float64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        keys = candidate_keys[left:right]
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"candidate keys are not unique in query {query}")
        canonical = np.argsort(keys, kind="stable") + left
        n = len(canonical)
        for arm, seed in enumerate(seeds):
            payload = f"{seed}|{query_keys[query]}".encode("utf-8")
            shift = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (n - 1)
            output[arm, canonical] = values[np.roll(canonical, shift)]
    return output


def build_candidate_features(
    *,
    spectral_primary: np.ndarray,
    v1_score: np.ndarray,
    neutral_loss: np.ndarray,
    chem_primary: np.ndarray,
    chem_references: np.ndarray,
    query_ptr: np.ndarray,
) -> np.ndarray:
    """Build the same feature layout for real and counterfactual arms.

    ``chem_references`` has shape ``(n_nulls, n_molecules)``.  Swapping which
    chemistry arm is primary therefore changes values, never feature identity.
    """
    arrays = [spectral_primary, v1_score, neutral_loss, chem_primary]
    molecule_count = len(np.asarray(spectral_primary))
    query_ptr = validate_query_ledger(query_ptr, molecule_count)
    for value in arrays:
        if np.asarray(value).shape != (molecule_count,):
            raise ValueError("candidate evidence shape mismatch")
    references = np.asarray(chem_references, dtype=np.float64)
    if references.ndim != 2 or references.shape[1] != molecule_count or references.shape[0] < 1:
        raise ValueError("chem_references must be [null_arm, molecule]")
    numeric = [spectral_primary, v1_score, neutral_loss, chem_primary, references]
    if any(not np.all(np.isfinite(np.asarray(value, dtype=np.float64))) for value in numeric):
        raise ValueError("non-finite evidence reached candidate features")

    spectral_pct = percentile_by_query(spectral_primary, query_ptr)
    v1_pct = percentile_by_query(v1_score, query_ptr)
    neutral_pct = percentile_by_query(neutral_loss, query_ptr)
    chem_primary = np.asarray(chem_primary, dtype=np.float64).copy()
    advantages = chem_primary[None, :] - references
    advantage_mean = advantages.mean(axis=0)
    advantage_min = advantages.min(axis=0)
    advantage_max = advantages.max(axis=0)
    advantage_positive = (advantages > 0).mean(axis=0)
    features = np.column_stack((
        spectral_pct,
        spectral_pct - v1_pct,
        neutral_pct,
        advantage_mean,
        advantage_min,
        advantage_max,
        advantage_positive,
    )).astype(np.float32)
    if features.shape != (molecule_count, len(FEATURE_NAMES)) or not np.all(np.isfinite(features)):
        raise RuntimeError("conditional-null feature contract drifted")
    return features


class BoundedResidualEnergy(nn.Module):
    """Small residual energy head with an exact zero initial state."""

    def __init__(self, feature_count: int, hidden: int = 16, residual_bound: float = 0.25):
        super().__init__()
        if feature_count < 1 or hidden < 1 or residual_bound <= 0:
            raise ValueError("invalid residual energy dimensions")
        self.residual_bound = float(residual_bound)
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.residual_bound * torch.tanh(self.network(features).squeeze(-1))

    def forward(self, base_score: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return base_score + self.residual(features)


class AdditiveBoundedResidualEnergy(nn.Module):
    """Conservative no-interaction comparator with separate evidence heads.

    Each head receives only one evidence family.  The two scalar outputs are
    independently bounded and then summed, so neither hidden units nor a
    shared output nonlinearity can form a cross-family interaction.  Each head receives the full registered hidden
    width; this deliberately gives the additive comparator more parameters
    than the joint head rather than making interaction easy to win.
    """

    def __init__(
        self,
        feature_count: int,
        spectral_feature_count: int = 3,
        hidden: int = 16,
        residual_bound: float = 0.25,
    ):
        super().__init__()
        if not 0 < spectral_feature_count < feature_count:
            raise ValueError("additive split must leave both evidence families non-empty")
        if hidden < 1 or residual_bound <= 0:
            raise ValueError("invalid additive residual dimensions")
        self.spectral_feature_count = int(spectral_feature_count)
        self.residual_bound = float(residual_bound)
        self.spectral_network = nn.Sequential(
            nn.Linear(self.spectral_feature_count, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.chemical_network = nn.Sequential(
            nn.Linear(feature_count - self.spectral_feature_count, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for network in (self.spectral_network, self.chemical_network):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        spectral = torch.tanh(
            self.spectral_network(features[:, : self.spectral_feature_count]).squeeze(-1)
        )
        chemical_features = features[:, self.spectral_feature_count :]
        chemical_zero = torch.zeros_like(chemical_features)
        chemical = torch.tanh(
            self.chemical_network(chemical_features).squeeze(-1)
            - self.chemical_network(chemical_zero).squeeze(-1)
        )
        return 0.5 * self.residual_bound * (spectral + chemical)

    def forward(self, base_score: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return base_score + self.residual(features)


class AnchoredEvidenceEnergy(nn.Module):
    """Anchored spectral/chemical residual with an optional learned interaction.

    Chemical contribution is defined as f(spectral, chemical)-f(spectral, 0),
    so zero chemical contrast is an exact no-op after any amount of training.
    The spectral-only and chemistry-only arms use the same half-bound available
    to their corresponding component inside the joint model.
    """

    MODES = ("joint", "spectral_only", "chem_only")

    def __init__(
        self,
        feature_count: int,
        mode: str,
        spectral_feature_count: int = 3,
        hidden: int = 16,
        residual_bound: float = 0.25,
    ):
        super().__init__()
        if mode not in self.MODES or not 0 < spectral_feature_count < feature_count:
            raise ValueError("invalid anchored evidence mode or split")
        self.mode = mode
        self.spectral_feature_count = int(spectral_feature_count)
        self.residual_bound = float(residual_bound)
        if mode in ("joint", "spectral_only"):
            self.spectral_network = nn.Sequential(
                nn.Linear(self.spectral_feature_count, hidden), nn.Tanh(), nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.spectral_network[-1].weight)
            nn.init.zeros_(self.spectral_network[-1].bias)
        if mode == "joint":
            chemical_input = feature_count
        elif mode == "chem_only":
            chemical_input = feature_count - self.spectral_feature_count
        else:
            chemical_input = 0
        if chemical_input:
            self.chemical_network = nn.Sequential(
                nn.Linear(chemical_input, hidden), nn.Tanh(), nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.chemical_network[-1].weight)
            nn.init.zeros_(self.chemical_network[-1].bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        spectral_features = features[:, : self.spectral_feature_count]
        chemical_features = features[:, self.spectral_feature_count :]
        output = torch.zeros(len(features), dtype=features.dtype, device=features.device)
        if self.mode in ("joint", "spectral_only"):
            output = output + 0.5 * self.residual_bound * torch.tanh(
                self.spectral_network(spectral_features).squeeze(-1)
            )
        if self.mode == "joint":
            actual = self.chemical_network(features).squeeze(-1)
            anchored = self.chemical_network(torch.cat((
                spectral_features, torch.zeros_like(chemical_features),
            ), dim=1)).squeeze(-1)
            output = output + 0.5 * self.residual_bound * torch.tanh(actual - anchored)
        elif self.mode == "chem_only":
            actual = self.chemical_network(chemical_features).squeeze(-1)
            anchored = self.chemical_network(torch.zeros_like(chemical_features)).squeeze(-1)
            output = output + 0.5 * self.residual_bound * torch.tanh(actual - anchored)
        return output

    def forward(self, base_score: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return base_score + self.residual(features)


def strict_top1(scores: np.ndarray, labels: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    query_ptr = validate_query_ledger(query_ptr, len(scores))
    if labels.shape != scores.shape:
        raise ValueError("score/label shape mismatch")
    result = np.zeros(len(query_ptr) - 1, dtype=np.int8)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        block_labels = labels[left:right]
        if int(block_labels.sum()) != 1 or np.any(~np.isin(block_labels, (0, 1))):
            raise ValueError("every query must contain exactly one positive candidate")
        block = scores[left:right]
        winners = np.flatnonzero(block == np.max(block))
        result[query] = int(len(winners) == 1 and block_labels[int(winners[0])] == 1)
    return result


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    listwise: torch.Tensor
    worst_null: torch.Tensor
    trust: torch.Tensor


def conditional_null_loss(
    *,
    model: BoundedResidualEnergy | AdditiveBoundedResidualEnergy | AnchoredEvidenceEnergy,
    base_score: torch.Tensor,
    actual_features: torch.Tensor,
    null_features: torch.Tensor,
    labels: torch.Tensor,
    query_ptr: np.ndarray,
    null_margin: float = 0.02,
    null_weight: float = 0.5,
    trust_weight: float = 0.05,
) -> LossBreakdown:
    """Listwise candidate loss plus a worst-null falsification constraint.

    ``null_features`` is ``[arm, molecule, feature]``.  The null term compares
    complete query-level listwise loss under aligned evidence against every
    registered null and optimizes the worst arm.  This is not an expert
    selector and cannot choose a different scoring system per query.
    """
    if null_features.ndim != 3 or null_features.shape[1:] != actual_features.shape:
        raise ValueError("null feature tensor must be [arm, molecule, feature]")
    query_ptr = validate_query_ledger(query_ptr, len(base_score))
    if labels.shape != base_score.shape or actual_features.shape[0] != len(base_score):
        raise ValueError("training tensor alignment drifted")
    actual_score = model(base_score, actual_features)
    null_scores = torch.stack(
        [model(base_score, null_features[arm]) for arm in range(null_features.shape[0])],
        dim=0,
    )
    lengths = torch.as_tensor(np.diff(query_ptr), dtype=torch.long, device=base_score.device)
    query_index = torch.repeat_interleave(
        torch.arange(len(lengths), device=base_score.device), lengths,
    )
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("labels must be binary 0/1")
    positive = labels > 0
    positive_count = torch.zeros(len(lengths), dtype=torch.long, device=base_score.device)
    positive_count.scatter_add_(0, query_index, positive.to(torch.long))
    if not bool(torch.all(positive_count == 1)):
        raise ValueError("every query must contain exactly one positive candidate")
    maximum = torch.full(
        (len(lengths),), -torch.inf, dtype=actual_score.dtype, device=actual_score.device,
    )
    maximum.scatter_reduce_(0, query_index, actual_score, reduce="amax", include_self=True)
    exp_sum = torch.zeros_like(maximum)
    exp_sum.scatter_add_(0, query_index, torch.exp(actual_score - maximum[query_index]))
    log_partition = maximum + torch.log(exp_sum)
    true_actual = actual_score[positive]
    actual_query_loss = log_partition - true_actual
    listwise = torch.mean(actual_query_loss)
    null_query_losses = []
    for arm in range(null_scores.shape[0]):
        arm_score = null_scores[arm]
        arm_maximum = torch.full_like(maximum, -torch.inf)
        arm_maximum.scatter_reduce_(
            0, query_index, arm_score, reduce="amax", include_self=True,
        )
        arm_exp_sum = torch.zeros_like(maximum)
        arm_exp_sum.scatter_add_(
            0, query_index, torch.exp(arm_score - arm_maximum[query_index]),
        )
        arm_partition = arm_maximum + torch.log(arm_exp_sum)
        null_query_losses.append(arm_partition - arm_score[positive])
    null_query_loss = torch.stack(null_query_losses, dim=0)
    arm_losses = torch.relu(
        float(null_margin) + actual_query_loss[None, :] - null_query_loss
    ).mean(dim=1)
    worst_null = torch.max(arm_losses)
    trust = torch.mean(model.residual(actual_features) ** 2)
    total = listwise + float(null_weight) * worst_null + float(trust_weight) * trust
    return LossBreakdown(total=total, listwise=listwise, worst_null=worst_null, trust=trust)
