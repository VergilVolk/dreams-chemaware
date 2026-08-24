"""Core model and losses for the G8R P2 molecule-listwise residual reranker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RAW_FEATURES = [
    "sqrt_cosine", "linear_cosine", "entropy_similarity",
    "intensity_coverage_min", "intensity_coverage_mean",
    "matched_peak_fraction_min", "top10_match_fraction",
    "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean", "peak_count_ratio",
]
FEATURE_NAMES = ["dreams_similarity", *RAW_FEATURES]


class ResidualListwiseRanker(nn.Module):
    """Bounded residual on top of the frozen DreaMS cosine score.

    The last layer is initialized to zero, therefore the untrained model is
    exactly DreaMS. ``delta_bound`` makes the largest possible score change
    explicit and auditable.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        delta_bound: float = 0.06,
    ) -> None:
        super().__init__()
        if n_features < 2 or hidden_dim < 0 or delta_bound <= 0:
            raise ValueError("invalid residual-ranker dimensions")
        self.delta_bound = float(delta_bound)
        if hidden_dim == 0:
            self.net = nn.Linear(n_features, 1)
            final = self.net
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features_standardized: torch.Tensor, baseline: torch.Tensor):
        raw_delta = self.net(features_standardized).squeeze(-1)
        delta = self.delta_bound * torch.tanh(raw_delta)
        return baseline + delta, delta


def molecule_max_scores(pair_scores: torch.Tensor, molecule_ptr: torch.Tensor) -> torch.Tensor:
    """Deployment-aligned max aggregation from spectra to candidate molecules."""
    if molecule_ptr.ndim != 1 or len(molecule_ptr) < 2:
        raise ValueError("molecule_ptr must contain at least [0, n_pairs]")
    if int(molecule_ptr[0]) != 0 or int(molecule_ptr[-1]) != len(pair_scores):
        raise ValueError("molecule_ptr does not span pair_scores")
    counts = molecule_ptr[1:] - molecule_ptr[:-1]
    if bool(torch.any(counts <= 0)):
        raise ValueError("empty candidate molecule")
    molecule_index = torch.repeat_interleave(
        torch.arange(len(counts), device=pair_scores.device), counts,
    )
    scores = torch.full(
        (len(counts),), -torch.inf, dtype=pair_scores.dtype, device=pair_scores.device,
    )
    scores.scatter_reduce_(0, molecule_index, pair_scores, reduce="amax", include_self=True)
    return scores


@dataclass(frozen=True)
class QueryLoss:
    total: torch.Tensor
    listwise: torch.Tensor
    safety: torch.Tensor
    residual: torch.Tensor
    final_margin: torch.Tensor
    baseline_margin: torch.Tensor


def query_listwise_loss(
    model: ResidualListwiseRanker,
    features_standardized: torch.Tensor,
    baseline_scores: torch.Tensor,
    molecule_ptr: torch.Tensor,
    positive_molecule: int,
    temperature: float = 0.1,
    safety_weight: float = 1.0,
    allowed_margin_drop: float = 0.005,
    residual_weight: float = 0.01,
) -> QueryLoss:
    """One-query top-one listwise loss with a DreaMS safety constraint.

    For a baseline-correct query, the hinge requires the learned positive-vs-
    hardest-negative margin to remain within ``allowed_margin_drop`` of the
    official DreaMS margin. Baseline-wrong queries are free to cross the boundary.
    """
    if temperature <= 0 or safety_weight < 0 or residual_weight < 0:
        raise ValueError("invalid loss hyperparameter")
    final_pair, delta = model(features_standardized, baseline_scores)
    final_mol = molecule_max_scores(final_pair, molecule_ptr)
    base_mol = molecule_max_scores(baseline_scores, molecule_ptr)
    n_mol = len(final_mol)
    if n_mol < 2 or not 0 <= positive_molecule < n_mol:
        raise ValueError("each query needs one positive and at least one negative molecule")

    target = torch.as_tensor([positive_molecule], device=final_mol.device)
    listwise = F.cross_entropy((final_mol / temperature).unsqueeze(0), target)
    negative_mask = torch.ones(n_mol, dtype=torch.bool, device=final_mol.device)
    negative_mask[positive_molecule] = False
    final_margin = final_mol[positive_molecule] - final_mol[negative_mask].max()
    baseline_margin = base_mol[positive_molecule] - base_mol[negative_mask].max()
    safety_target = torch.clamp(baseline_margin.detach() - allowed_margin_drop, min=0.0)
    safety = F.relu(safety_target - final_margin)
    residual = (delta / model.delta_bound).square().mean()
    total = listwise + safety_weight * safety + residual_weight * residual
    return QueryLoss(total, listwise, safety, residual, final_margin, baseline_margin)


def evaluate_query_scores(
    pair_scores: np.ndarray,
    molecule_ptr: Sequence[int],
    positive_molecule: int,
) -> dict[str, float | int | bool]:
    """Strict molecule-level rank: ties against the positive count as errors."""
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if ptr[0] != 0 or ptr[-1] != len(pair_scores) or np.any(np.diff(ptr) <= 0):
        raise ValueError("invalid molecule_ptr")
    molecule_scores = np.asarray([pair_scores[l:r].max() for l, r in zip(ptr[:-1], ptr[1:])])
    positive = float(molecule_scores[positive_molecule])
    negatives = np.delete(molecule_scores, positive_molecule)
    rank = 1 + int(np.sum(negatives >= positive))
    return {
        "rank": rank,
        "top1": rank == 1,
        "mrr": 1.0 / rank,
        "margin": positive - float(negatives.max()),
    }


def deterministic_formula_fold(formula: str, n_folds: int) -> int:
    import hashlib

    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    digest = hashlib.blake2b(str(formula).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_folds
