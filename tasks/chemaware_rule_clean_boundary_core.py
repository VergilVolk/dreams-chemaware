"""Pure contracts for rule-selected clean-boundary embedding transfer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def choose_opposed_boundaries(
    scores: np.ndarray,
    labels: np.ndarray,
    predicate: np.ndarray,
) -> tuple[int, int, int] | None:
    """Return positive, chemistry-negative, and hardness-matched control.

    The chemistry negative lacks the confirmed parent predicate.  The control
    negative has the predicate, is a different molecule, and is chosen nearest
    in frozen official score to prevent generic hardness from explaining an
    arm difference.  Returning ``None`` is a deliberate abstention.
    """
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int8)
    present = np.asarray(predicate, dtype=bool)
    if score.ndim != 1 or label.shape != score.shape or present.shape != score.shape:
        raise ValueError("candidate score, label, and predicate arrays must align")
    if not np.all(np.isfinite(score)):
        raise ValueError("candidate scores must be finite")
    positive = np.flatnonzero(label == 1)
    if len(positive) != 1 or not present[positive[0]]:
        return None
    negative = label == 0
    chemistry_pool = np.flatnonzero(negative & ~present)
    control_pool = np.flatnonzero(negative & present)
    if not len(chemistry_pool) or not len(control_pool):
        return None
    chemistry = int(chemistry_pool[np.argmax(score[chemistry_pool])])
    distance = np.abs(score[control_pool] - score[chemistry])
    control_order = np.lexsort((control_pool, -score[control_pool], distance))
    control = int(control_pool[control_order[0]])
    if len({int(positive[0]), chemistry, control}) != 3:
        raise RuntimeError("rule/control boundary did not select three molecules")
    return int(positive[0]), chemistry, control


def formula_balanced_weights(formulas: np.ndarray) -> np.ndarray:
    """Give every formula equal total mass and retain mean-one scale."""
    values = np.asarray(formulas).astype(str)
    if values.ndim != 1 or not len(values):
        raise ValueError("formula weights require a non-empty vector")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def paired_margin_each(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Smooth clean-spectrum positive-vs-negative boundary loss per query."""
    if (
        query.ndim != 2
        or positive.shape != query.shape
        or negative.shape != query.shape
        or margin < 0
        or temperature <= 0
    ):
        raise ValueError("invalid paired clean-boundary loss inputs")
    positive_score = torch.sum(query * positive, dim=1)
    negative_score = torch.sum(query * negative, dim=1)
    return temperature * F.softplus(
        (float(margin) + negative_score - positive_score) / float(temperature)
    )


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("weighted mean inputs must be aligned vectors")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")
    return torch.sum(values * weights) / torch.sum(weights)


def retrieval_metrics(ranks: np.ndarray, candidate_counts: np.ndarray) -> dict:
    rank = np.asarray(ranks, dtype=np.int64)
    count = np.asarray(candidate_counts, dtype=np.int64)
    if (
        rank.ndim != 1
        or count.shape != rank.shape
        or not len(rank)
        or np.any(count < 2)
        or np.any(rank < 1)
        or np.any(rank > count)
    ):
        raise ValueError("invalid retrieval rank/count arrays")
    per_query_auc = (count - rank) / (count - 1)
    output = {
        f"recall{cutoff}": float(np.mean(rank <= cutoff))
        for cutoff in (1, 5, 10, 20, 50)
    }
    output.update(
        {
            "mrr": float(np.mean(1.0 / rank)),
            "macro_auc": float(np.mean(per_query_auc)),
            "micro_auc": float(np.sum(count - rank) / np.sum(count - 1)),
        }
    )
    return output


def formula_cluster_interval(
    values: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    value = np.asarray(values, dtype=np.float64)
    formula = np.asarray(formulas).astype(str)
    if value.ndim != 1 or formula.shape != value.shape or not len(value) or draws < 1000:
        raise ValueError("invalid formula-cluster interval inputs")
    unique = np.unique(formula)
    macro = np.asarray([np.mean(value[formula == key]) for key in unique])
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled[draw] = np.mean(macro[rng.integers(0, len(macro), len(macro))])
    return {
        "formula_macro_mean": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }
