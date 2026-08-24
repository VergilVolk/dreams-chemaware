"""Frozen P2b local candidate-group rank fusion (deployable).

This is the deployment-clean copy of the frozen local rank-fusion scorer.  It
re-ranks the DreaMS candidates inside a single strict-10 ppm, same-adduct
candidate group using four pre-registered features, with the frozen weights:

    score = 0.10 * dreams_similarity
          + 0.00 * sqrt_cosine
          + 0.10 * entropy_similarity
          + 0.80 * neutral_loss_sqrt_cosine

(normalization ``absolute``, min_support 1, min_advantage 0.0).

Evidence (sealed P3, one-shot -- see docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md):

    P3-main-real-pristine (n=3000):  Recall@1 0.8793 -> 0.8900  (+1.07 pp)
                                     formula-cluster bootstrap CI [+0.24, +1.89] pp
                                     McNemar exact p = 0.0101
    P3-near-core-real-pristine (n=496): Recall@1 0.4879 -> 0.4456  (-4.23 pp,
                                     CI entirely negative, p = 0.0099)

The fusion is performed at the spectrum-pair level and only then aggregated to
candidate molecules, preventing a candidate from combining the best value of
each feature from different reference spectra (a "Frankenstein feature" leak).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

# --------------------------------------------------------------------------
# Frozen configuration (do NOT edit -- this is the sealed artifact).
# Provenance: data/validation/g8r_p2b_rank_fusion.json
# --------------------------------------------------------------------------
FROZEN_CONFIG: dict = {
    "normalization": "absolute",
    "weights": [0.1, 0.0, 0.1, 0.8],
    "min_support": 1,
    "min_advantage": 0.0,
    "selected_features": [
        "dreams_similarity",
        "sqrt_cosine",
        "entropy_similarity",
        "neutral_loss_sqrt_cosine",
    ],
    "artifact_sha256": "3ed4af01661556be198058db4dacdfb4f7e1893928586231cc02476ff28046fc",
    "selection_sha256": "af669c313403f5320d33cc4b26cc61e853e5a89b6a5a4837eb72bf8204bbc4e1",
}


@dataclass(frozen=True)
class FusionConfiguration:
    normalization: str
    weights: tuple[float, ...]
    min_support: int = 0
    min_advantage: float = 0.0


def fusion_configuration_from_mapping(body: Mapping) -> FusionConfiguration:
    """Deserialize one frozen JSON configuration into a hashable object.

    JSON has no tuple type, so ``weights`` is always read back as a list.  A
    plain ``FusionConfiguration(**body)`` therefore creates a frozen dataclass
    which is nevertheless unhashable.  Centralizing the conversion also makes
    the ablation and sealed-test evaluator reject malformed or drifted
    artifacts in exactly the same way.
    """
    expected = {"normalization", "weights", "min_support", "min_advantage"}
    keys = set(body)
    if keys != expected:
        raise ValueError(
            f"invalid fusion-configuration fields: missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    weights = tuple(float(value) for value in body["weights"])
    if len(weights) != 4 or not np.all(np.isfinite(weights)):
        raise ValueError("fusion weights must contain four finite values")
    normalization = str(body["normalization"])
    if normalization not in {"absolute", "query_minmax"}:
        raise ValueError(f"unknown normalization: {normalization}")
    min_support = int(body["min_support"])
    min_advantage = float(body["min_advantage"])
    if min_support < 0 or not np.isfinite(min_advantage):
        raise ValueError("invalid fusion gate")
    return FusionConfiguration(
        normalization=normalization,
        weights=weights,
        min_support=min_support,
        min_advantage=min_advantage,
    )


def validate_ptr(ptr: np.ndarray, size: int, name: str) -> np.ndarray:
    ptr = np.asarray(ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or ptr[-1] != size:
        raise ValueError(f"invalid {name}")
    if np.any(np.diff(ptr) <= 0):
        raise ValueError(f"{name} contains an empty group")
    return ptr


def grouped_max(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Maximum for non-empty contiguous groups defined by ``ptr``."""
    values = np.asarray(values)
    ptr = validate_ptr(ptr, len(values), "group pointer")
    return np.maximum.reduceat(values, ptr[:-1], axis=0)


def normalize_pair_features(
    pair_features: np.ndarray,
    query_pair_ptr: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Put selected higher-is-better similarities on comparable scales.

    ``absolute`` maps DreaMS cosine from [-1, 1] to [0, 1] and clips the
    three classical similarities to [0, 1]. ``query_minmax`` then applies a
    query-local min-max transform.  Constant features become zero and cannot
    create arbitrary ordering.
    """
    values = np.asarray(pair_features, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("pair_features must be a two-dimensional matrix")
    query_pair_ptr = validate_ptr(query_pair_ptr, len(values), "query-pair pointer")
    values[:, 0] = np.clip((values[:, 0] + 1.0) / 2.0, 0.0, 1.0)
    values[:, 1:] = np.clip(values[:, 1:], 0.0, 1.0)
    if mode == "absolute":
        return values
    if mode != "query_minmax":
        raise ValueError(f"unknown normalization: {mode}")
    for left, right in zip(query_pair_ptr[:-1], query_pair_ptr[1:]):
        block = values[left:right]
        low = block.min(axis=0)
        span = block.max(axis=0) - low
        values[left:right] = np.divide(
            block - low,
            span,
            out=np.zeros_like(block),
            where=span > 1e-12,
        )
    return values


def strict_rank(scores: np.ndarray, positive_index: int = 0) -> tuple[int, float, float]:
    """Rank with every negative tie counted against the positive."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not 0 <= positive_index < len(scores):
        raise ValueError("invalid candidate score vector")
    positive = float(scores[positive_index])
    negatives = np.delete(scores, positive_index)
    rank = 1 + int(np.sum(negatives >= positive))
    return rank, 1.0 / rank, positive - float(negatives.max())


def unique_top_index(scores: np.ndarray, eps: float = 1e-12) -> int | None:
    scores = np.asarray(scores, dtype=np.float64)
    best = float(scores.max())
    winners = np.flatnonzero(scores >= best - eps)
    return int(winners[0]) if len(winners) == 1 else None


def fuse_one_query(
    normalized_pairs: np.ndarray,
    baseline_pairs: np.ndarray,
    molecule_ptr: np.ndarray,
    weights: np.ndarray,
    raw_vote_columns: tuple[int, ...],
    min_support: int,
    min_advantage: float,
) -> tuple[np.ndarray, bool, int]:
    """Return molecule scores, whether fusion was used, and raw vote support."""
    normalized_pairs = np.asarray(normalized_pairs, dtype=np.float64)
    baseline_pairs = np.asarray(baseline_pairs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    molecule_ptr = validate_ptr(molecule_ptr, len(normalized_pairs), "molecule pointer")
    if len(baseline_pairs) != len(normalized_pairs) or normalized_pairs.shape[1] != len(weights):
        raise ValueError("fusion arrays are not aligned")
    baseline_molecule = grouped_max(baseline_pairs, molecule_ptr)
    fused_pair = normalized_pairs @ weights
    fused_molecule = grouped_max(fused_pair, molecule_ptr)
    fused_top = unique_top_index(fused_molecule)
    baseline_top = unique_top_index(baseline_molecule)
    if fused_top is None:
        return baseline_molecule, False, 0

    support = 0
    for column in raw_vote_columns:
        raw_molecule = grouped_max(normalized_pairs[:, column], molecule_ptr)
        if unique_top_index(raw_molecule) == fused_top:
            support += 1
    if fused_top == baseline_top:
        advantage = 0.0
    elif baseline_top is None:
        advantage = float("inf")
    else:
        advantage = float(fused_molecule[fused_top] - fused_molecule[baseline_top])
    use_fusion = support >= min_support and advantage + 1e-12 >= min_advantage
    return (fused_molecule if use_fusion else baseline_molecule), use_fusion, support


def frozen_weights(config: Mapping = FROZEN_CONFIG) -> np.ndarray:
    """Return the frozen 4-element weight vector as a float64 array."""
    cfg = fusion_configuration_from_mapping(
        {k: config[k] for k in ("normalization", "weights", "min_support", "min_advantage")}
    )
    return np.asarray(cfg.weights, dtype=np.float64)
