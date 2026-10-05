"""Empirical-Bayes weights for deployable ChemAware rule-response channels."""

from __future__ import annotations

import hashlib

import numpy as np


def aggregate_formula_contrasts(
    query_contrast: np.ndarray, formulas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average query contrasts within formula before any channel is selected."""

    values = np.asarray(query_contrast, dtype=np.float64)
    formula = np.asarray(formulas).astype(str)
    if values.ndim != 2 or formula.shape != (len(values),):
        raise ValueError("query contrasts and formulas are not aligned")
    unique, inverse = np.unique(formula, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse)
    return sums / counts[:, None], unique


def reliability_weights(
    formula_contrast: np.ndarray,
    formula_names: np.ndarray,
    document_frequency: np.ndarray,
    documents: int,
    *,
    seed: int,
    consensus_fraction: float = 2 / 3,
    standard_error_floor: float = 1e-4,
    z_cap: float = 5.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return nonnegative, cross-split-consistent identity-evidence weights."""

    values = np.asarray(formula_contrast, dtype=np.float64)
    names = np.asarray(formula_names).astype(str)
    df = np.asarray(document_frequency, dtype=np.float64)
    if values.ndim != 2 or names.shape != (len(values),) or df.shape != (values.shape[1],):
        raise ValueError("formula contrasts and document frequency are not aligned")
    if not len(values) or documents <= 0 or not 0 < consensus_fraction <= 1:
        raise ValueError("invalid reliability input")
    mean = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(max(1, len(values)))
    z = mean / np.maximum(standard_error, standard_error_floor)
    split = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little") % 3
        for value in names
    ])
    split_mean = np.stack([
        np.mean(values[split == fold], axis=0) if np.any(split == fold) else np.zeros(values.shape[1])
        for fold in range(3)
    ])
    agreement = np.mean(split_mean > 0, axis=0)
    idf = np.log((documents + 1.0) / (df + 1.0)) + 1.0
    raw = np.maximum(z, 0.0)
    raw[agreement < consensus_fraction] = 0.0
    raw = np.minimum(raw, z_cap) * idf
    positive = raw > 0
    if np.any(positive):
        raw /= np.median(raw[positive])
    report = {
        "formula_clusters": int(len(values)),
        "positive_weight_channels": int(np.sum(positive)),
        "positive_effect_channels": int(np.sum(mean > 0)),
        "three_of_three_positive_channels": int(np.sum(agreement == 1.0)),
        "two_of_three_or_better_channels": int(np.sum(agreement >= 2 / 3)),
        "consensus_fraction": float(consensus_fraction),
        "weight_quantiles_positive": (
            np.quantile(raw[positive], (0.0, 0.25, 0.5, 0.75, 1.0)).astype(float).tolist()
            if np.any(positive) else [0.0] * 5
        ),
    }
    return raw.astype(np.float32), report


def prevalence_matched_permutation(
    weights: np.ndarray,
    document_frequency: np.ndarray,
    category_boundary: int,
    *,
    seed: int,
    bins: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Break channel identity while preserving category and prevalence strata."""

    value = np.asarray(weights, dtype=np.float32)
    df = np.asarray(document_frequency, dtype=np.float64)
    if value.ndim != 1 or df.shape != value.shape or not 0 < category_boundary < len(value):
        raise ValueError("invalid matched-permutation input")
    rng = np.random.default_rng(seed)
    output = value.copy()
    moved = 0
    for left, right in ((0, category_boundary), (category_boundary, len(value))):
        local = np.arange(left, right)
        edges = np.quantile(df[local], np.linspace(0, 1, bins + 1))
        group = np.clip(np.searchsorted(edges[1:-1], df[local], side="right"), 0, bins - 1)
        for group_index in range(bins):
            index = local[group == group_index]
            if len(index) < 2:
                continue
            permutation = rng.permutation(index)
            output[index] = value[permutation]
            moved += int(np.sum(index != permutation))
    return output, {
        "moved_channels": moved,
        "category_preserved": True,
        "document_frequency_bins": int(bins),
        "weight_multiset_preserved": bool(np.array_equal(np.sort(output), np.sort(value))),
    }

