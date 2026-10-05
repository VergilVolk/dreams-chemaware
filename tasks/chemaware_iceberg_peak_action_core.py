"""Pure helpers for structure-differential ICEBERG peak actions.

The helpers deliberately never expose a candidate at deployment.  They turn a
training-only pair of forward predictions (true structure versus the hardest
same-formula negative) into an intensity-only counterfactual of the observed
experimental spectrum.
"""
from __future__ import annotations

import numpy as np
import torch


def hard_negative_indices(query_ptr: np.ndarray, flat_distance: np.ndarray) -> np.ndarray:
    """Return the lowest-distance non-positive candidate for every query."""
    ptr = np.asarray(query_ptr, dtype=np.int64)
    score = np.asarray(flat_distance, dtype=np.float32)
    if ptr.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(score):
        raise ValueError("query pointer does not span flat candidate scores")
    if np.any(np.diff(ptr) < 2):
        raise ValueError("every query requires at least one negative candidate")
    return np.asarray([
        int(left + 1 + np.argmin(score[int(left) + 1:int(right)]))
        for left, right in zip(ptr[:-1], ptr[1:])
    ], dtype=np.int64)


def peak_bins(mz: np.ndarray, n_bins: int = 15_000, upper_mz: float = 1500.0) -> np.ndarray:
    """Match the bin lookup used by the frozen ICEBERG teacher audits."""
    bins = np.linspace(0.0, upper_mz, n_bins)
    return np.clip(np.digitize(np.asarray(mz), bins=bins), 0, n_bins - 1)


def differential_evidence(
    true_prediction: np.ndarray,
    negative_prediction: np.ndarray,
    observed_mz: np.ndarray,
) -> np.ndarray:
    """Signed peak evidence after per-candidate max normalization.

    Positive values mean the true candidate predicts an observed peak more
    strongly; negative values mean the hardest negative predicts it more
    strongly.  Square-root scaling reduces domination by a few intense bins.
    """
    true = np.asarray(true_prediction, dtype=np.float32)
    negative = np.asarray(negative_prediction, dtype=np.float32)
    if true.shape != negative.shape or true.ndim != 1:
        raise ValueError("candidate predictions must be aligned one-dimensional arrays")
    true = true / max(float(np.max(true)), 1e-12)
    negative = negative / max(float(np.max(negative)), 1e-12)
    index = peak_bins(observed_mz, len(true), 1500.0)
    return np.sqrt(np.maximum(true[index], 0.0)) - np.sqrt(np.maximum(negative[index], 0.0))


def apply_peak_action(
    clean: torch.Tensor,
    evidence: np.ndarray,
    mode: str,
    strength: float,
    top_k: int = 10,
) -> torch.Tensor:
    """Create an intensity-only action while preserving observed peak m/z.

    ``signed_exp`` continuously boosts true-supported peaks and attenuates
    negative-supported peaks.  The two top-k modes change an identical number
    of peak slots in every arm, which makes swapped/permuted controls tightly
    matched in action capacity.
    """
    if clean.ndim != 2 or clean.shape[1] != 2 or len(clean) < 2:
        raise ValueError("expected a preprocessed DreaMS spectrum tensor")
    if len(evidence) != len(clean) - 1:
        raise ValueError("evidence must align to non-precursor peak slots")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    output = clean.clone()
    intensity = output[1:, 1]
    valid = (output[1:, 0] > 0) & (intensity > 0)
    positions = np.flatnonzero(valid.cpu().numpy())
    if not len(positions):
        return output
    value = np.asarray(evidence, dtype=np.float32)[positions]
    if mode == "signed_exp":
        factor = np.exp(np.clip(strength * value, -4.0, 4.0)).astype(np.float32)
        intensity[torch.as_tensor(positions)] *= torch.from_numpy(factor).to(intensity)
    elif mode in {"conflict_attenuate", "support_boost"}:
        take = min(int(top_k), len(positions))
        if take <= 0:
            raise ValueError("top_k must be positive")
        order = np.argsort(value, kind="stable")
        chosen = positions[order[:take] if mode == "conflict_attenuate" else order[-take:]]
        index = torch.as_tensor(chosen)
        if mode == "conflict_attenuate":
            intensity[index] *= max(0.0, 1.0 - strength)
        else:
            intensity[index] *= 1.0 + strength
    else:
        raise ValueError(f"unknown peak action mode: {mode}")
    maximum = intensity.max()
    if float(maximum) > 0:
        intensity /= maximum
    return output
