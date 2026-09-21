"""Core contracts for paired, nuisance-residual ChemAware policies.

The deployable intervention can see candidate-set score geometry, but the
chemical claim is restricted to the incremental signal carried by the
correct rule channel relative to a content-permuted channel.  This module
keeps that contrast and its negative controls explicit.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def base_and_chemical_feature_indices(
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split non-rule nuisance features from rule/action-dependent features."""
    base: list[int] = []
    chemical: list[int] = []
    for index, name in enumerate(names):
        lowered = str(name).lower()
        if "rule" in lowered or "action" in lowered:
            chemical.append(index)
        else:
            base.append(index)
    if not base or not chemical:
        raise ValueError("both nuisance and chemical feature blocks are required")
    return np.asarray(base, dtype=np.int64), np.asarray(chemical, dtype=np.int64)


def validate_matched_tables(
    correct: dict[str, np.ndarray], control: dict[str, np.ndarray],
) -> None:
    """Fail closed unless two arms describe exactly the same candidate rows."""
    for key in ("valid", "proposed_candidate", "baseline_candidate", "baseline_rank"):
        if key not in correct or key not in control:
            raise KeyError(f"matched-table key absent: {key}")
        if not np.array_equal(np.asarray(correct[key]), np.asarray(control[key])):
            raise ValueError(f"correct/control candidate tables drifted at {key}")
    if np.asarray(correct["feature"]).shape != np.asarray(control["feature"]).shape:
        raise ValueError("correct/control feature shapes differ")


def signed_rule_contrast(
    correct: dict[str, np.ndarray], control: dict[str, np.ndarray],
    chemical_indices: np.ndarray,
) -> np.ndarray:
    """Return correct-minus-control features, zeroing padded candidate slots."""
    validate_matched_tables(correct, control)
    indices = np.asarray(chemical_indices, dtype=np.int64)
    delta = (
        np.asarray(correct["feature"], dtype=np.float32)[..., indices]
        - np.asarray(control["feature"], dtype=np.float32)[..., indices]
    )
    output = np.asarray(delta, dtype=np.float32)
    output[~np.asarray(correct["valid"], dtype=bool)] = 0.0
    return output


def symmetric_center_contrast(
    tables: Sequence[dict[str, np.ndarray]],
    center_index: int,
    chemical_indices: np.ndarray,
) -> np.ndarray:
    """Describe one putative center against every other exchangeable center.

    Each chemical feature is summarized by its mean, lower envelope, upper
    envelope, dispersion, positive fraction, and negative fraction across
    pairwise center-minus-other contrasts.  The same representation can be
    evaluated with a real rule center or any matched null as the center.
    """
    if len(tables) < 3:
        raise ValueError("symmetric center contrast requires at least three centers")
    if center_index < 0 or center_index >= len(tables):
        raise IndexError("center index out of range")
    center = tables[center_index]
    indices = np.asarray(chemical_indices, dtype=np.int64)
    differences = []
    for index, table in enumerate(tables):
        validate_matched_tables(center, table)
        if index == center_index:
            continue
        differences.append(
            np.asarray(center["feature"], dtype=np.float32)[..., indices]
            - np.asarray(table["feature"], dtype=np.float32)[..., indices]
        )
    stacked = np.stack(differences, axis=0)
    summary = np.concatenate((
        np.mean(stacked, axis=0),
        np.min(stacked, axis=0),
        np.max(stacked, axis=0),
        np.std(stacked, axis=0),
        np.mean(stacked > 0, axis=0),
        np.mean(stacked < 0, axis=0),
    ), axis=-1).astype(np.float32)
    summary[~np.asarray(center["valid"], dtype=bool)] = 0.0
    return summary


def permute_candidate_contrast_within_strata(
    contrast: np.ndarray,
    valid: np.ndarray,
    baseline_rank: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute candidate-row contrasts within difficulty-matched strata.

    Strata preserve candidate count and whether the official baseline is
    already correct.  The same candidate-row multiset is therefore retained,
    while query/candidate alignment is broken.
    """
    values = np.asarray(contrast, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    baseline = np.asarray(baseline_rank)
    if values.shape[:2] != mask.shape or len(baseline) != len(mask):
        raise ValueError("contrast/valid/baseline shapes are incompatible")
    output = np.zeros_like(values)
    source = np.full(mask.shape, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate_count = mask.sum(axis=1)
    for count in np.unique(candidate_count):
        for correct in (False, True):
            queries = np.flatnonzero(
                (candidate_count == count) & ((baseline == 1) == correct)
            )
            if not len(queries):
                continue
            flat_positions = np.argwhere(mask[queries])
            destination_query = queries[flat_positions[:, 0]]
            destination_slot = flat_positions[:, 1]
            order = rng.permutation(len(flat_positions))
            source_query = destination_query[order]
            source_slot = destination_slot[order]
            output[destination_query, destination_slot] = values[source_query, source_slot]
            source[destination_query, destination_slot] = (
                source_query.astype(np.int64) * mask.shape[1] + source_slot.astype(np.int64)
            )
    return output, source


def rotate_candidate_contrast_truthblind(
    contrast: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Break candidate alignment without labels or cross-query state.

    Valid candidate rows are cyclically rotated within each query.  This keeps
    the query-level chemical-contrast multiset fixed while assigning every
    contrast to a different candidate whenever at least two candidates exist.
    Single-candidate queries receive a zero contrast because no within-query
    negative alignment exists.  The transform is deterministic and accepts no
    truth rank, identity, formula, or outcome, so the resulting null arm is
    available during deployment as well as development.
    """
    values = np.asarray(contrast, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if values.shape[:2] != mask.shape:
        raise ValueError("contrast/valid shapes are incompatible")
    output = np.zeros_like(values)
    source = np.full(mask.shape, -1, dtype=np.int64)
    width = mask.shape[1]
    for query in range(len(mask)):
        slots = np.flatnonzero(mask[query])
        if len(slots) < 2:
            continue
        source_slots = np.roll(slots, 1)
        output[query, slots] = values[query, source_slots]
        source[query, slots] = query * width + source_slots
    return output, source


def combine_residual_probability(
    nuisance_probability: np.ndarray,
    residual: np.ndarray,
    dose: float,
) -> np.ndarray:
    """Add a bounded residual correction to a nuisance probability."""
    if float(dose) < 0:
        raise ValueError("residual dose must be nonnegative")
    nuisance = np.asarray(nuisance_probability, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if nuisance.shape != correction.shape:
        raise ValueError("nuisance and residual shapes differ")
    return np.clip(nuisance + float(dose) * correction, 0.0, 1.0)


def combine_residual_score(
    nuisance_probability: np.ndarray,
    residual: np.ndarray,
    dose: float,
) -> np.ndarray:
    """Form an unclipped orthogonal score.

    A residual regressor estimates a conditional label residual, not a second
    calibrated probability.  Keeping the sum linear avoids creating a
    selection shortcut at probability clipping boundaries.
    """
    if float(dose) < 0:
        raise ValueError("residual dose must be nonnegative")
    nuisance = np.asarray(nuisance_probability, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if nuisance.shape != correction.shape:
        raise ValueError("nuisance and residual shapes differ")
    return nuisance + float(dose) * correction
