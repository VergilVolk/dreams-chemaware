"""Pure helpers for ICEBERG-to-DreaMS direct shared-embedding training."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


ARMS = ("clean_duplicate", "correct_synthetic", "candidate_swapped", "peak_permuted")


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 3:
        raise ValueError("at least three folds are required")
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}|{str(value)}".encode()).digest()[:8], "little") % folds
        for value in formulas
    ], dtype=np.int16)


def action_prediction_indices(query_ptr: np.ndarray, arm: str) -> np.ndarray:
    """Map each teacher query to the prediction used as its action spectrum.

    Candidate molecules are ordered with the true molecule first.  The
    candidate-swapped null exactly matches ``np.roll(block, 1)`` in the frozen
    teacher audit, so the prediction assigned to the positive slot is the last
    (necessarily negative) candidate prediction.
    """
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or np.any(np.diff(ptr) < 2):
        raise ValueError("every teacher query must contain at least two candidates")
    if arm == "candidate_swapped":
        return ptr[1:] - 1
    return ptr[:-1].copy()


def rescue_positions(
    official_rank: np.ndarray,
    teacher_rank: np.ndarray,
    formula_fold: np.ndarray,
    inner_fold: int,
    outer_fold: int,
) -> np.ndarray:
    arrays = tuple(map(np.asarray, (official_rank, teacher_rank, formula_fold)))
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("teacher selection arrays are not aligned")
    if inner_fold == outer_fold:
        raise ValueError("inner and outer folds must differ")
    train = (arrays[2] != inner_fold) & (arrays[2] != outer_fold)
    return np.flatnonzero(train & (arrays[0] != 1) & (arrays[1] == 1))


def identity_balanced_positions(
    positions: np.ndarray, identities: np.ndarray, seed: int,
) -> np.ndarray:
    """Return at most one observation per identity in deterministic random order."""
    positions = np.asarray(positions, dtype=np.int64)
    identities = np.asarray(identities, dtype=str)
    if np.any((positions < 0) | (positions >= len(identities))):
        raise ValueError("position outside identity array")
    rng = np.random.default_rng(seed)
    order = positions[rng.permutation(len(positions))]
    selected: list[int] = []
    seen: set[str] = set()
    for position in order:
        identity = str(identities[position])
        if identity not in seen:
            selected.append(int(position))
            seen.add(identity)
    return np.asarray(selected, dtype=np.int64)


def assert_formula_disjoint(
    formulas: np.ndarray, formula_fold: np.ndarray, inner_fold: int, outer_fold: int,
) -> None:
    formulas = np.asarray(formulas, dtype=str)
    formula_fold = np.asarray(formula_fold, dtype=np.int16)
    groups = [
        set(formulas[(formula_fold != inner_fold) & (formula_fold != outer_fold)]),
        set(formulas[formula_fold == inner_fold]),
        set(formulas[formula_fold == outer_fold]),
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("formula leakage across train/inner/outer partitions")


def molecule_scores_from_pairs(pair_scores, local_ptr: np.ndarray):
    """Differentiable max-over-reference aggregation for one candidate list."""
    ptr = np.asarray(local_ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 3 or ptr[0] != 0 or ptr[-1] != len(pair_scores):
        raise ValueError("invalid local molecule pointer")
    return torch.stack([
        pair_scores[int(left):int(right)].max()
        for left, right in zip(ptr[:-1], ptr[1:])
    ])
