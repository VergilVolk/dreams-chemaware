"""Shared, label-free rule-evidence helpers for the ChemAware G0 audit.

The functions in this module deliberately operate on observed spectrum mass
patterns.  They do not infer a unique fragment structure and must never be
used to define positive or negative molecular identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


M_H = 1.00782503223
POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class CompiledRules:
    rules: tuple[dict, ...]
    fragment_loss_indices: np.ndarray
    fragment_loss_targets: np.ndarray
    precursor_offset_indices: np.ndarray
    precursor_offset_targets: np.ndarray
    peak_mz_indices: np.ndarray
    peak_mz_targets: np.ndarray
    other_indices: np.ndarray


def compile_rules(rules: list[dict]) -> CompiledRules:
    """Compile the two large exact-target families for vectorized matching."""
    loss_indices: list[int] = []
    loss_targets: list[float] = []
    offset_indices: list[int] = []
    offset_targets: list[float] = []
    peak_indices: list[int] = []
    peak_targets: list[float] = []
    other_indices: list[int] = []
    for index, rule in enumerate(rules):
        kind = str(rule.get("match_type"))
        if kind == "mass_diff":
            if str(rule.get("source", "")) == "MassBank record-derived":
                # build_massbank_rules.py derives these values from
                # |precursor_mz - exact_mass|.  They describe precursor/adduct
                # offsets and are not fragment neutral-loss rules.
                offset_indices.append(index)
                offset_targets.append(float(rule["value"]))
            else:
                loss_indices.append(index)
                loss_targets.append(float(rule["value"]))
        elif kind == "peak_mz":
            peak_indices.append(index)
            peak_targets.append(float(rule["value"]))
        else:
            other_indices.append(index)
    return CompiledRules(
        rules=tuple(rules),
        fragment_loss_indices=np.asarray(loss_indices, dtype=np.int64),
        fragment_loss_targets=np.asarray(loss_targets, dtype=np.float64),
        precursor_offset_indices=np.asarray(offset_indices, dtype=np.int64),
        precursor_offset_targets=np.asarray(offset_targets, dtype=np.float64),
        peak_mz_indices=np.asarray(peak_indices, dtype=np.int64),
        peak_mz_targets=np.asarray(peak_targets, dtype=np.float64),
        other_indices=np.asarray(other_indices, dtype=np.int64),
    )


def target_hits(sorted_values: np.ndarray, targets: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """Vectorized equivalent of nearest exact-target matching."""
    values = np.asarray(sorted_values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if len(values) == 0:
        return np.zeros(len(targets), dtype=bool)
    positions = np.searchsorted(values, targets, side="left")
    right = np.minimum(positions, len(values) - 1)
    left = np.maximum(positions - 1, 0)
    return np.minimum(np.abs(values[right] - targets), np.abs(values[left] - targets)) < tolerance


def _range_hit(sorted_values: np.ndarray, low: float, high: float) -> bool:
    if len(sorted_values) == 0:
        return False
    position = int(np.searchsorted(sorted_values, low, side="left"))
    return position < len(sorted_values) and float(sorted_values[position]) <= high


def match_compiled_rules(
    mz_padded: np.ndarray,
    precursor_mz: float,
    compiled: CompiledRules,
    tolerance: float = 0.02,
    parent_mass: float | None = None,
) -> np.ndarray:
    """Return one spectrum-level rule-hit vector.

    Fragment neutral-loss rules are matched against precursor-fragment loss,
    not arbitrary peak-pair differences.  The 79 MassBank rules generated
    from precursor-versus-exact-mass offsets are evaluated separately when a
    neutral parent mass is available.  ISO/HR rules retain peak-pair deltas.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    mz = np.sort(np.asarray(mz_padded, dtype=np.float64))
    mz = mz[np.isfinite(mz) & (mz > 0)]
    if len(mz):
        diffs = np.sort(np.abs(mz[:, None] - mz[None, :]).reshape(-1))
        neutral_losses = np.sort((float(precursor_mz) - mz)[(float(precursor_mz) - mz) > 0])
    else:
        diffs = np.empty(0, dtype=np.float64)
        neutral_losses = np.empty(0, dtype=np.float64)
    labels = np.zeros(len(compiled.rules), dtype=np.uint8)
    labels[compiled.fragment_loss_indices] = target_hits(
        neutral_losses, compiled.fragment_loss_targets, tolerance,
    ).astype(np.uint8)
    if parent_mass is not None and np.isfinite(parent_mass):
        precursor_offset = np.asarray([abs(float(precursor_mz) - float(parent_mass))])
        labels[compiled.precursor_offset_indices] = target_hits(
            precursor_offset, compiled.precursor_offset_targets, tolerance,
        ).astype(np.uint8)
    labels[compiled.peak_mz_indices] = target_hits(
        mz, compiled.peak_mz_targets, tolerance,
    ).astype(np.uint8)
    for index in compiled.other_indices:
        rule = compiled.rules[int(index)]
        kind = str(rule.get("match_type"))
        value = rule.get("value")
        if kind == "mass_range":
            low, high = map(float, value)
            labels[index] = _range_hit(diffs, low, high)
        elif kind == "hr_shift":
            count = float(value)
            if count == 0:
                eligible = diffs[diffs >= 12.0]
                labels[index] = bool(
                    len(eligible) and np.any(np.abs(eligible - np.round(eligible)) < tolerance)
                )
            else:
                labels[index] = bool(target_hits(diffs, np.asarray([abs(count) * M_H]), tolerance)[0])
        elif kind == "parity":
            labels[index] = bool(
                len(diffs)
                and np.any((np.round(diffs).astype(np.int64) % 2) == (round(precursor_mz) % 2))
            )
        elif kind == "mass_diff_range":
            low, high = map(float, value)
            labels[index] = bool(len(diffs) and np.any((diffs > high) | (diffs < low)))
        else:
            raise ValueError(f"unsupported rule match_type: {kind}")
    return labels


def packed_mask(indices: np.ndarray, n_rules: int) -> np.ndarray:
    values = np.zeros(n_rules, dtype=np.uint8)
    values[np.asarray(indices, dtype=np.int64)] = 1
    return np.packbits(values, bitorder="little")


def packed_jaccard(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Row-wise Jaccard for packed bit matrices; empty unions are NaN."""
    a = np.asarray(left, dtype=np.uint8)
    b = np.asarray(right, dtype=np.uint8)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("packed operands must be aligned two-dimensional arrays")
    if mask is not None:
        m = np.asarray(mask, dtype=np.uint8)
        if m.ndim != 1 or len(m) != a.shape[1]:
            raise ValueError("packed mask has wrong shape")
        a = np.bitwise_and(a, m)
        b = np.bitwise_and(b, m)
    intersection = POPCOUNT[np.bitwise_and(a, b)].sum(axis=1, dtype=np.int64)
    union = POPCOUNT[np.bitwise_or(a, b)].sum(axis=1, dtype=np.int64)
    return np.divide(
        intersection,
        union,
        out=np.full(len(a), np.nan, dtype=np.float32),
        where=union > 0,
    )


def nan_group_max(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """NaN-aware maximum for contiguous non-empty groups."""
    values = np.asarray(values, dtype=np.float64)
    ptr = np.asarray(ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or ptr[-1] != len(values):
        raise ValueError("invalid group pointer")
    output = np.full(len(ptr) - 1, np.nan, dtype=np.float64)
    for index, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        block = values[int(left):int(right)]
        if np.any(np.isfinite(block)):
            output[index] = float(np.nanmax(block))
    return output
