"""Candidate-identity aligned cross-spectrum consensus for ChemAware teachers."""
from __future__ import annotations

import numpy as np


AGGREGATIONS = ("mean", "median", "q25", "lcb1")


def aggregate(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("consensus values must be finite and nonempty")
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "q25":
        return float(np.quantile(values, 0.25))
    if mode == "lcb1":
        return float(np.mean(values) - np.std(values) / np.sqrt(len(values)))
    raise ValueError(f"unknown aggregation: {mode}")


def crossview_policy(
    baseline_rank: np.ndarray,
    candidate_identity: np.ndarray,
    candidate_valid: np.ndarray,
    proposal_rank: np.ndarray,
    utility: np.ndarray,
    query_identity: np.ndarray,
    *,
    threshold: float,
    aggregation: str,
    min_context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply leave-one-spectrum-out consensus without using outcome labels."""

    baseline_rank = np.asarray(baseline_rank, dtype=np.int64)
    candidate_identity = np.asarray(candidate_identity).astype(str)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    proposal_rank = np.asarray(proposal_rank, dtype=np.int64)
    utility = np.asarray(utility, dtype=np.float64)
    query_identity = np.asarray(query_identity).astype(str)
    shape = candidate_valid.shape
    if any(array.shape != shape for array in (candidate_identity, proposal_rank, utility)):
        raise ValueError("candidate arrays are not aligned")
    if len(baseline_rank) != shape[0] or len(query_identity) != shape[0]:
        raise ValueError("query arrays are not aligned")
    if aggregation not in AGGREGATIONS or min_context < 1:
        raise ValueError("invalid consensus setting")

    rank = baseline_rank.copy()
    selected_slot = np.full(len(rank), -1, dtype=np.int16)
    selected_score = np.full(len(rank), -np.inf, dtype=np.float64)
    support = np.zeros(len(rank), dtype=np.int16)
    for identity in np.unique(query_identity):
        group = np.flatnonzero(query_identity == identity)
        if len(group) < 2:
            continue
        for held in group:
            context = group[group != held]
            candidates = np.flatnonzero(candidate_valid[held])
            scored: list[tuple[float, str, int, int]] = []
            for slot in candidates:
                candidate = candidate_identity[held, slot]
                values = []
                for other in context:
                    matches = np.flatnonzero(
                        candidate_valid[other] & (candidate_identity[other] == candidate)
                    )
                    if len(matches):
                        value = float(utility[other, int(matches[0])])
                        if np.isfinite(value):
                            values.append(value)
                if len(values) >= min_context:
                    scored.append((aggregate(np.asarray(values), aggregation), candidate, int(slot), len(values)))
            if not scored:
                continue
            # Candidate identity breaks exact score ties deterministically.
            score, _candidate, slot, count = max(scored, key=lambda item: (item[0], item[1]))
            selected_score[held] = score
            support[held] = count
            if score >= threshold:
                selected_slot[held] = slot
                rank[held] = proposal_rank[held, slot]
    return rank, selected_slot, selected_score, support


def crossview_dominance_policy(
    baseline_rank: np.ndarray,
    candidate_identity: np.ndarray,
    candidate_valid: np.ndarray,
    proposal_rank: np.ndarray,
    arm_utility: np.ndarray,
    query_identity: np.ndarray,
    *,
    primary_arm: int,
    absolute_threshold: float,
    dominance_threshold: float,
    aggregation: str,
    min_context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a matched arm into the primary role and require counterfactual dominance."""

    arm_utility = np.asarray(arm_utility, dtype=np.float64)
    if arm_utility.ndim != 3:
        raise ValueError("arm utility must have shape [arm, query, candidate]")
    if not 0 <= primary_arm < arm_utility.shape[0] or arm_utility.shape[0] < 2:
        raise ValueError("invalid primary arm")
    baseline_rank = np.asarray(baseline_rank, dtype=np.int64)
    candidate_identity = np.asarray(candidate_identity).astype(str)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    proposal_rank = np.asarray(proposal_rank, dtype=np.int64)
    query_identity = np.asarray(query_identity).astype(str)
    if arm_utility.shape[1:] != candidate_valid.shape:
        raise ValueError("arm utility and candidate arrays are not aligned")
    if candidate_identity.shape != candidate_valid.shape or proposal_rank.shape != candidate_valid.shape:
        raise ValueError("candidate arrays are not aligned")
    if len(baseline_rank) != candidate_valid.shape[0] or len(query_identity) != len(baseline_rank):
        raise ValueError("query arrays are not aligned")
    if aggregation not in AGGREGATIONS or min_context < 1:
        raise ValueError("invalid consensus setting")

    rank = baseline_rank.copy()
    selected_slot = np.full(len(rank), -1, dtype=np.int16)
    primary_score = np.full(len(rank), -np.inf, dtype=np.float64)
    dominance = np.full(len(rank), -np.inf, dtype=np.float64)
    support = np.zeros(len(rank), dtype=np.int16)
    for identity in np.unique(query_identity):
        group = np.flatnonzero(query_identity == identity)
        if len(group) < 2:
            continue
        for held in group:
            context = group[group != held]
            candidates = np.flatnonzero(candidate_valid[held])
            scored: list[tuple[float, float, str, int, int]] = []
            for slot in candidates:
                candidate = candidate_identity[held, slot]
                values_by_arm: list[list[float]] = [[] for _ in range(arm_utility.shape[0])]
                for other in context:
                    matches = np.flatnonzero(
                        candidate_valid[other] & (candidate_identity[other] == candidate)
                    )
                    if not len(matches):
                        continue
                    other_slot = int(matches[0])
                    for arm in range(arm_utility.shape[0]):
                        value = float(arm_utility[arm, other, other_slot])
                        if np.isfinite(value):
                            values_by_arm[arm].append(value)
                if min(map(len, values_by_arm)) < min_context:
                    continue
                arm_score = np.asarray([
                    aggregate(np.asarray(values), aggregation) for values in values_by_arm
                ])
                primary = float(arm_score[primary_arm])
                rival = float(np.max(np.delete(arm_score, primary_arm)))
                scored.append((primary - rival, primary, candidate, int(slot), min(map(len, values_by_arm))))
            if not scored:
                continue
            advantage, primary, _candidate, slot, count = max(
                scored, key=lambda item: (item[0], item[1], item[2]),
            )
            primary_score[held] = primary; dominance[held] = advantage; support[held] = count
            if primary >= absolute_threshold and advantage >= dominance_threshold:
                selected_slot[held] = slot
                rank[held] = proposal_rank[held, slot]
    return rank, selected_slot, primary_score, dominance, support
