"""Leave-one-spectrum-out, candidate-pair exact ChemAware evidence."""
from __future__ import annotations

import numpy as np

from chemaware_crossview_consensus_core import AGGREGATIONS, aggregate


def crossview_pair_boundary_proof(
    candidate_identity: np.ndarray,
    candidate_valid: np.ndarray,
    arm_utility: np.ndarray,
    query_identity: np.ndarray,
    truth_identity: np.ndarray,
    false_identity: np.ndarray,
    baseline_identity: np.ndarray | None = None,
    *,
    primary_arm: int = 0,
    absolute_threshold: float = -np.inf,
    dominance_threshold: float = 0.0,
    aggregation: str = "q25",
    min_context: int = 1,
) -> dict[str, np.ndarray]:
    """Prove a true-vs-current-false boundary using other spectra only.

    The held spectrum contributes candidate identities but contributes no
    utility.  For each candidate and arm, utilities are aligned by molecular
    identity across the other spectra of the held query's known training
    identity.  A proof requires a positive primary true-minus-false boundary
    and strict excess over every matched null-arm boundary.
    """
    candidate_identity = np.asarray(candidate_identity).astype(str)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    arm_utility = np.asarray(arm_utility, dtype=np.float64)
    query_identity = np.asarray(query_identity).astype(str)
    truth_identity = np.asarray(truth_identity).astype(str)
    false_identity = np.asarray(false_identity).astype(str)
    queries = candidate_valid.shape[0]
    if baseline_identity is None:
        baseline_identity = np.full(queries, "", dtype="U1")
    baseline_identity = np.asarray(baseline_identity).astype(str)
    if candidate_identity.shape != candidate_valid.shape:
        raise ValueError("candidate identity and validity arrays are not aligned")
    if arm_utility.ndim != 3 or arm_utility.shape[1:] != candidate_valid.shape:
        raise ValueError("arm utility must have shape [arm, query, candidate]")
    if any(len(x) != queries for x in (query_identity, truth_identity, false_identity, baseline_identity)):
        raise ValueError("query identity arrays are not aligned")
    if not 0 <= primary_arm < arm_utility.shape[0] or arm_utility.shape[0] < 2:
        raise ValueError("at least one primary and one null arm are required")
    if aggregation not in AGGREGATIONS or min_context < 1:
        raise ValueError("invalid cross-view aggregation setting")

    boundary = np.full((arm_utility.shape[0], queries), np.nan, dtype=np.float64)
    truth_score = np.full((arm_utility.shape[0], queries), np.nan, dtype=np.float64)
    false_score = np.full((arm_utility.shape[0], queries), np.nan, dtype=np.float64)
    truth_support = np.zeros(queries, dtype=np.int16)
    false_support = np.zeros(queries, dtype=np.int16)
    specificity = np.full(queries, np.nan, dtype=np.float64)
    eligible = np.zeros(queries, dtype=bool)

    for identity in np.unique(query_identity):
        group = np.flatnonzero(query_identity == identity)
        if len(group) < 2:
            continue
        for held in group:
            if not truth_identity[held] or not false_identity[held]:
                continue
            context = group[group != held]
            scores: list[np.ndarray] = []
            supports: list[int] = []
            for target in (truth_identity[held], false_identity[held]):
                values_by_arm: list[list[float]] = [
                    [] for _ in range(arm_utility.shape[0])
                ]
                for other in context:
                    if target == baseline_identity[other]:
                        # Candidate tables encode actions relative to the
                        # official top candidate.  Keeping that candidate is
                        # the exact no-action reference and has zero utility
                        # under every matched arm.
                        for arm in range(arm_utility.shape[0]):
                            values_by_arm[arm].append(0.0)
                        continue
                    matches = np.flatnonzero(
                        candidate_valid[other]
                        & (candidate_identity[other] == target)
                    )
                    if len(matches) > 1:
                        raise RuntimeError("candidate identity appears twice in one query")
                    if not len(matches):
                        continue
                    slot = int(matches[0])
                    values = arm_utility[:, other, slot]
                    if np.all(np.isfinite(values)):
                        for arm, value in enumerate(values):
                            values_by_arm[arm].append(float(value))
                support = min(map(len, values_by_arm))
                supports.append(support)
                if support < min_context:
                    scores.append(np.full(arm_utility.shape[0], np.nan))
                else:
                    scores.append(np.asarray([
                        aggregate(np.asarray(values), aggregation)
                        for values in values_by_arm
                    ]))
            if min(supports) < min_context:
                continue
            truth_score[:, held], false_score[:, held] = scores
            truth_support[held], false_support[held] = supports
            boundary[:, held] = scores[0] - scores[1]
            primary = float(boundary[primary_arm, held])
            null_best = float(np.max(np.delete(boundary[:, held], primary_arm)))
            specificity[held] = primary - null_best
            eligible[held] = bool(
                primary >= max(0.0, absolute_threshold)
                and specificity[held] > dominance_threshold
            )

    return {
        "eligible": eligible,
        "boundary": boundary,
        "specificity": specificity,
        "truth_score": truth_score,
        "false_score": false_score,
        "truth_support": truth_support,
        "false_support": false_support,
    }
