"""Pure targets for direct chemical-prior injection into shared retrieval scores.

The chemical prior is a training-time ``query x candidate x rule`` relation.
It is not a spectrum perturbation and it is never selected with the current
DreaMS margin or input Jacobian.  A deployable student still receives only one
spectrum at a time; candidate structures are used solely to compile frozen
candidate-score targets during training.

The key object is a candidate-centred residual.  Centring removes the common
score offset that cannot affect ranking and prevents the optimizer from spending
capacity on increasing every candidate similarity.  A disabled prior produces
an exact all-zero target, unlike a softplus margin floor at its boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectChemicalPrior:
    """Frozen candidate-level chemical supervision for a ragged query graph."""

    raw_score: np.ndarray
    centered_residual: np.ndarray
    active_query: np.ndarray


def validate_query_ptr(query_ptr: np.ndarray, candidates: int) -> np.ndarray:
    """Validate and return a canonical int64 ragged-query pointer."""
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        ptr.ndim != 1
        or len(ptr) < 2
        or int(ptr[0]) != 0
        or int(ptr[-1]) != int(candidates)
        or np.any(np.diff(ptr) < 2)
    ):
        raise ValueError(
            "query_ptr must cover every candidate and each query needs a positive "
            "plus at least one negative"
        )
    return ptr


def candidate_center(
    values: np.ndarray,
    query_ptr: np.ndarray,
) -> np.ndarray:
    """Remove the molecule-equal mean independently inside every query."""
    score = np.asarray(values, dtype=np.float64)
    if score.ndim != 1 or not np.all(np.isfinite(score)):
        raise ValueError("candidate values must be a finite vector")
    ptr = validate_query_ptr(query_ptr, len(score))
    result = np.empty_like(score)
    for left, right in zip(ptr[:-1], ptr[1:]):
        block = score[int(left) : int(right)]
        result[int(left) : int(right)] = block - np.mean(block)
    return result.astype(np.float32)


def compile_structure_fragment_prior(
    query_rule_evidence: np.ndarray,
    candidate_rule_presence: np.ndarray,
    query_ptr: np.ndarray,
    *,
    rule_confidence: np.ndarray | None = None,
    minimum_total_evidence: float = 0.0,
) -> DirectChemicalPrior:
    """Compile structure-conditioned fragment evidence into candidate targets.

    ``query_rule_evidence[q, r]`` is the calibrated evidence that query ``q``
    contains the observed fragment or neutral loss required by rule ``r``.
    ``candidate_rule_presence[m, r]`` is the training-only probability that
    candidate molecule ``m`` satisfies the parent-structure predicate of rule
    ``r``.  Their weighted overlap is additive.  It is deliberately *not*
    normalized by the query's observed evidence: doing that would erase
    absolute reliability and, for a one-rule bank, make a weak acquisition
    domain as strong as a well-supported one.  A final dose cap belongs to the
    frozen ledger or loss contract and must be shared by every matched arm.

    Candidate zero in every query is still the identity label, but this function
    never reads that label and never forces the chemical teacher to prefer it.
    Whether the calibrated prior is useful must be decided against structure-
    swapped and peak-permuted controls on formula-held retrieval.
    """
    evidence = np.asarray(query_rule_evidence, dtype=np.float64)
    presence = np.asarray(candidate_rule_presence, dtype=np.float64)
    if evidence.ndim != 2 or presence.ndim != 2:
        raise ValueError("rule evidence and presence must be matrices")
    if evidence.shape[1] != presence.shape[1]:
        raise ValueError("query evidence and candidate predicates use different rules")
    if not np.all(np.isfinite(evidence)) or not np.all(np.isfinite(presence)):
        raise ValueError("chemical-prior inputs must be finite")
    if np.any(evidence < 0) or np.any(presence < 0) or np.any(presence > 1):
        raise ValueError("evidence must be nonnegative and presence must lie in [0, 1]")
    ptr = validate_query_ptr(query_ptr, len(presence))
    if len(ptr) != len(evidence) + 1:
        raise ValueError("query evidence does not align with candidate blocks")
    if minimum_total_evidence < 0:
        raise ValueError("minimum_total_evidence must be nonnegative")
    confidence = (
        np.ones(evidence.shape[1], dtype=np.float64)
        if rule_confidence is None
        else np.asarray(rule_confidence, dtype=np.float64)
    )
    if (
        confidence.shape != (evidence.shape[1],)
        or not np.all(np.isfinite(confidence))
        or np.any(confidence < 0)
        or np.any(confidence > 1)
    ):
        raise ValueError("rule_confidence must be a finite vector in [0, 1]")

    weighted_evidence = evidence * confidence[None, :]
    total = np.sum(weighted_evidence, axis=1)
    active = total > float(minimum_total_evidence)
    raw = np.zeros(len(presence), dtype=np.float64)
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        if not active[query]:
            continue
        raw[int(left) : int(right)] = (
            presence[int(left) : int(right)] @ weighted_evidence[query]
        )
    centered = candidate_center(raw, ptr)
    # Evidence is not a trainable signal when every candidate receives the
    # same chemical score.  Treat such blocks as inactive rather than letting
    # coverage accounting call a zero residual an "active" prior.
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        if active[query] and not np.any(centered[int(left) : int(right)] != 0.0):
            active[query] = False

    # Queries without admissible, candidate-discriminating evidence are exact
    # no-ops, including under future loss implementations that consume only
    # this residual vector.
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        if not active[query]:
            centered[int(left) : int(right)] = 0.0
    return DirectChemicalPrior(
        raw_score=raw.astype(np.float32),
        centered_residual=centered,
        active_query=active.astype(bool),
    )


def centered_student_residual(
    student_score: np.ndarray,
    official_score: np.ndarray,
    query_ptr: np.ndarray,
) -> np.ndarray:
    """Return the ranking-relevant part of the student's change from official."""
    student = np.asarray(student_score, dtype=np.float64)
    official = np.asarray(official_score, dtype=np.float64)
    if student.shape != official.shape:
        raise ValueError("student and official candidate scores must align")
    return candidate_center(student - official, query_ptr)


def direct_prior_target(
    prior: DirectChemicalPrior,
    alpha: float,
) -> np.ndarray:
    """Scale a frozen prior; alpha zero is an exact numerical no-op."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0:
        return np.zeros_like(prior.centered_residual, dtype=np.float32)
    return (float(alpha) * prior.centered_residual).astype(np.float32)
