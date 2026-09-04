"""Auditable max-product inference for typed metabolite candidate factors.

The core intentionally knows nothing about phenotype labels or ground truth.
Callers provide calibrated unary log scores and explicit compatibility matrices.
Unknown is a normal candidate state, so sparse networks cannot force an identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


UNKNOWN = "__UNKNOWN__"


@dataclass(frozen=True)
class CandidateVariable:
    node_id: str
    candidates: tuple[str, ...]
    unary_log_score: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.unary_log_score, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(self.candidates) or not np.isfinite(values).all():
            raise ValueError(f"invalid unary scores for {self.node_id}")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"duplicate candidate state for {self.node_id}")
        if UNKNOWN not in self.candidates:
            raise ValueError(f"{self.node_id} lacks the mandatory unknown state")
        object.__setattr__(self, "unary_log_score", values)


@dataclass(frozen=True)
class PairFactor:
    factor_id: str
    left: str
    right: str
    compatibility: np.ndarray
    family: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(self.compatibility, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid compatibility matrix for {self.factor_id}")
        if self.left == self.right or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError(f"invalid factor endpoints/confidence for {self.factor_id}")
        object.__setattr__(self, "compatibility", values)


def calibrated_unary(
    candidate_ids: Iterable[str], scores: Iterable[float], *, temperature: float,
    unknown_margin: float,
) -> CandidateVariable:
    """Create centered log scores while preserving an explicit abstention state."""
    candidates = tuple(map(str, candidate_ids))
    values = np.asarray(list(scores), dtype=np.float64)
    if not candidates or len(candidates) != len(values) or temperature <= 0:
        raise ValueError("invalid candidate scores or temperature")
    centered = (values - np.max(values)) / temperature
    unknown = -abs(float(unknown_margin)) / temperature
    return CandidateVariable("", candidates + (UNKNOWN,), np.concatenate((centered, [unknown])))


class TypedFactorGraph:
    """Synchronous, degree-normalized max-product with bounded messages."""

    def __init__(
        self, variables: Iterable[CandidateVariable], factors: Iterable[PairFactor], *,
        damping: float = 0.5, message_cap: float = 2.0,
    ):
        self.variables = {variable.node_id: variable for variable in variables}
        self.factors = list(factors)
        if not self.variables or not 0 <= damping < 1 or message_cap <= 0:
            raise ValueError("invalid graph or inference parameters")
        self.damping = float(damping)
        self.message_cap = float(message_cap)
        self.incident: dict[str, list[int]] = {node: [] for node in self.variables}
        factor_ids = set()
        for index, factor in enumerate(self.factors):
            if factor.factor_id in factor_ids:
                raise ValueError(f"duplicate factor id: {factor.factor_id}")
            factor_ids.add(factor.factor_id)
            if factor.left not in self.variables or factor.right not in self.variables:
                raise ValueError(f"factor {factor.factor_id} references an unknown node")
            expected = (len(self.variables[factor.left].candidates), len(self.variables[factor.right].candidates))
            if factor.compatibility.shape != expected:
                raise ValueError(f"factor {factor.factor_id} shape {factor.compatibility.shape}, expected {expected}")
            self.incident[factor.left].append(index)
            self.incident[factor.right].append(index)

    def _degree_scale(self, factor: PairFactor) -> float:
        # Prevent hubs or dense reaction networks from receiving unbounded votes.
        degree = np.sqrt(max(1, len(self.incident[factor.left])) * max(1, len(self.incident[factor.right])))
        return factor.confidence / degree

    def infer(self, iterations: int = 20, tolerance: float = 1e-6) -> dict:
        if iterations < 1 or tolerance <= 0:
            raise ValueError("invalid inference controls")
        messages: dict[tuple[int, str], np.ndarray] = {}
        for index, factor in enumerate(self.factors):
            messages[(index, factor.left)] = np.zeros(len(self.variables[factor.left].candidates))
            messages[(index, factor.right)] = np.zeros(len(self.variables[factor.right].candidates))

        converged = False
        final_change = float("inf")
        for iteration in range(1, iterations + 1):
            beliefs = {}
            for node, variable in self.variables.items():
                belief = variable.unary_log_score.copy()
                for factor_index in self.incident[node]:
                    belief += messages[(factor_index, node)]
                beliefs[node] = belief
            updated: dict[tuple[int, str], np.ndarray] = {}
            final_change = 0.0
            for factor_index, factor in enumerate(self.factors):
                scale = self._degree_scale(factor)
                left_without = beliefs[factor.left] - messages[(factor_index, factor.left)]
                right_without = beliefs[factor.right] - messages[(factor_index, factor.right)]
                to_right = np.max(left_without[:, None] + scale * factor.compatibility, axis=0)
                to_left = np.max(right_without[None, :] + scale * factor.compatibility, axis=1)
                for node, proposed in ((factor.right, to_right), (factor.left, to_left)):
                    proposed -= np.max(proposed)
                    proposed = np.clip(proposed, -self.message_cap, self.message_cap)
                    old = messages[(factor_index, node)]
                    value = self.damping * old + (1.0 - self.damping) * proposed
                    updated[(factor_index, node)] = value
                    final_change = max(final_change, float(np.max(np.abs(value - old))))
            messages = updated
            if final_change <= tolerance:
                converged = True
                break

        beliefs = {}
        decisions = {}
        explanation = {}
        for node, variable in self.variables.items():
            belief = variable.unary_log_score.copy()
            contributions = []
            for factor_index in self.incident[node]:
                message = messages[(factor_index, node)]
                belief += message
                factor = self.factors[factor_index]
                contributions.append({
                    "factor_id": factor.factor_id, "family": factor.family,
                    "other_node": factor.right if factor.left == node else factor.left,
                    "message": message.tolist(),
                })
            order = np.argsort(-belief, kind="stable")
            winner, runner = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            beliefs[node] = belief
            decisions[node] = {
                "candidate_id": variable.candidates[winner],
                "abstained": variable.candidates[winner] == UNKNOWN,
                "margin": float(belief[winner] - belief[runner]),
                "belief": belief.tolist(), "candidates": list(variable.candidates),
            }
            explanation[node] = contributions
        return {
            "decisions": decisions, "messages": explanation,
            "iterations": iteration, "converged": converged,
            "maximum_final_message_change": final_change,
        }


def relation_compatibility(
    left: CandidateVariable, right: CandidateVariable,
    supported_pairs: set[tuple[str, str]], *, reward: float,
    unknown_value: float = 0.0,
) -> np.ndarray:
    """Compatibility with absence treated as unknown, never as negative truth."""
    output = np.zeros((len(left.candidates), len(right.candidates)), dtype=np.float64)
    for i, a in enumerate(left.candidates):
        for j, b in enumerate(right.candidates):
            if UNKNOWN in (a, b):
                output[i, j] = unknown_value
            elif tuple(sorted((a, b))) in supported_pairs:
                output[i, j] = reward
    return output


def identity_family_compatibility(
    left: CandidateVariable, right: CandidateVariable, *, reward: float, conflict: float,
) -> np.ndarray:
    """Ion-family factor: same neutral identity is rewarded, incompatible identities conflict."""
    output = np.zeros((len(left.candidates), len(right.candidates)), dtype=np.float64)
    for i, a in enumerate(left.candidates):
        for j, b in enumerate(right.candidates):
            if UNKNOWN in (a, b):
                output[i, j] = 0.0
            else:
                output[i, j] = reward if a == b else -abs(conflict)
    return output
