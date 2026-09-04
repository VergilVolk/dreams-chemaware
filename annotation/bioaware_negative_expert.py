"""Frozen negative-ion BioAware network expert inference.

This module consumes candidate-level network features already constructed from
sample-level MS1 nodes, identity-heldout seeds, and raw-MS2 edge validation. It
does not fit a model and does not accept phenotype or truth columns as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_INPUT_COLUMNS = {
    "truth_candidate_id", "truth_formula", "is_positive", "baseline_correct",
    "corrected", "introduced", "phenotype", "group", "case_control",
}
REQUIRED_BASE_COLUMNS = {"query_id", "candidate_id", "spectral_score"}


@dataclass(frozen=True)
class FrozenNegativeBioAwareExpert:
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    model_coef: np.ndarray
    maximum_dreams_top1_top2_gap: float
    minimum_pairwise_proposal_probability: float
    requires_raw_step0_edge_validation: bool
    scope: str

    @classmethod
    def load(cls, path: str | Path) -> "FrozenNegativeBioAwareExpert":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "bioaware_metdna3_negative_network_expert_frozen":
            raise ValueError("not a frozen negative BioAware artifact")
        features = tuple(map(str, payload["feature_names"]))
        mean = np.asarray(payload["scaler_mean"], dtype=float)
        scale = np.asarray(payload["scaler_scale"], dtype=float)
        coefficient = np.asarray(payload["model_coef"], dtype=float)
        if not features or not (len(features) == len(mean) == len(scale) == len(coefficient)):
            raise ValueError("artifact dimensions do not agree")
        if (~np.isfinite(mean)).any() or (~np.isfinite(scale)).any() or (~np.isfinite(coefficient)).any():
            raise ValueError("artifact contains non-finite parameters")
        if (scale <= 0).any():
            raise ValueError("artifact scaler scale must be positive")
        configuration = payload["configuration"]
        return cls(
            feature_names=features,
            scaler_mean=mean,
            scaler_scale=scale,
            model_coef=coefficient,
            maximum_dreams_top1_top2_gap=float(configuration["maximum_dreams_top1_top2_gap"]),
            minimum_pairwise_proposal_probability=float(configuration["minimum_pairwise_proposal_probability"]),
            requires_raw_step0_edge_validation=bool(configuration["requires_raw_step0_edge_validation"]),
            scope=str(payload["scope"]),
        )

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("candidate feature matrix has the wrong shape")
        if (~np.isfinite(values)).any():
            raise ValueError("candidate feature matrix contains non-finite values")
        return ((values - self.scaler_mean) / self.scaler_scale) @ self.model_coef


def _unique_top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    top = group[np.isclose(group[column], maximum, atol=1e-12, rtol=0)]
    candidate = str(top.sort_values("candidate_id").iloc[0]["candidate_id"])
    return candidate, len(top) == 1


def _top_gap(group: pd.DataFrame) -> float:
    values = np.sort(group["spectral_score"].to_numpy(float))[::-1]
    if len(values) < 2:
        raise ValueError("each query requires at least two candidates")
    return float(values[0] - values[1])


def apply_frozen_negative_bioaware_expert(
    candidates: pd.DataFrame,
    expert: FrozenNegativeBioAwareExpert,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score and safely rerank candidate groups.

    Returns ``(scored_candidates, decisions)``.  The decision table contains no
    correctness labels because inference is outcome-blind.
    """
    forbidden = sorted(FORBIDDEN_INPUT_COLUMNS & set(candidates.columns))
    if forbidden:
        raise ValueError(f"outcome/phenotype columns are forbidden at inference: {forbidden}")
    required = REQUIRED_BASE_COLUMNS | set(expert.feature_names)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate table is missing required columns: {missing}")
    frame = candidates.copy()
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("candidate table contains duplicate query/candidate rows")
    frame["bioaware_network_score"] = expert.score(
        frame[list(expert.feature_names)].to_numpy(float)
    )
    decisions = []
    for query_id, group in frame.groupby("query_id", sort=False):
        if len(group) < 2:
            raise ValueError(f"query {query_id} has fewer than two candidates")
        baseline, baseline_unique = _unique_top(group, "spectral_score")
        proposed, proposal_unique = _unique_top(group, "bioaware_network_score")
        baseline_row = group[group["candidate_id"].astype(str).eq(baseline)].iloc[0]
        proposed_row = group[group["candidate_id"].astype(str).eq(proposed)].iloc[0]
        score_advantage = float(
            proposed_row.bioaware_network_score - baseline_row.bioaware_network_score
        )
        proposal_probability = float(1.0 / (1.0 + np.exp(-score_advantage)))
        gap = _top_gap(group)
        raw_edge_validated = bool(
            float(proposed_row["edge0_complete_fraction"]) > 0
            and float(proposed_row["edge0_bottleneck_mean"]) > 0
        )
        reasons = []
        if not baseline_unique:
            reasons.append("dreams_top_tie")
        if not proposal_unique:
            reasons.append("network_top_tie")
        if proposed == baseline:
            reasons.append("same_top1")
        if gap > expert.maximum_dreams_top1_top2_gap:
            reasons.append("dreams_confident")
        if proposal_probability < expert.minimum_pairwise_proposal_probability:
            reasons.append("network_advantage_below_gate")
        if expert.requires_raw_step0_edge_validation and not raw_edge_validated:
            reasons.append("raw_step0_edge_unvalidated")
        intervene = not reasons
        final = proposed if intervene else baseline
        decisions.append({
            "query_id": query_id,
            "dreams_top_candidate_id": baseline,
            "network_top_candidate_id": proposed,
            "final_candidate_id": final,
            "intervene": intervene,
            "abstention_reasons": "|".join(reasons),
            "dreams_top1_top2_gap": gap,
            "network_proposal_probability": proposal_probability,
            "raw_step0_edge_validated": raw_edge_validated,
        })
    return frame, pd.DataFrame(decisions)
