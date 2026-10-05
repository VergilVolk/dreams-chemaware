"""Bounded action-panel and lossless selector-frontier utilities."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from noise_corrected_action_routing_v3 import select_diverse_routed_actions_v3


def select_initial_e8_error_boundary_queries(
    eligible_queries: np.ndarray,
    clean_ranks: np.ndarray,
    clean_margins: np.ndarray,
    identity_keys: np.ndarray,
    *,
    boundary_multiplier: float,
    minimum_boundary_correct: int,
    maximum_boundary_correct: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep every current-E8 error plus identity-diverse vulnerable correct rows.

    Corrective actions are only possible on current-E8 errors.  Harmful and
    robustness actions are most informative on the smallest-margin correct
    boundaries.  Selecting those rows from the outer-training geometry avoids
    the old full-all-query combinatorial expansion without looking at held
    formulas or historical action outcomes.
    """
    query = np.asarray(eligible_queries, dtype=np.int64)
    rank = np.asarray(clean_ranks, dtype=np.int64)
    margin = np.asarray(clean_margins, dtype=np.float64)
    identity = np.asarray(identity_keys, dtype=str)
    if not (query.ndim == rank.ndim == margin.ndim == identity.ndim == 1):
        raise ValueError("initial-E8 panel inputs must be one-dimensional")
    if not (len(query) == len(rank) == len(margin) == len(identity)) or not len(query):
        raise ValueError("initial-E8 panel inputs must be aligned and non-empty")
    if (
        len(np.unique(query)) != len(query)
        or np.any(rank < 1)
        or not np.isfinite(margin).all()
        or np.any(identity == "")
    ):
        raise ValueError("initial-E8 panel inputs are malformed")
    if (
        boundary_multiplier < 0
        or minimum_boundary_correct < 0
        or maximum_boundary_correct < minimum_boundary_correct
    ):
        raise ValueError("initial-E8 boundary selection limits are invalid")

    error_mask = rank > 1
    errors = query[error_mask]
    correct = pd.DataFrame({
        "query_index": query[~error_mask],
        "margin": margin[~error_mask],
        "identity": identity[~error_mask],
    }).sort_values(["identity", "margin", "query_index"], kind="stable")
    # Before taking a second spectrum from an identity, expose the first
    # vulnerable spectrum from every other identity.  The final ordering still
    # prioritizes smaller margins within each multiplicity round.
    correct["identity_round"] = correct.groupby("identity", sort=False).cumcount()
    correct = correct.sort_values(
        ["identity_round", "margin", "query_index"], kind="stable",
    )
    requested_boundary = max(
        int(minimum_boundary_correct),
        int(math.ceil(float(boundary_multiplier) * len(errors))),
    )
    requested_boundary = min(int(maximum_boundary_correct), requested_boundary)
    boundary = correct.head(requested_boundary).query_index.to_numpy(np.int64)
    selected = np.sort(np.concatenate([errors, boundary]).astype(np.int64, copy=False))
    if not set(map(int, errors)).issubset(set(map(int, selected))):
        raise RuntimeError("current-E8 error query was lost from the action panel")
    return selected, {
        "eligible_outer_train_queries": int(len(query)),
        "initial_E8_error_queries": int(len(errors)),
        "selected_boundary_correct_queries": int(len(boundary)),
        "selected_action_queries": int(len(selected)),
        "boundary_multiplier": float(boundary_multiplier),
        "minimum_boundary_correct": int(minimum_boundary_correct),
        "maximum_boundary_correct": int(maximum_boundary_correct),
        "boundary_maximum_initial_E8_margin": (
            float(correct.head(len(boundary)).margin.max()) if len(boundary) else None
        ),
        "selection_uses_historical_action_outcomes": False,
        "selection_uses_outer_held_formulas": False,
    }


def lossless_selector_frontier(
    actions: pd.DataFrame,
    *,
    maximum_corrective_per_query: int,
    maximum_harmful_per_query: int,
    maximum_robust_per_query: int,
) -> pd.DataFrame:
    """Retain the per-mechanism prefix that can survive global v3 selection."""
    routed = select_diverse_routed_actions_v3(
        actions,
        maximum_corrective_per_query=maximum_corrective_per_query,
        maximum_harmful_per_query=maximum_harmful_per_query,
        maximum_robust_per_query=maximum_robust_per_query,
    )
    keep = routed.selected_corrective | routed.selected_harmful | routed.selected_robust
    return routed.loc[keep].copy()
