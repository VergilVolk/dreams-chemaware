"""Pure routing and diversity contracts for corrected-graph noise actions.

The router observes action outcomes only on the outer-training formulas in the
frozen initialization geometry.  It does not create an embedding target.  Its
job is narrower: separate directly useful corrective interventions from risk
examples, then keep several mechanistically distinct actions without turning
recipe multiplicity into optimizer dose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoutingThresholds:
    paired_advantage: float = 0.01
    harm_margin: float = 0.01
    robustness_slack: float = 0.005

    def validate(self) -> None:
        if min(self.paired_advantage, self.harm_margin, self.robustness_slack) < 0:
            raise ValueError("routing thresholds must be non-negative")


def route_action(
    *,
    clean_rank: int,
    clean_margin: float,
    action_rank: int,
    action_margin: float,
    control_margin: float,
    thresholds: RoutingThresholds = RoutingThresholds(),
) -> str:
    """Classify one paired action without converting its margin into a target."""
    thresholds.validate()
    values = np.asarray([clean_margin, action_margin, control_margin], dtype=float)
    if clean_rank < 1 or action_rank < 1 or not np.isfinite(values).all():
        raise ValueError("rank/margin inputs are malformed")
    paired = float(action_margin - control_margin)
    change = float(action_margin - clean_margin)
    introduced = clean_rank == 1 and action_rank > 1
    if (
        introduced
        or change <= -thresholds.harm_margin
        or paired <= -thresholds.paired_advantage
    ):
        return "harmful"
    if (
        clean_rank > 1
        and change >= thresholds.paired_advantage
        and paired >= thresholds.paired_advantage
    ):
        return "corrective"
    if (
        clean_rank == 1
        and action_rank == 1
        and change >= -thresholds.robustness_slack
        and paired >= 0
    ):
        return "robustness_only"
    return "uncertain"


def _round_robin_indices(
    block: pd.DataFrame,
    *,
    maximum: int,
    group_columns: tuple[str, ...],
    score_column: str,
    descending: bool,
) -> list[int]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    missing = (set(group_columns) | {score_column, "action_id"}) - set(block.columns)
    if missing:
        raise KeyError(f"diversity block misses {sorted(missing)}")
    grouped: dict[tuple[str, ...], list[int]] = {}
    key_arg: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for key, values in block.groupby(key_arg, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ordered = values.sort_values(
            [score_column, "action_id"],
            ascending=[not descending, True], kind="stable",
        )
        grouped[tuple(map(str, key))] = list(map(int, ordered.index))
    selected: list[int] = []
    cursor = 0
    while len(selected) < min(maximum, len(block)):
        progressed = False
        for key in sorted(grouped):
            values = grouped[key]
            if cursor < len(values):
                selected.append(values[cursor])
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def select_diverse_routed_actions(
    frame: pd.DataFrame,
    *,
    maximum_corrective_per_query: int = 16,
    maximum_risk_per_query: int = 8,
) -> pd.DataFrame:
    """Keep bounded multi-action corrective and risk panels per query.

    Corrective actions are ordered by their conservative paired gain, while
    harmful actions are ordered by harm magnitude.  Selection round-robins
    source/family strata before taking a second action from any stratum.
    Uncertain actions and robustness-only actions remain auditable but receive
    no corrective/risk optimizer exposure.
    """
    required = {
        "query_index", "action_id", "source", "family", "route",
        "margin_change", "paired_advantage",
    }
    if missing := required - set(frame.columns):
        raise KeyError(f"routed action table misses {sorted(missing)}")
    if maximum_corrective_per_query < 1 or maximum_risk_per_query < 1:
        raise ValueError("per-query action limits must be positive")
    if frame.action_id.astype(str).duplicated().any():
        raise RuntimeError("action IDs must be globally unique before selection")
    output = frame.copy()
    output["conservative_gain"] = np.minimum(
        output.margin_change.to_numpy(float),
        output.paired_advantage.to_numpy(float),
    )
    output["harm_strength"] = np.maximum(
        -output.margin_change.to_numpy(float),
        -output.paired_advantage.to_numpy(float),
    )
    output["selected_corrective"] = False
    output["selected_risk"] = False
    for _, block in output.groupby("query_index", sort=True):
        corrective = block.loc[block.route.astype(str).eq("corrective")]
        if len(corrective):
            indices = _round_robin_indices(
                corrective, maximum=maximum_corrective_per_query,
                group_columns=("source", "family"),
                score_column="conservative_gain", descending=True,
            )
            output.loc[indices, "selected_corrective"] = True
        harmful = block.loc[block.route.astype(str).eq("harmful")]
        if len(harmful):
            indices = _round_robin_indices(
                harmful, maximum=maximum_risk_per_query,
                group_columns=("source", "family"),
                score_column="harm_strength", descending=True,
            )
            output.loc[indices, "selected_risk"] = True
    if (output.selected_corrective & ~output.route.eq("corrective")).any():
        raise RuntimeError("non-corrective action entered the corrective panel")
    if (output.selected_risk & ~output.route.eq("harmful")).any():
        raise RuntimeError("non-harmful action entered the risk panel")
    return output
