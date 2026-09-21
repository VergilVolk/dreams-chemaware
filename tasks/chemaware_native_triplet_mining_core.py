"""Pure NumPy contracts for high-coverage ChemAware triplet mining.

The chemistry is used only to decide which native DreaMS triplets are safe to
show to the optimizer.  Correct chemistry and all semantic-null chemistries
are treated exchangeably during recipe selection; each event needs 2-of-3
null consensus and the correct arm must create more valid directional events
than any null arm under the same thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MiningRecipe:
    min_support: float
    min_support_delta: float
    min_region: float
    min_neighbor: float
    min_advantage_delta: float
    max_official_rank_fraction: float


REQUIRED_METRICS = (
    "action_top_fraction",
    "action_largest_region_fraction",
    "action_same_neighbor_fraction",
    "action_best_advantage_over_baseline",
    "candidate_official_rank_fraction",
)


class RecipeSelectionError(RuntimeError):
    def __init__(self, message: str, rows: list[dict[str, object]]):
        super().__init__(message)
        self.rows = rows


def recipe_dict(recipe: MiningRecipe) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(recipe).items()}


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).tolist()
    )


def validate_evidence(evidence: Mapping[str, np.ndarray]) -> dict[str, int]:
    required = {
        "query", "formula", "identity", "valid", "proposed_candidate",
        "baseline_candidate", "baseline_rank", "benefit", "harmful",
        "arm_names", "metric_names", "arm_metric", "action_count",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise KeyError(f"triplet evidence fields absent: {missing}")
    query = np.asarray(evidence["query"])
    valid = np.asarray(evidence["valid"], dtype=bool)
    arm_metric = np.asarray(evidence["arm_metric"])
    arm_names = _decode(evidence["arm_names"])
    metric_names = _decode(evidence["metric_names"])
    if len(set(query.tolist())) != len(query):
        raise ValueError("triplet evidence repeats query rows")
    if arm_names[0] != "correct" or len(arm_names) != 4:
        raise ValueError("exactly one correct chemistry plus three semantic nulls are required")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("triplet evidence arm names are not unique")
    if int(np.asarray(evidence["action_count"])) < 2:
        raise ValueError("triplet evidence action grid is too small")
    absent_metrics = sorted(set(REQUIRED_METRICS) - set(metric_names))
    if absent_metrics:
        raise KeyError(f"triplet evidence metrics absent: {absent_metrics}")
    if arm_metric.shape[:3] != (len(arm_names), *valid.shape):
        raise ValueError("arm metric tensor is not arm x query x candidate")
    if arm_metric.shape[-1] != len(metric_names):
        raise ValueError("arm metric names/tensor differ")
    official_rank_index = metric_names.index("candidate_official_rank_fraction")
    if not np.allclose(
        arm_metric[..., official_rank_index],
        arm_metric[0:1, ..., official_rank_index],
        rtol=0.0, atol=0.0,
    ):
        raise ValueError("official candidate hardness unexpectedly depends on chemistry arm")
    for name in ("proposed_candidate", "benefit", "harmful"):
        if np.asarray(evidence[name]).shape != valid.shape:
            raise ValueError(f"candidate field has wrong shape: {name}")
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)
    baseline = np.asarray(evidence["baseline_candidate"], dtype=np.int64)
    for row in range(len(valid)):
        candidates = proposed[row, valid[row]]
        if np.any(candidates < 0) or len(np.unique(candidates)) != len(candidates):
            raise ValueError("valid proposed candidates must be unique and nonnegative")
        if np.any(candidates == baseline[row]):
            raise ValueError("candidate evidence includes the official baseline as a proposal")
    if np.any(np.asarray(evidence["benefit"], dtype=bool) & np.asarray(evidence["harmful"], dtype=bool)):
        raise ValueError("a candidate event cannot be both correction and protection")
    if np.any(~np.isfinite(arm_metric[:, valid])):
        raise ValueError("valid action evidence contains non-finite values")
    return {
        "queries": int(len(query)),
        "candidate_events": int(valid.sum()),
        "arms": int(len(arm_names)),
        "metrics": int(len(metric_names)),
    }


def _metric(evidence: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    names = _decode(evidence["metric_names"])
    return np.asarray(evidence["arm_metric"], dtype=np.float64)[..., names.index(name)]


def directional_masks(
    evidence: Mapping[str, np.ndarray], recipe: MiningRecipe, center: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return correction and chemistry-protection candidate masks for one arm."""
    validate_evidence(evidence)
    arms = len(np.asarray(evidence["arm_names"]))
    if center < 0 or center >= arms:
        raise IndexError("chemistry arm index is invalid")
    others = np.asarray([index for index in range(arms) if index != center], dtype=np.int64)
    valid = np.asarray(evidence["valid"], dtype=bool)
    support = _metric(evidence, "action_top_fraction")
    region = _metric(evidence, "action_largest_region_fraction")
    neighbor = _metric(evidence, "action_same_neighbor_fraction")
    advantage = _metric(evidence, "action_best_advantage_over_baseline")
    official_rank = _metric(evidence, "candidate_official_rank_fraction")[0]

    # With three semantic nulls, the median is a 2-of-3 consensus comparator.
    # It gives useful coverage without accepting a one-null accident.
    support_consensus = np.median(support[others], axis=0)
    region_consensus = np.median(region[others], axis=0)
    advantage_consensus = np.median(advantage[others], axis=0)
    support_high_delta = support[center] - support_consensus
    region_high_delta = region[center] - region_consensus
    advantage_high_delta = advantage[center] - advantage_consensus
    support_low_delta = support_consensus - support[center]
    advantage_low_delta = advantage_consensus - advantage[center]

    # Correction requires the correct arm to be a stable, uniquely stronger
    # promoter of the true candidate than a 2-of-3 semantic-null consensus.
    correction = (
        valid
        & np.asarray(evidence["benefit"], dtype=bool)
        & (support[center] >= recipe.min_support)
        & (region[center] >= recipe.min_region)
        & (neighbor[center] >= recipe.min_neighbor)
        & (support_high_delta >= recipe.min_support_delta)
        & (region_high_delta >= 0.0)
        & (advantage_high_delta >= recipe.min_advantage_delta)
    )

    # Protection is not a harmful correct-rule promotion.  It is a hard false
    # challenger promoted by a 2-of-3 null consensus but specifically rejected by the
    # center arm.  This supplies a chemically justified negative candidate.
    protection = (
        valid
        & np.asarray(evidence["harmful"], dtype=bool)
        & (support_consensus >= recipe.min_support)
        & (support_low_delta >= recipe.min_support_delta)
        & (advantage_low_delta >= recipe.min_advantage_delta)
        & (official_rank <= recipe.max_official_rank_fraction)
    )
    diagnostics = {
        "support_high_delta": support_high_delta,
        "advantage_high_delta": advantage_high_delta,
        "support_low_delta": support_low_delta,
        "advantage_low_delta": advantage_low_delta,
    }
    return correction, protection, diagnostics


def summarize_masks(
    evidence: Mapping[str, np.ndarray], correction: np.ndarray, protection: np.ndarray,
) -> dict[str, int]:
    selected = np.asarray(correction, dtype=bool) | np.asarray(protection, dtype=bool)
    query_active = selected.any(axis=1)
    formulas = np.asarray(evidence["formula"]).astype(str)
    return {
        "candidate_directions": int(selected.sum()),
        "correction_directions": int(np.asarray(correction).sum()),
        "protection_directions": int(np.asarray(protection).sum()),
        "anchor_queries": int(query_active.sum()),
        "unique_formulas": int(len(np.unique(formulas[query_active]))),
    }


def recipe_grid(evidence: Mapping[str, np.ndarray]) -> list[MiningRecipe]:
    """Small, auditable grid; no learned scorer and no hidden-label fitting."""
    validate_evidence(evidence)
    advantage = _metric(evidence, "action_best_advantage_over_baseline")
    valid = np.asarray(evidence["valid"], dtype=bool)
    other_max = np.max(advantage[1:], axis=0)
    positive = (advantage[0] - other_max)[valid]
    positive = positive[np.isfinite(positive) & (positive >= 0.0)]
    advantage_grid = [0.0]
    if len(positive):
        advantage_grid.extend(float(x) for x in np.quantile(positive, (0.25, 0.50)))
    advantage_grid = sorted(set(advantage_grid))
    action_step = 1.0 / float(int(np.asarray(evidence["action_count"])))
    return [
        MiningRecipe(*values)
        for values in product(
            (action_step, 2 * action_step, 4 * action_step),
            (action_step, 2 * action_step),
            (action_step, 2 * action_step),
            (0.50, 0.75),
            advantage_grid,
            (0.25, 0.50),
        )
    ]


def select_recipe(
    evidence: Mapping[str, np.ndarray], *, min_anchor_queries: int = 128,
    min_formulas: int = 64, min_specificity_ratio: float = 1.20,
    recipes: Sequence[MiningRecipe] | None = None,
) -> tuple[MiningRecipe, dict[str, object], list[dict[str, object]]]:
    """Freeze one recipe on role 2 by correct-vs-null event specificity."""
    audit = validate_evidence(evidence)
    candidates = list(recipes) if recipes is not None else recipe_grid(evidence)
    rows: list[dict[str, object]] = []
    for recipe in candidates:
        arm_summaries = []
        for center in range(audit["arms"]):
            correction, protection, _ = directional_masks(evidence, recipe, center)
            arm_summaries.append(summarize_masks(evidence, correction, protection))
        correct = arm_summaries[0]
        maximum_null_directions = max(row["candidate_directions"] for row in arm_summaries[1:])
        ratio = (correct["candidate_directions"] + 1.0) / (maximum_null_directions + 1.0)
        surplus = correct["candidate_directions"] - maximum_null_directions
        admissible = (
            correct["anchor_queries"] >= int(min_anchor_queries)
            and correct["unique_formulas"] >= int(min_formulas)
            and surplus > 0
            and ratio >= float(min_specificity_ratio)
        )
        rows.append({
            "recipe": recipe_dict(recipe),
            "correct": correct,
            "nulls": arm_summaries[1:],
            "maximum_null_candidate_directions": int(maximum_null_directions),
            "specific_candidate_surplus": int(surplus),
            "specificity_ratio": float(ratio),
            "admissible": bool(admissible),
        })
    admissible_rows = [row for row in rows if row["admissible"]]
    if not admissible_rows:
        raise RecipeSelectionError(
            "no high-coverage chemically specific triplet recipe passed role 2", rows,
        )
    selected = max(
        admissible_rows,
        key=lambda row: (
            int(row["specific_candidate_surplus"]),
            int(row["correct"]["unique_formulas"]),
            int(row["correct"]["anchor_queries"]),
            float(row["specificity_ratio"]),
        ),
    )
    recipe = MiningRecipe(**selected["recipe"])
    return recipe, selected, rows
