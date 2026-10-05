"""Pure helpers for ChemAware development-only policy disagreement audits."""
from __future__ import annotations

import numpy as np


def action_outcome(baseline_rank: np.ndarray, proposed_rank: np.ndarray) -> dict[str, object]:
    baseline = np.asarray(baseline_rank, dtype=np.int64)
    proposed = np.asarray(proposed_rank, dtype=np.int64)
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("rank arrays are not aligned")
    corrected = (baseline > 1) & (proposed == 1)
    introduced = (baseline == 1) & (proposed > 1)
    return {
        "queries": int(len(baseline)),
        "baseline_errors": int(np.sum(baseline > 1)),
        "corrected_at_1": int(np.sum(corrected)),
        "introduced_at_1": int(np.sum(introduced)),
        "risk_utility_at_1": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "delta_recall1": float(np.mean(proposed == 1) - np.mean(baseline == 1)),
        "delta_mrr": float(np.mean(1.0 / proposed - 1.0 / baseline)),
        "rank_improved": int(np.sum(proposed < baseline)),
        "rank_worsened": int(np.sum(proposed > baseline)),
    }


def masked_outcome(
    mask: np.ndarray, baseline_rank: np.ndarray, proposed_rank: np.ndarray,
) -> dict[str, object]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != np.asarray(baseline_rank).shape:
        raise ValueError("mask and rank arrays are not aligned")
    if not np.any(selected):
        return {
            "queries": 0, "baseline_errors": 0, "corrected_at_1": 0,
            "introduced_at_1": 0, "risk_utility_at_1": 0,
            "delta_recall1": None, "delta_mrr": None,
            "rank_improved": 0, "rank_worsened": 0,
        }
    return action_outcome(
        np.asarray(baseline_rank)[selected], np.asarray(proposed_rank)[selected],
    )


def disagreement_partition(left_slot: np.ndarray, right_slot: np.ndarray) -> dict[str, np.ndarray]:
    left = np.asarray(left_slot, dtype=np.int64)
    right = np.asarray(right_slot, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("policy action slots are not aligned")
    left_active = left >= 0
    right_active = right >= 0
    return {
        "both_abstain": ~left_active & ~right_active,
        "left_only": left_active & ~right_active,
        "right_only": ~left_active & right_active,
        "both_same_action": left_active & right_active & (left == right),
        "both_different_action": left_active & right_active & (left != right),
    }


def pair_audit(
    left_name: str, right_name: str,
    baseline_rank: np.ndarray,
    left_rank: np.ndarray, right_rank: np.ndarray,
    left_slot: np.ndarray, right_slot: np.ndarray,
) -> dict[str, object]:
    groups = disagreement_partition(left_slot, right_slot)
    output: dict[str, object] = {
        "left": left_name,
        "right": right_name,
        "left_overall": action_outcome(baseline_rank, left_rank),
        "right_overall": action_outcome(baseline_rank, right_rank),
        "partitions": {},
    }
    for name, mask in groups.items():
        output["partitions"][name] = {
            "queries": int(np.sum(mask)),
            "left": masked_outcome(mask, baseline_rank, left_rank),
            "right": masked_outcome(mask, baseline_rank, right_rank),
            "left_top1_minus_right": float(
                np.mean(np.asarray(left_rank)[mask] == 1)
                - np.mean(np.asarray(right_rank)[mask] == 1)
            ) if np.any(mask) else None,
        }
    return output


def selected_utility_summary(slot: np.ndarray, utility: np.ndarray) -> dict[str, object]:
    selected = np.asarray(slot, dtype=np.int64)
    values = np.asarray(utility, dtype=np.float64)
    if values.ndim != 2 or selected.shape != (len(values),):
        raise ValueError("selected slots and utility table are not aligned")
    active = selected >= 0
    chosen = values[np.flatnonzero(active), selected[active]]
    quantiles = np.quantile(chosen, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) if len(chosen) else []
    return {
        "selected": int(np.sum(active)),
        "quantiles_0_10_25_50_75_90_100": [float(value) for value in quantiles],
    }


def intervention_labels(baseline_rank: np.ndarray, proposed_rank: np.ndarray) -> np.ndarray:
    """Encode action outcome as +1 correction, -1 introduction, or 0 otherwise."""
    baseline = np.asarray(baseline_rank, dtype=np.int64)
    proposed = np.asarray(proposed_rank, dtype=np.int64)
    if baseline.shape != proposed.shape:
        raise ValueError("rank arrays are not aligned")
    output = np.zeros(len(baseline), dtype=np.int8)
    output[(baseline > 1) & (proposed == 1)] = 1
    output[(baseline == 1) & (proposed > 1)] = -1
    return output


def arbitrate_two_policies(
    baseline_rank: np.ndarray,
    primary_rank: np.ndarray, primary_slot: np.ndarray,
    fallback_rank: np.ndarray, fallback_slot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the primary action when present, otherwise the fallback action.

    This is a deployment-visible deterministic composition.  It never inspects
    labels or action outcomes; ``baseline_rank`` is carried only to represent
    abstentions in an evaluation ledger.
    """
    baseline = np.asarray(baseline_rank, dtype=np.int16)
    first_rank = np.asarray(primary_rank, dtype=np.int16)
    first_slot = np.asarray(primary_slot, dtype=np.int16)
    second_rank = np.asarray(fallback_rank, dtype=np.int16)
    second_slot = np.asarray(fallback_slot, dtype=np.int16)
    if not (
        baseline.shape == first_rank.shape == first_slot.shape
        == second_rank.shape == second_slot.shape
    ):
        raise ValueError("policy arbitration arrays are not aligned")
    first_active = first_slot >= 0
    second_active = second_slot >= 0
    use_second = ~first_active & second_active
    rank = baseline.copy()
    slot = np.full(len(baseline), -1, dtype=np.int16)
    source = np.zeros(len(baseline), dtype=np.int8)
    rank[first_active] = first_rank[first_active]
    slot[first_active] = first_slot[first_active]
    source[first_active] = 1
    rank[use_second] = second_rank[use_second]
    slot[use_second] = second_slot[use_second]
    source[use_second] = 2
    return rank, slot, source


def arbitrate_exclusive_fallback(
    baseline_rank: np.ndarray,
    primary_rank: np.ndarray, primary_slot: np.ndarray,
    fallback_rank: np.ndarray, fallback_slot: np.ndarray,
    control_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use fallback only when primary and every counterfactual control abstain."""
    baseline = np.asarray(baseline_rank, dtype=np.int16)
    first_rank = np.asarray(primary_rank, dtype=np.int16)
    first_slot = np.asarray(primary_slot, dtype=np.int16)
    second_rank = np.asarray(fallback_rank, dtype=np.int16)
    second_slot = np.asarray(fallback_slot, dtype=np.int16)
    controls = np.asarray(control_slots, dtype=np.int16)
    if controls.ndim != 2 or controls.shape[0] != len(baseline) or controls.shape[1] < 1:
        raise ValueError("counterfactual control slots must be query by control")
    if not (
        baseline.shape == first_rank.shape == first_slot.shape
        == second_rank.shape == second_slot.shape
    ):
        raise ValueError("exclusive arbitration arrays are not aligned")
    use_primary = first_slot >= 0
    exclusive = (second_slot >= 0) & np.all(controls < 0, axis=1)
    use_fallback = ~use_primary & exclusive
    rank = baseline.copy()
    slot = np.full(len(baseline), -1, dtype=np.int16)
    source = np.zeros(len(baseline), dtype=np.int8)
    rank[use_primary] = first_rank[use_primary]
    slot[use_primary] = first_slot[use_primary]
    source[use_primary] = 1
    rank[use_fallback] = second_rank[use_fallback]
    slot[use_fallback] = second_slot[use_fallback]
    source[use_fallback] = 2
    return rank, slot, source, exclusive


def formula_cluster_ci(
    formula: np.ndarray, value: np.ndarray, *, draws: int, seed: int,
) -> list[float]:
    labels = np.asarray(formula, dtype=str)
    values = np.asarray(value, dtype=np.float64)
    if labels.shape != values.shape or labels.ndim != 1 or draws <= 0:
        raise ValueError("formula bootstrap inputs are invalid")
    unique, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        selected = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        result[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(result, (0.025, 0.975))]
