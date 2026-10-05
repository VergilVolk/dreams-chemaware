"""Formula-disjoint cross-calibration for selective ChemAware intervention."""
from __future__ import annotations

import hashlib

import numpy as np


def stable_formula_roles(formula: np.ndarray, roles: int, seed: int) -> np.ndarray:
    if roles < 2:
        raise ValueError("at least two calibration roles are required")
    return np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{str(value)}".encode()).digest()[:8], "little"
        ) % roles
        for value in np.asarray(formula, dtype=str)
    ], dtype=np.int16)


def ranks_at_threshold(
    baseline_rank: np.ndarray, proposal_rank: np.ndarray,
    utility: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline_rank, dtype=np.int16)
    proposal = np.asarray(proposal_rank, dtype=np.int16)
    values = np.asarray(utility, dtype=np.float64)
    if proposal.shape != values.shape or proposal.shape[0] != len(baseline):
        raise ValueError("candidate tables are not aligned")
    selected = np.argmax(values, axis=1).astype(np.int16)
    best = values[np.arange(len(values)), selected]
    active = np.isfinite(best) & (best >= float(threshold))
    rank = baseline.copy()
    rank[active] = proposal[np.flatnonzero(active), selected[active]]
    selected[~active] = -1
    return rank, selected, best


def reward(baseline_rank: np.ndarray, rank: np.ndarray, risk_penalty: float) -> np.ndarray:
    baseline = np.asarray(baseline_rank, dtype=np.int64)
    proposed = np.asarray(rank, dtype=np.int64)
    if baseline.shape != proposed.shape:
        raise ValueError("rank arrays are not aligned")
    return (
        ((baseline > 1) & (proposed == 1)).astype(np.float64)
        - float(risk_penalty) * ((baseline == 1) & (proposed > 1)).astype(np.float64)
    )


def summarize(
    baseline_rank: np.ndarray, rank: np.ndarray, selected: np.ndarray,
    formula: np.ndarray, risk_penalty: float,
) -> dict[str, object]:
    baseline = np.asarray(baseline_rank, dtype=np.int64)
    proposed = np.asarray(rank, dtype=np.int64)
    active = np.asarray(selected, dtype=np.int64) >= 0
    corrected = (baseline > 1) & (proposed == 1)
    introduced = (baseline == 1) & (proposed > 1)
    return {
        "queries": int(len(baseline)),
        "selected": int(np.sum(active)),
        "selected_formulas": int(len(np.unique(np.asarray(formula, dtype=str)[active]))),
        "corrected_at_1": int(np.sum(corrected)),
        "introduced_at_1": int(np.sum(introduced)),
        "risk_utility_at_1": float(
            np.sum(corrected) - float(risk_penalty) * np.sum(introduced)
        ),
        "delta_recall1": float(np.mean(proposed == 1) - np.mean(baseline == 1)),
        "delta_mrr": float(np.mean(1.0 / proposed - 1.0 / baseline)),
    }


def threshold_grid(utility: np.ndarray) -> np.ndarray:
    best = np.max(np.asarray(utility, dtype=np.float64), axis=1)
    finite = best[np.isfinite(best)]
    if not len(finite):
        raise ValueError("no finite candidate utility")
    return np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, 101)),
        np.nextafter(np.max(finite), np.inf),
    ])


def select_threshold(
    baseline_rank: np.ndarray, proposal_rank: np.ndarray,
    utilities: dict[str, np.ndarray], formula: np.ndarray,
    *, risk_penalty: float, min_selected_formulas: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if "correct" not in utilities or len(utilities) < 2:
        raise ValueError("correct and at least one matched control are required")
    controls = tuple(name for name in utilities if name != "correct")
    rows = []
    for threshold in threshold_grid(utilities["correct"]):
        arm = {}
        arm_reward = {}
        for name, utility in utilities.items():
            rank, selected, _ = ranks_at_threshold(
                baseline_rank, proposal_rank, utility, float(threshold),
            )
            arm[name] = summarize(
                baseline_rank, rank, selected, formula, risk_penalty,
            )
            arm_reward[name] = reward(baseline_rank, rank, risk_penalty)
        correct = arm["correct"]
        advantages = {
            name: float(np.sum(arm_reward["correct"] - arm_reward[name]))
            for name in controls
        }
        admissible = bool(
            (correct["selected"] == 0 or correct["selected_formulas"] >= min_selected_formulas)
            and correct["risk_utility_at_1"] > 0
            and correct["corrected_at_1"] > risk_penalty * correct["introduced_at_1"]
            and min(advantages.values()) > 0
        )
        rows.append({
            "threshold": float(threshold),
            "admissible": admissible,
            "correct": correct,
            "control_risk_advantage": advantages,
            "minimum_control_risk_advantage": float(min(advantages.values())),
        })
    eligible = [row for row in rows if row["admissible"]]
    if not eligible:
        finite = np.max(utilities["correct"], axis=1)
        threshold = np.nextafter(float(np.max(finite[np.isfinite(finite)])), np.inf)
        return {
            "threshold": threshold,
            "no_admissible_threshold": True,
            "correct": {
                "selected": 0, "selected_formulas": 0, "corrected_at_1": 0,
                "introduced_at_1": 0, "risk_utility_at_1": 0.0,
                "delta_recall1": 0.0, "delta_mrr": 0.0,
            },
        }, rows
    selected = max(
        eligible,
        key=lambda row: (
            row["minimum_control_risk_advantage"],
            row["correct"]["risk_utility_at_1"],
            row["correct"]["corrected_at_1"],
            row["threshold"],
        ),
    )
    return selected, rows


def formula_cluster_ci(
    formula: np.ndarray, values: np.ndarray, *, draws: int, seed: int,
) -> list[float]:
    labels = np.asarray(formula, dtype=str)
    value = np.asarray(values, dtype=np.float64)
    if labels.shape != value.shape or draws <= 0:
        raise ValueError("formula bootstrap inputs are invalid")
    unique, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=value)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        selected = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        result[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(x) for x in np.quantile(result, (0.025, 0.975))]


def crosscalibrate(
    baseline_rank: np.ndarray, proposal_rank: np.ndarray,
    utilities: dict[str, np.ndarray], formula: np.ndarray,
    *, roles: int, seed: int, risk_penalty: float,
    min_selected_formulas: int, bootstrap_draws: int,
) -> dict[str, object]:
    formula = np.asarray(formula, dtype=str)
    role = stable_formula_roles(formula, roles, seed)
    controls = tuple(name for name in utilities if name != "correct")
    oof_rank = {name: np.asarray(baseline_rank, dtype=np.int16).copy() for name in utilities}
    oof_selected = {name: np.full(len(formula), -1, dtype=np.int16) for name in utilities}
    folds = []
    thresholds = []
    for held in range(roles):
        train = role != held
        test = role == held
        selected, grid = select_threshold(
            np.asarray(baseline_rank)[train], np.asarray(proposal_rank)[train],
            {name: value[train] for name, value in utilities.items()}, formula[train],
            risk_penalty=risk_penalty,
            min_selected_formulas=min_selected_formulas,
        )
        threshold = float(selected["threshold"])
        thresholds.append(threshold)
        held_report = {}
        for name, utility in utilities.items():
            rank, action, _ = ranks_at_threshold(
                np.asarray(baseline_rank)[test], np.asarray(proposal_rank)[test],
                utility[test], threshold,
            )
            oof_rank[name][test] = rank
            oof_selected[name][test] = action
            held_report[name] = summarize(
                np.asarray(baseline_rank)[test], rank, action, formula[test], risk_penalty,
            )
        folds.append({
            "held_role": int(held), "threshold": threshold,
            "selection": selected, "selection_grid_size": int(len(grid)),
            "held": held_report,
        })
    oof = {}
    correct_reward = reward(baseline_rank, oof_rank["correct"], risk_penalty)
    for name in utilities:
        values = reward(baseline_rank, oof_rank[name], risk_penalty)
        oof[name] = {
            **summarize(
                baseline_rank, oof_rank[name], oof_selected[name], formula, risk_penalty,
            ),
            "formula_cluster_ci_mean_risk_reward": formula_cluster_ci(
                formula, values, draws=bootstrap_draws, seed=seed + 100 + len(oof),
            ),
        }
        if name != "correct":
            oof[name]["correct_minus_control_formula_ci_mean_risk_reward"] = formula_cluster_ci(
                formula, correct_reward - values,
                draws=bootstrap_draws, seed=seed + 200 + len(oof),
            )
    return {
        "roles": int(roles),
        "role_counts": {str(i): int(np.sum(role == i)) for i in range(roles)},
        "folds": folds,
        "oof": oof,
        "deployment_threshold_rule": "maximum cross-calibrated threshold",
        "deployment_threshold": float(np.max(thresholds)),
        "thresholds": [float(x) for x in thresholds],
        "oof_gate": {
            "correct_risk_ci_positive": oof["correct"]["formula_cluster_ci_mean_risk_reward"][0] > 0,
            "corrected_exceeds_risk_weighted_introduced": (
                oof["correct"]["corrected_at_1"]
                > risk_penalty * oof["correct"]["introduced_at_1"]
            ),
            **{
                f"correct_beats_{name}_risk_ci":
                oof[name]["correct_minus_control_formula_ci_mean_risk_reward"][0] > 0
                for name in controls
            },
        },
    }
