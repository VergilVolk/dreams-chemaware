"""Pure helpers for cross-view event-calibrated Rankmax transfer."""
from __future__ import annotations

import numpy as np
import torch


def boundary_coordinate_gradients(
    fixed: np.ndarray,
    product: np.ndarray,
    valid: np.ndarray,
    positive: np.ndarray,
    rule_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the local parent-margin gradient and active top negative.

    For parent score ``s_j(w)=s_j(1)+beta <p_j,w-1>``, the Top-1 margin
    gradient is ``beta * (p_positive - p_top_negative)``.  Keeping this
    calculation explicit makes it possible to distinguish supervision noise
    from an intrinsically conflicting shared diagonal metric.
    """
    fixed = np.asarray(fixed, dtype=np.float64)
    product = np.asarray(product, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    positive = np.asarray(positive, dtype=np.int64)
    if fixed.ndim != 2 or product.ndim != 3 or valid.shape != fixed.shape:
        raise ValueError("boundary-gradient tensors have invalid shapes")
    if product.shape[:2] != fixed.shape or len(positive) != len(fixed):
        raise ValueError("boundary-gradient tensors must align")
    if not np.isfinite(rule_beta) or rule_beta <= 0:
        raise ValueError("rule_beta must be positive and finite")
    row = np.arange(len(fixed))
    if np.any(positive < 0) or np.any(positive >= fixed.shape[1]):
        raise ValueError("positive indices are out of range")
    negative_score = np.where(valid, fixed, -np.inf)
    negative_score[row, positive] = -np.inf
    top_negative = np.argmax(negative_score, axis=1).astype(np.int64)
    if np.any(~np.isfinite(negative_score[row, top_negative])):
        raise ValueError("every query needs at least one valid negative")
    gradient = float(rule_beta) * (
        product[row, positive] - product[row, top_negative]
    )
    return gradient.astype(np.float64), top_negative


def boundary_symmetric_operator_gradients(
    query: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    rule_beta: float,
) -> np.ndarray:
    """Vectorize the symmetric full-metric margin gradient for each event."""
    query = np.asarray(query, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if query.ndim != 2 or positive.shape != query.shape or negative.shape != query.shape:
        raise ValueError("operator-gradient rule vectors must align")
    if not np.isfinite(rule_beta) or rule_beta <= 0:
        raise ValueError("rule_beta must be positive and finite")
    difference = positive - negative
    outer = query[:, :, None] * difference[:, None, :]
    symmetric = 0.5 * (outer + np.swapaxes(outer, 1, 2))
    return (float(rule_beta) * symmetric).reshape(len(query), -1)


def gradient_coherence_report(
    gradient: np.ndarray,
    identity: np.ndarray,
    active: np.ndarray,
) -> dict[str, float | int | list[float]]:
    """Summarize whether identity-balanced event gradients admit one direction.

    Query gradients are averaged within identity first.  Reported leave-one-
    identity-out cosine avoids the optimistic self-alignment of comparing an
    event with an aggregate that already contains that event.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    active = np.asarray(active, dtype=bool)
    if gradient.ndim != 2 or len(identity) != len(gradient) or active.shape != identity.shape:
        raise ValueError("gradient-coherence arrays must align")
    values = np.unique(identity[active])
    identity_gradient = []
    for value in values:
        vector = np.mean(gradient[active & (identity == value)], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0 and np.isfinite(norm):
            identity_gradient.append(vector / norm)
    if len(identity_gradient) < 2:
        raise ValueError("gradient coherence requires at least two nonzero identities")
    matrix = np.stack(identity_gradient)
    cosine = matrix @ matrix.T
    upper = cosine[np.triu_indices(len(matrix), 1)]
    total = np.sum(matrix, axis=0)
    loo = []
    for vector in matrix:
        other = total - vector
        denominator = float(np.linalg.norm(other))
        loo.append(float(vector @ other / denominator) if denominator > 0 else 0.0)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular ** 2
    return {
        "queries": int(np.sum(active)),
        "identities": int(len(matrix)),
        "mean_pairwise_cosine": float(np.mean(upper)),
        "median_pairwise_cosine": float(np.median(upper)),
        "negative_pair_fraction": float(np.mean(upper < 0)),
        "mean_leave_one_identity_out_cosine": float(np.mean(loo)),
        "median_leave_one_identity_out_cosine": float(np.median(loo)),
        "negative_leave_one_identity_out_fraction": float(np.mean(np.asarray(loo) < 0)),
        "normalized_resultant_length": float(np.linalg.norm(np.mean(matrix, axis=0))),
        "top_singular_energy_fraction": float(energy[0] / np.sum(energy)),
        "top5_singular_energy_fraction": float(np.sum(energy[:5]) / np.sum(energy)),
        "leave_one_identity_out_cosine_quantiles": [
            float(value) for value in np.quantile(loo, (0.0, 0.25, 0.5, 0.75, 1.0))
        ],
    }


def gradient_transfer_report(
    gradient: np.ndarray,
    identity: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int | list[float]]:
    """Measure whether one source-domain direction helps unseen target identities."""
    gradient = np.asarray(gradient, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    source = np.asarray(source, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if gradient.ndim != 2 or any(len(value) != len(gradient) for value in (identity, source, target)):
        raise ValueError("gradient-transfer arrays must align")
    if set(identity[source]) & set(identity[target]):
        raise ValueError("gradient-transfer identities must be disjoint")

    def identity_units(mask: np.ndarray) -> np.ndarray:
        output = []
        for value in np.unique(identity[mask]):
            vector = np.mean(gradient[mask & (identity == value)], axis=0)
            norm = float(np.linalg.norm(vector))
            if norm > 0 and np.isfinite(norm):
                output.append(vector / norm)
        return np.stack(output) if output else np.empty((0, gradient.shape[1]))

    source_unit = identity_units(source)
    target_unit = identity_units(target)
    if not len(source_unit) or not len(target_unit):
        raise ValueError("gradient transfer requires nonzero source and target identities")
    direction = np.mean(source_unit, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("source aggregate gradient is zero")
    cosine = target_unit @ (direction / norm)
    return {
        "source_queries": int(np.sum(source)),
        "source_identities": int(len(source_unit)),
        "target_queries": int(np.sum(target)),
        "target_identities": int(len(target_unit)),
        "mean_target_cosine": float(np.mean(cosine)),
        "median_target_cosine": float(np.median(cosine)),
        "negative_target_fraction": float(np.mean(cosine < 0)),
        "target_cosine_quantiles": [
            float(value) for value in np.quantile(cosine, (0.0, 0.25, 0.5, 0.75, 1.0))
        ],
    }


def identity_risk_route(
    identity: np.ndarray,
    scope: np.ndarray,
    base_rank: np.ndarray,
    correct_selected_identity: np.ndarray,
    control_selected_identity: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Select identities whose correct-arm Top-1 risk is positive and control-specific."""
    identity = np.asarray(identity).astype(str)
    scope = np.asarray(scope, dtype=bool)
    base_rank = np.asarray(base_rank, dtype=np.int64)
    correct_selected_identity = np.asarray(correct_selected_identity).astype(str)
    control_selected_identity = np.asarray(control_selected_identity).astype(str)
    if any(len(value) != len(identity) for value in (
        scope, base_rank, correct_selected_identity, control_selected_identity,
    )):
        raise ValueError("identity-risk arrays must align")
    selected: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    for value in np.unique(identity[scope]):
        index = np.flatnonzero(scope & (identity == value))
        correct_fix = int(np.sum((base_rank[index] > 1) & (correct_selected_identity[index] == value)))
        correct_harm = int(np.sum((base_rank[index] == 1) & (correct_selected_identity[index] != "") & (correct_selected_identity[index] != value)))
        control_fix = int(np.sum((base_rank[index] > 1) & (control_selected_identity[index] == value)))
        control_harm = int(np.sum((base_rank[index] == 1) & (control_selected_identity[index] != "") & (control_selected_identity[index] != value)))
        correct_risk = correct_fix - 2 * correct_harm
        control_risk = control_fix - 2 * control_harm
        audit[value] = {
            "correct_fix": correct_fix, "correct_harm": correct_harm,
            "correct_risk": correct_risk, "control_fix": control_fix,
            "control_harm": control_harm, "control_risk": control_risk,
        }
        if correct_risk > 0 and correct_risk > control_risk:
            selected.append(value)
    return np.asarray(sorted(selected), dtype="U14"), audit


def identity_balanced_weights(identity: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Give each active identity equal total dose and normalize total dose to one."""
    identity = np.asarray(identity).astype(str)
    active = np.asarray(active, dtype=bool)
    if active.shape != identity.shape:
        raise ValueError("identity and active mask must align")
    output = np.zeros(len(identity), dtype=np.float32)
    values = np.unique(identity[active])
    if not len(values):
        return output
    for value in values:
        index = np.flatnonzero(active & (identity == value))
        output[index] = 1.0 / (len(values) * len(index))
    if not np.isclose(float(output.sum()), 1.0, atol=1e-6):
        raise RuntimeError("identity-balanced dose does not sum to one")
    return output


def matched_error_identities(
    selected_identity: np.ndarray,
    identity: np.ndarray,
    formula: np.ndarray,
    scope: np.ndarray,
    error: np.ndarray,
    margin: np.ndarray,
) -> np.ndarray:
    """Greedily match error identities by repeat count and median boundary depth."""
    selected_identity = np.asarray(selected_identity).astype(str)
    identity = np.asarray(identity).astype(str)
    formula = np.asarray(formula).astype(str)
    scope = np.asarray(scope, dtype=bool)
    error = np.asarray(error, dtype=bool)
    margin = np.asarray(margin, dtype=np.float64)
    if any(len(value) != len(identity) for value in (formula, scope, error, margin)):
        raise ValueError("matched-route arrays must align")
    selected = set(map(str, selected_identity))
    pool = [
        value for value in np.unique(identity[scope & error])
        if value not in selected
    ]
    if len(pool) < len(selected):
        raise RuntimeError("insufficient error identities for matched control")

    def signature(value: str) -> tuple[float, float, str]:
        index = np.flatnonzero(scope & (identity == value))
        errors = index[error[index]]
        return (
            float(np.median(margin[errors])), float(len(index)), str(formula[index[0]]),
        )

    signatures = {value: signature(value) for value in set(pool) | selected}
    margin_scale = max(float(np.std([signatures[value][0] for value in pool])), 1e-6)
    repeat_scale = max(float(np.std([signatures[value][1] for value in pool])), 1.0)
    available = set(pool); chosen: list[str] = []
    for value in sorted(selected):
        target_margin, target_repeat, target_formula = signatures[value]
        candidates = [item for item in available if signatures[item][2] != target_formula]
        if not candidates:
            candidates = list(available)
        winner = min(candidates, key=lambda item: (
            abs(signatures[item][0] - target_margin) / margin_scale
            + abs(signatures[item][1] - target_repeat) / repeat_scale,
            item,
        ))
        chosen.append(winner); available.remove(winner)
    if len(chosen) != len(selected) or len(set(chosen)) != len(chosen):
        raise RuntimeError("matched identity control is not one-to-one")
    return np.asarray(chosen, dtype="U14")


def rankmax_top1_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    sample_weight: torch.Tensor,
    target_margin: float,
) -> torch.Tensor:
    """Adaptive Top-1 loss: only negatives violating the desired boundary contribute."""
    if positive.ndim != 1 or negative.ndim != 2:
        raise ValueError("Rankmax expects one positive and a negative matrix")
    if negative.shape[0] != len(positive) or sample_weight.shape != positive.shape:
        raise ValueError("Rankmax tensors must align")
    if target_margin <= 0 or torch.any(sample_weight < 0):
        raise ValueError("Rankmax margin and weights are invalid")
    violation = torch.relu(negative - positive[:, None] + float(target_margin))
    each = torch.log1p(torch.sum(violation, dim=1) / float(target_margin))
    denominator = torch.sum(sample_weight)
    if float(denominator.detach()) <= 0:
        return torch.sum(positive * 0.0)
    return torch.sum(each * sample_weight) / denominator
