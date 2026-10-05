"""CPU-only safety contracts for no-distillation ChemAware fine-tuning."""
from __future__ import annotations

import numpy as np
import torch


def gradient_norm(values: list[torch.Tensor | None]) -> float:
    return float(np.sqrt(sum(
        float(torch.sum(value.detach().float() ** 2))
        for value in values if value is not None
    )))


def gradient_dot(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> float:
    if len(left) != len(right):
        raise ValueError("gradient vectors do not align")
    return float(sum(
        float(torch.sum(a.detach().float() * b.detach().float()))
        for a, b in zip(left, right) if a is not None and b is not None
    ))


def virtual_adamw_descent_updates(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Evaluate one AdamW descent update without mutating optimizer state.

    This is the dependency-minimal form of the Noise direct-v3 audit.  It
    mirrors AdamW's parameter-dtype arithmetic so the virtual full update can
    be checked against the subsequent real optimizer step.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("optimizer counterfactual currently requires AdamW")
    if len(parameters) != len(gradients):
        raise ValueError("virtual AdamW parameters and gradients must align")
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if set(groups) != {id(parameter) for parameter in parameters}:
        raise RuntimeError("virtual AdamW did not receive the complete parameter set")
    updates: list[torch.Tensor | None] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            updates.append(None); continue
        group = groups[id(parameter)]
        beta1, beta2 = map(float, group["betas"])
        if group.get("differentiable", False):
            raise RuntimeError("differentiable AdamW is unsupported")
        value = gradient.detach()
        if group.get("maximize", False):
            value = -value
        state = optimizer.state.get(parameter, {})
        raw_step = state.get("step", 0)
        step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        step += 1
        average = state.get("exp_avg")
        square = state.get("exp_avg_sq")
        average = torch.zeros_like(parameter) if average is None else average.detach().clone()
        square = torch.zeros_like(parameter) if square is None else square.detach().clone()
        average.lerp_(value, 1.0 - beta1)
        square.mul_(beta2).addcmul_(value, value.conj(), value=1.0 - beta2)
        if group.get("amsgrad", False):
            maximum = state.get("max_exp_avg_sq")
            maximum = torch.zeros_like(parameter) if maximum is None else maximum.detach()
            denominator_square = torch.maximum(maximum, square)
        else:
            denominator_square = square
        denominator = denominator_square.sqrt() / np.sqrt(1.0 - beta2 ** step)
        denominator.add_(float(group["eps"]))
        after = parameter.detach().clone()
        after.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        after.addcdiv_(
            average, denominator,
            value=-(float(group["lr"]) / (1.0 - beta1 ** step)),
        )
        updates.append(parameter.detach() - after)
    return updates


def validate_action_role_matrix(
    role_code: np.ndarray,
    selected_setting: int,
    action_fold: np.ndarray,
    discovery_folds: tuple[int, ...],
    confirmation_fold: int,
    admitted_codes: tuple[int, ...],
) -> None:
    """Validate evaluated roles and the action bank's -1 unseen sentinel."""
    role_code = np.asarray(role_code)
    action_fold = np.asarray(action_fold)
    if (
        role_code.ndim != 2 or action_fold.ndim != 1
        or role_code.shape[1] != len(action_fold)
        or not 0 <= selected_setting < role_code.shape[0]
    ):
        raise RuntimeError("action bank role matrix or selected setting is invalid")
    discovery = np.isin(action_fold, discovery_folds)
    for setting in range(role_code.shape[0]):
        evaluated = discovery.copy()
        if setting == selected_setting:
            evaluated |= action_fold == confirmation_fold
        expected_unseen = ~evaluated
        actual_unseen = role_code[setting] == -1
        if not np.array_equal(actual_unseen, expected_unseen):
            raise RuntimeError("action bank role -1 sentinel does not match evaluated folds")
        if not np.isin(role_code[setting, evaluated], admitted_codes).all():
            raise RuntimeError("action bank evaluated role contains an unknown code")


def assert_no_distillation_objective(
    *,
    action_kind: str,
    transfer_target: str,
    lambda_consistency: float,
    lambda_margin_floor: float,
    lambda_preserve: float,
    lambda_peak_contrast: float,
    allow_clean_regularizers: bool = False,
) -> None:
    """Reject teacher-transfer losses while optionally allowing clean safety anchors.

    ``lambda_margin_floor`` and ``lambda_preserve`` compare the current clean
    encoder with its own frozen official initialization.  They do not transfer
    an ICEBERG score, action embedding, or candidate-conditioned target and are
    therefore valid in the guarded direct objective.
    """
    if action_kind != "differential":
        raise ValueError("direct dual-view training requires observed-peak differential actions")
    if transfer_target != "symmetric":
        raise ValueError("direct dual-view training forbids frozen action embedding targets")
    forbidden = {
        "lambda_consistency": lambda_consistency,
        "lambda_peak_contrast": lambda_peak_contrast,
    }
    if not allow_clean_regularizers:
        forbidden.update({
            "lambda_margin_floor": lambda_margin_floor,
            "lambda_preserve": lambda_preserve,
        })
    elif float(lambda_margin_floor) < 0 or float(lambda_preserve) < 0:
        raise ValueError("clean safety regularizer weights must be non-negative")
    active = {name: value for name, value in forbidden.items() if float(value) != 0.0}
    if active:
        raise ValueError(f"direct dual-view training forbids auxiliary target losses: {active}")


def qualified_action_positions(
    selected_queries: np.ndarray,
    bank_selected_queries: np.ndarray,
    eligible: np.ndarray,
    formula_fold: np.ndarray,
    inner_fold: int,
    outer_fold: int,
) -> np.ndarray:
    """Return bank-qualified rows while proving evaluation-fold exclusion."""
    selected_queries = np.asarray(selected_queries, dtype=np.int64)
    bank_selected_queries = np.asarray(bank_selected_queries, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    formula_fold = np.asarray(formula_fold, dtype=np.int16)
    lengths = {len(selected_queries), len(bank_selected_queries), len(eligible), len(formula_fold)}
    if len(lengths) != 1 or not np.array_equal(selected_queries, bank_selected_queries):
        raise ValueError("action bank does not align exactly with the teacher query ledger")
    positions = np.flatnonzero(eligible)
    if not len(positions):
        raise ValueError("qualified action bank contains no eligible actions")
    leaked = positions[np.isin(formula_fold[positions], (inner_fold, outer_fold))]
    if len(leaked):
        raise ValueError("qualified action bank leaks into embedding evaluation folds")
    return positions.astype(np.int64, copy=False)


def projected_guarded_auxiliary(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    maximum_auxiliary_ratio: float = 0.25,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Add a bounded action gradient without opposing clean retrieval.

    This implements a one-sided PCGrad-style BioAware/Noise safety rule: project away
    a conflicting auxiliary component and cap the surviving action norm as a
    fraction of the primary retrieval/safety norm. Missing gradients become
    aligned zeros, so the result maps one-to-one to trainable parameters.
    """
    if len(primary) != len(parameters) or len(auxiliary) != len(parameters):
        raise ValueError("gradient tuples do not align with trainable parameters")
    if not 0 < float(maximum_auxiliary_ratio) <= 1:
        raise ValueError("maximum_auxiliary_ratio must be in (0, 1]")
    primary_value = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, primary)
    ]
    auxiliary_value = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, auxiliary)
    ]
    primary_sq = sum(torch.sum(value.float() ** 2) for value in primary_value)
    auxiliary_sq = sum(torch.sum(value.float() ** 2) for value in auxiliary_value)
    dot = sum(
        torch.sum(left.float() * right.float())
        for left, right in zip(primary_value, auxiliary_value)
    )
    primary_norm = torch.sqrt(primary_sq)
    auxiliary_norm = torch.sqrt(auxiliary_sq)
    if not bool(torch.isfinite(primary_norm)) or not bool(torch.isfinite(auxiliary_norm)) or not bool(torch.isfinite(dot)):
        raise RuntimeError("non-finite primary/action gradient geometry")
    if float(primary_norm.detach()) <= 0:
        raise RuntimeError("primary retrieval gradient is zero")
    if float(auxiliary_norm.detach()) <= 0:
        raise RuntimeError("chemical action gradient is zero")
    raw_cosine = dot / (primary_norm * auxiliary_norm)
    conflict = bool(float(dot.detach()) <= 0)
    projected = [value.clone() for value in auxiliary_value]
    if conflict:
        coefficient = dot / primary_sq
        projected = [
            action - coefficient.to(action.dtype) * clean
            for action, clean in zip(projected, primary_value)
        ]
    projected_norm = torch.sqrt(sum(torch.sum(value.float() ** 2) for value in projected))
    cap = primary_norm * float(maximum_auxiliary_ratio)
    capped = bool(float(projected_norm.detach()) > float(cap.detach()))
    scale = torch.clamp(cap / torch.clamp(projected_norm, min=1e-12), max=1.0)
    safe = [value * scale.to(value.dtype) for value in projected]
    safe_norm = torch.sqrt(sum(torch.sum(value.float() ** 2) for value in safe))
    safe_dot = sum(
        torch.sum(left.float() * right.float())
        for left, right in zip(primary_value, safe)
    )
    if not bool(torch.isfinite(safe_norm)) or not bool(torch.isfinite(safe_dot)):
        raise RuntimeError("non-finite projected action gradient geometry")
    total = [clean + action for clean, action in zip(primary_value, safe)]
    return total, {
        "primary_norm": float(primary_norm.detach()),
        "auxiliary_raw_norm": float(auxiliary_norm.detach()),
        "auxiliary_raw_cosine": float(raw_cosine.detach()),
        "auxiliary_conflict_projected": conflict,
        "auxiliary_norm_capped": capped,
        "auxiliary_safe_norm": float(safe_norm.detach()),
        "auxiliary_safe_to_primary_ratio": float((safe_norm / primary_norm).detach()),
        "auxiliary_safe_dot_primary": float(safe_dot.detach()),
    }
