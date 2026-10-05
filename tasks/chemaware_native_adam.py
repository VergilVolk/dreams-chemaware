"""Dependency-light native Adam continuation contracts for ChemAware."""
from __future__ import annotations

import torch


def adam_state_steps(state: dict) -> dict[int, int]:
    body = state.get("state")
    if not isinstance(body, dict) or not body:
        raise RuntimeError("native Adam state has no registered parameters")
    output: dict[int, int] = {}
    for parameter, slot in body.items():
        if not isinstance(slot, dict) or "step" not in slot:
            raise RuntimeError("native Adam parameter state lacks a step counter")
        step = slot["step"]
        output[int(parameter)] = int(step.item() if torch.is_tensor(step) else step)
    return output


def restore_native_adam_state(
    optimizer: torch.optim.Optimizer,
    state: dict,
    expected_lr: float,
    expected_weight_decay: float,
) -> dict[int, int]:
    if type(optimizer) is not torch.optim.Adam:
        raise RuntimeError("ChemAware continuation optimizer is not native Adam")
    optimizer.load_state_dict(state)
    groups = optimizer.param_groups
    if (
        len(groups) != 1
        or abs(float(groups[0]["lr"]) - float(expected_lr)) > 1e-15
        or abs(float(groups[0]["weight_decay"]) - float(expected_weight_decay)) > 1e-15
    ):
        raise RuntimeError("restored ChemAware Adam hyperparameters drifted")
    return adam_state_steps(optimizer.state_dict())
