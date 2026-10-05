"""Contracts for a truly no-distillation ChemAware objective."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chemaware_direct_training_core import (
    assert_no_distillation_objective,
    gradient_dot,
    gradient_norm,
    projected_guarded_auxiliary,
    qualified_action_positions,
    validate_action_role_matrix,
    virtual_adamw_descent_updates,
)


def expect_failure(**kwargs) -> None:
    try:
        assert_no_distillation_objective(**kwargs)
    except ValueError:
        return
    raise AssertionError("forbidden direct objective was accepted")


def main() -> None:
    clean = dict(
        action_kind="differential",
        transfer_target="symmetric",
        lambda_consistency=0.0,
        lambda_margin_floor=0.0,
        lambda_preserve=0.0,
        lambda_peak_contrast=0.0,
    )
    assert_no_distillation_objective(**clean)
    for field in (
        "lambda_consistency", "lambda_margin_floor", "lambda_preserve", "lambda_peak_contrast"
    ):
        bad = clean | {field: 0.01}
        expect_failure(**bad)
    expect_failure(**(clean | {"transfer_target": "frozen_action"}))

    guarded = clean | {
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 20.0,
        "allow_clean_regularizers": True,
    }
    assert_no_distillation_objective(**guarded)
    expect_failure(**(guarded | {"lambda_consistency": 0.01}))
    expect_failure(**(guarded | {"lambda_preserve": -1.0}))

    selected = np.arange(6)
    positions = qualified_action_positions(
        selected, selected.copy(), np.asarray([1, 0, 1, 0, 0, 0]),
        np.asarray([0, 1, 2, 3, 4, 4]), 3, 4,
    )
    assert positions.tolist() == [0, 2]
    try:
        qualified_action_positions(
            selected, selected.copy(), np.asarray([1, 0, 0, 1, 0, 0]),
            np.asarray([0, 1, 2, 3, 4, 4]), 3, 4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("evaluation-fold action leakage was accepted")

    parameter = torch.nn.Parameter(torch.zeros(2))
    combined, audit = projected_guarded_auxiliary(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-1.0, 2.0]),),
        [parameter],
        maximum_auxiliary_ratio=0.25,
    )
    assert audit["auxiliary_conflict_projected"] is True
    assert audit["auxiliary_norm_capped"] is True
    assert audit["auxiliary_safe_dot_primary"] >= -1e-7
    assert audit["auxiliary_safe_to_primary_ratio"] <= 0.250001
    assert torch.allclose(combined[0], torch.tensor([1.0, 0.25]), atol=1e-6)
    assert abs(gradient_norm([combined[0]]) - float(torch.linalg.vector_norm(combined[0]))) < 1e-7
    assert abs(gradient_dot([combined[0]], [combined[0]]) - float(torch.sum(combined[0] ** 2))) < 1e-7

    adam_parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5], dtype=torch.float64))
    adam = torch.optim.AdamW(
        [adam_parameter], lr=0.01, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.02,
    )
    for gradient in (
        torch.tensor([0.3, -0.4], dtype=torch.float64),
        torch.tensor([-0.2, 0.1], dtype=torch.float64),
    ):
        virtual = virtual_adamw_descent_updates(
            adam, [adam_parameter], [gradient],
        )[0]
        before = adam_parameter.detach().clone()
        adam_parameter.grad = gradient.clone(); adam.step(); adam.zero_grad(set_to_none=True)
        assert virtual is not None
        assert torch.allclose(before - adam_parameter.detach(), virtual, atol=1e-12, rtol=1e-10)
    folds = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)
    roles = np.asarray([
        [0, 1, -1, -1, -1],
        [2, 3, 1, -1, -1],
    ], dtype=np.int8)
    validate_action_role_matrix(roles, 1, folds, (0, 1), 2, (0, 1, 2, 3, 4))
    invalid_roles = roles.copy(); invalid_roles[1, 3] = 0
    try:
        validate_action_role_matrix(invalid_roles, 1, folds, (0, 1), 2, (0, 1, 2, 3, 4))
    except RuntimeError as error:
        assert "sentinel" in str(error)
    else:
        raise AssertionError("evaluated/unseen role mismatch was accepted")
    trainer_source = (Path(__file__).parent / "train_chemaware_iceberg_direct_shared.py").read_text(
        encoding="utf-8",
    )
    assert (
        'None if delta_transfer or near_delta is None else near_delta >= 0'
        in trainer_source
    )
    assert "passed = all(applicable_gates.values())" in trainer_source
    print("no-distillation direct training contracts passed")


if __name__ == "__main__":
    main()
