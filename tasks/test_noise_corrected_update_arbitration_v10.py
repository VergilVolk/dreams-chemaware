"""CPU-only tests for V10 hard-safe exact-dose update composition."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from noise_corrected_update_arbitration_v10 import (
    compose_safe_exact_corrective_updates,
    compose_safe_exact_corrective_updates_by_group,
)


def _norm(values: list[torch.Tensor | None]) -> float:
    return math.sqrt(sum(
        float(torch.sum(value.double() ** 2))
        for value in values if value is not None
    ))


def test_unsafe_original_update_is_replaced_by_a_true_hard_floor() -> None:
    protective = [torch.tensor([1.0, 0.0])]
    baseline = [torch.tensor([0.20, 1.00])]
    combined = [torch.tensor([0.10, 1.20])]
    report, safe_baseline, floor_enforced, _ = (
        compose_safe_exact_corrective_updates(
            combined,
            baseline,
            protective,
            target_attributable_fraction=0.25,
            minimum_protective_component_retention=0.90,
        )
    )
    assert floor_enforced
    assert report.target_reached
    assert report.risk_component_retention >= 0.90 - 1e-6
    assert safe_baseline[0] is not None
    assert float(safe_baseline[0][0]) >= 0.90 - 1e-6
    attributable = [report.updates[0] - safe_baseline[0]]
    assert math.isclose(
        _norm(attributable) / _norm(report.updates), 0.25,
        rel_tol=0.0, abs_tol=2e-6,
    )


def test_safe_baseline_and_positive_corrective_direction_reach_exact_dose() -> None:
    protective = [torch.tensor([1.0, 0.0, 0.0])]
    baseline = [torch.tensor([0.95, 0.40, 0.00])]
    combined = [torch.tensor([0.95, 0.45, 0.20])]
    report, safe_baseline, floor_enforced, _ = (
        compose_safe_exact_corrective_updates(
            combined,
            baseline,
            protective,
            target_attributable_fraction=0.25,
            minimum_protective_component_retention=0.90,
        )
    )
    assert floor_enforced and report.target_reached
    assert report.risk_component_retention >= 0.90
    attributable = [report.updates[0] - safe_baseline[0]]
    assert math.isclose(
        _norm(attributable) / _norm(report.updates), 0.25,
        rel_tol=0.0, abs_tol=2e-6,
    )


def test_groupwise_composition_matches_head_and_backbone_dose() -> None:
    protective = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    baseline = [
        torch.tensor([0.4, 0.3]),
        torch.tensor([0.2, 0.4]),
    ]
    combined = [
        torch.tensor([0.3, 0.5]),
        torch.tensor([0.4, 0.3]),
    ]
    result = compose_safe_exact_corrective_updates_by_group(
        combined,
        baseline,
        protective,
        {"head": [0], "backbone": [1]},
        target_attributable_fraction=0.25,
        minimum_protective_component_retention=0.90,
    )
    assert result.all_groups_target_reached
    assert result.all_groups_protective_floor_enforced
    assert result.minimum_group_risk_component_retention >= 0.90 - 1e-6
    assert result.maximum_group_attributable_fraction_abs_error <= 2e-6
    for name, positions in {"head": [0], "backbone": [1]}.items():
        updates = [result.updates[position] for position in positions]
        reference = [
            result.counterfactual_baseline_updates[position]
            for position in positions
        ]
        attributable = [
            value - base for value, base in zip(updates, reference)
        ]
        assert math.isclose(
            _norm(attributable) / _norm(updates), 0.25,
            rel_tol=0.0, abs_tol=2e-6,
        ), name


def test_zero_corrective_direction_is_fail_closed_but_still_safe() -> None:
    protective = [torch.tensor([1.0, 0.0])]
    baseline = [torch.tensor([0.2, 1.0])]
    report, safe_baseline, floor_enforced, _ = (
        compose_safe_exact_corrective_updates(
            baseline,
            baseline,
            protective,
            target_attributable_fraction=0.25,
            minimum_protective_component_retention=0.90,
        )
    )
    assert floor_enforced
    assert not report.target_reached
    assert report.final_attributable_fraction == 0.0
    assert float(safe_baseline[0][0]) >= 0.90 - 1e-6
    assert torch.equal(report.updates[0], safe_baseline[0])


def test_trainer_registers_v1_and_uses_its_safe_counterfactual_receipt() -> None:
    source = Path(__file__).with_name(
        "train_noise_corrected_routed_direct.py"
    ).read_text(encoding="utf-8")
    assert '"safe_exact_corrective"' in source
    assert "ActionInjectorV1(ActionInjectorV1Config(" in source
    assert "safe_exact_injector.prepare(" in source
    assert "safe_exact_injector.step_and_inject_(" in source
    assert "compose_safe_exact_corrective_updates_by_group(" not in source


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_noise_corrected_update_arbitration_v10] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
