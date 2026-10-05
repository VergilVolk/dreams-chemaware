"""CPU-only contracts for V9 branch separation and honest signal gates."""
from __future__ import annotations

from pathlib import Path

from noise_corrected_signal_v9 import (
    corrective_branch_policy,
    minimum_gate_passed,
    restoration_coverage_gate,
)


def test_branch_policy_is_an_exact_one_boundary_ablation() -> None:
    full = corrective_branch_policy("full_action_view")
    scalar = corrective_branch_policy("scalar_transfer_only")
    assert full.transfer and scalar.transfer
    assert full.payload and full.consistency
    assert not scalar.payload and not scalar.consistency


def test_minimum_gate_tolerates_only_numerical_roundoff() -> None:
    assert minimum_gate_passed(0.24999998, 0.25)
    assert not minimum_gate_passed(0.249, 0.25)
    assert not minimum_gate_passed(float("nan"), 0.25)


def test_restoration_coverage_depends_on_behavior_not_contract_name() -> None:
    assert not restoration_coverage_gate(
        active_arm=True, materialized=True,
        observed_fraction=0.8802, minimum_fraction=0.90,
    )
    assert restoration_coverage_gate(
        active_arm=True, materialized=True,
        observed_fraction=0.89999998, minimum_fraction=0.90,
    )
    assert restoration_coverage_gate(
        active_arm=True, materialized=False,
        observed_fraction=0.0, minimum_fraction=0.90,
    )


def test_trainer_uses_v9_behavioral_contracts() -> None:
    source = Path(__file__).with_name(
        "train_noise_corrected_routed_direct.py"
    ).read_text(encoding="utf-8")
    assert "restoration_target_coverage_gate = restoration_coverage_gate(" in source
    assert '"corrective_branch_mode": args.corrective_branch_mode' in source
    assert 'branch_scale["payload"] * corr_components["payload"]' in source
    assert 'branch_scale["consistency"]' in source


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_signal_v9] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
