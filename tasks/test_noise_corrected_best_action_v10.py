"""Static V10 trainer-contract checks that run before any GPU work."""
from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return Path(__file__).with_name(
        "train_noise_corrected_routed_direct.py"
    ).read_text(encoding="utf-8")


def test_v10_is_a_registered_direct_contract_not_a_teacher_route() -> None:
    source = _source()
    assert '"best_action_v10_safe_exact"' in source
    assert '"optimizer_restoration_scope": "safe_exact_corrective"' in source
    assert '"calibration_corrective_branch_mode": "full_action_view"' in source
    assert '"arm_invariant_targeted_calibration": True' in source


def test_v10_enforces_formula_disjoint_inner_holdout_before_admission() -> None:
    source = _source()
    split = source.index("inner_holdout_key = (")
    admission = source.index("_select_corrective_admission(", split)
    assert 'actions.query_formula.astype(str)' in source[split:admission]
    assert 'args.inner_holdout_unit == "formula"' in source[split:admission]
    assert "formula-disjoint inner holdout leaked formulas" in source
    assert "clean_training_pool" in source
    assert "inner-held formula leaked into clean training pool" in source


def test_v10_common_calibration_is_masked_only_after_calibration() -> None:
    source = _source()
    call = source.index(") = calibrate_v3(")
    mask = source.index("calibrated_v3_branch_scale =", call)
    train = source.index("train_v3_epochs(", mask)
    assert call < mask < train
    assert "disabled_training_corrective_branches_have_zero_effective_scale" in source


def test_v10_materializes_safe_baseline_and_fails_closed() -> None:
    trainer = _source()
    injector = Path(__file__).with_name(
        "noise_action_injector_v1.py"
    ).read_text(encoding="utf-8")
    assert "ActionInjectorV1(ActionInjectorV1Config(" in trainer
    assert "safe_exact_injector.prepare(" in trainer
    assert "safe_exact_injector.step_and_inject_(" in trainer
    assert "counterfactual_baseline_updates" in injector
    assert "V10 hard protective floor was not enforced" in injector
    assert "V10 exact optimizer corrective fraction was" in injector
    assert "V10 hard-safe update exceeds the registered" in injector


def test_development_panel_persists_formula_for_cluster_bootstrap() -> None:
    source = _source()
    function = source[source.index("def _paired_subset("):source.index("def main()")]
    assert '"query_formula"' in function
    assert "graph.query_formula" in function


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_best_action_v10] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
