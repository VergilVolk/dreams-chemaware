"""Static fail-closed checks for the Hybrid V2 one-GPU submission bundle."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_wires_corrected_e4_base_and_semantic_residual_separately() -> None:
    source = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8"
    )
    required = (
        "load_corrected_e4_base_actions",
        "train_e4_base_semantic_v2_epochs",
        "e4_base_examples=e4_base_action_examples",
        "semantic_examples=action_examples",
        "materialized_query_local_semantic_loss",
        "lambda_clean_rank=args.lambda_clean_rank",
        "injector.capture_semantic_corrective_",
        "base_loss, base_log = direct_action_loss",
        "e4_args.action_selection = \"fixed\"",
        "injector.step_and_inject_",
        "semantic_source_families",
        "v2_later_source_family_equal_effective_dose",
        "objective_reference_by_row is not training_anchor_by_row",
        "for epoch in (() if signal_preserving_v2 else range(1, epochs + 1))",
        "registered_formal_action_bank_configuration_verified",
        "dafffea08745e59a5af4f37324858bb0472283d9d2b9d30a8bf411d6ee6f7d1d",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"Hybrid V2 trainer wiring is incomplete: {missing}")
    semantic_module = (
        ROOT / "tasks/noise_e4_signal_preserving_hybrid_v2.py"
    ).read_text(encoding="utf-8")
    for token in (
        '"semantic_clean_boundary_rank_active": True',
        '"semantic_satisfied_action_rank_gradient_gated": True',
        '"semantic_symmetric_clean_action_consistency": True',
    ):
        if token not in semantic_module:
            raise AssertionError(f"Hybrid V2 semantic repair is missing: {token}")
    validator = (
        ROOT / "tasks/validate_noise_final_e4a_direct_augmentation.py"
    ).read_text(encoding="utf-8")
    summary = (
        ROOT / "tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py"
    ).read_text(encoding="utf-8")
    contract = "v2_e8_initialization_is_floor_and_preservation_target"
    if contract not in source or contract not in validator or contract not in summary:
        raise AssertionError("V2 E8 anchor is not enforced end to end")
    for contract in (
        "v2_later_semantic_clean_boundary_rank_active",
        "v2_later_satisfied_action_rank_gradient_gated",
        "v2_later_symmetric_clean_action_consistency_preserved",
    ):
        if contract not in source or contract not in validator:
            raise AssertionError(f"V2 clean-boundary repair is not fail-closed: {contract}")
    v2_branch = validator.split(
        'elif injection_mode == "e4_base_semantic_v2":', 1
    )[1].split("for key, value in v2_expected.items()", 1)[0]
    if '"materialized_satisfied_action_rank_gradient_gated": True' not in v2_branch:
        raise AssertionError("V2 validator still expects ungated satisfied actions")


def test_sbatch_uses_exactly_one_gpu_without_manual_memory() -> None:
    source = (ROOT / "tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch").read_text(
        encoding="utf-8"
    )
    if "#SBATCH --gpus=1" not in source or "#SBATCH --gpus=2" in source:
        raise AssertionError("Hybrid V2 must request exactly one GPU")
    forbidden = ("#SBATCH --mem", "#SBATCH --mem-per", "--mem=", "--mem-per-gpu")
    if any(value in source for value in forbidden):
        raise AssertionError("Hybrid V2 manually specifies memory")
    if "GPU_ONE" in source or "PID_TARGETED" in source or "PID_SHUFFLED" in source:
        raise AssertionError("one-GPU Hybrid V2 retained parallel-worker state")
    targeted = source.index('run_arm "$GPU_ZERO" targeted')
    shuffled = source.index('run_arm "$GPU_ZERO" shuffled')
    if targeted >= shuffled:
        raise AssertionError("one-GPU causal arms are not sequentially ordered")
    required = (
        "--initial-student-checkpoint \"$INITIAL\"",
        "--e4-base-action-dir \"$E4_BASE\"",
        "--materialized-injection-mode e4_base_semantic_v2",
        "--optimizer-boundary-mode signal_preserving_v2",
        "--materialized-action-arm \"$ARM\"",
        "targeted",
        "shuffled",
        "test_noise_e4_signal_preserving_hybrid_v2.py",
        "summarize_noise_e4_signal_preserving_hybrid_v2.py",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"Hybrid V2 sbatch is incomplete: {missing}")


def test_sbatch_binds_the_exact_source_manifest() -> None:
    sbatch = (
        ROOT / "tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    manifest_path = ROOT / "tasks/noise_e4_signal_preserving_hybrid_v2_source_manifest.sha256"
    observed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    match = re.search(
        r'^EXPECTED_SOURCE_MANIFEST_SHA256="([0-9a-f]{64})"$',
        sbatch,
        flags=re.MULTILINE,
    )
    if match is None or match.group(1) != observed:
        raise AssertionError(
            "Hybrid V2 SBATCH does not bind the exact current source manifest: "
            f"observed={observed}"
        )


def test_frozen_artifact_hashes_are_bound_in_summary() -> None:
    source = (ROOT / "tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py").read_text(
        encoding="utf-8"
    )
    sbatch = (
        ROOT / "tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    for digest in (
        "8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af",
        "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349",
        "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa",
        "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a",
        "9db644dc2592bb6eff2677779b731f6ee1f3f946f7df5868886b31c375cf3dfa",
        "1ce8c412f9ee0ef7a4ba4758554313c85f123280d31e73004cb62b76887c7be5",
    ):
        if digest not in source:
            raise AssertionError(f"summary does not bind frozen artifact {digest}")
        if digest not in sbatch:
            raise AssertionError(f"SBATCH does not bind frozen artifact {digest}")


def test_shuffled_receipt_matches_the_imported_four_column_producer() -> None:
    trainer = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8"
    )
    sbatch = (
        ROOT / "tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    validator = (
        ROOT / "tasks/validate_noise_final_e4a_direct_augmentation.py"
    ).read_text(encoding="utf-8")
    summary = (
        ROOT / "tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py"
    ).read_text(encoding="utf-8")
    strategy = (
        "supervision_source_family_exact_recipe_matched_"
        "cross_query_cyclic_shuffle"
    )
    columns = '["supervision_kind", "source", "family", "recipe_id"]'
    if "from noise_corrected_shuffled_control_v3 import" not in trainer:
        raise AssertionError("trainer does not import the frozen shuffled producer")
    if "test_noise_corrected_action_routing_v3.py" not in sbatch:
        raise AssertionError("SBATCH does not execute the shuffled-producer tests")
    for name, source in (("validator", validator), ("summary", summary)):
        if strategy not in source or columns not in source:
            raise AssertionError(f"{name} does not bind the four-column receipt")
        if "recipe_semantics_preserved" in source:
            raise AssertionError(f"{name} requires a receipt field the producer omits")


def test_update_norm_cap_uses_only_the_independent_e4_reference() -> None:
    source = (ROOT / "tasks/noise_e4_signal_preserving_hybrid_v2.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "composition.maximum_group_update_norm_ratio_to_original",
        "signal-preserving update exceeds combined-update norm cap",
        "maximum_update_norm_ratio_to_actual_combined",
    )
    if any(value in source for value in forbidden):
        raise AssertionError("V2 retained the cancellation-unstable combined-update cap")
    required = (
        "maximum_update_norm_ratio_to_independent_historical_e4",
        "final_to_shadow_e4_update_norm_ratio_by_group",
        "per-group signal-preserving update exceeds the absolute E4 norm cap",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"V2 lost the independent-E4 explosion guard: {missing}")


def test_summary_uses_familywise_formula_ci_and_honest_e4_base_coverage() -> None:
    summary = (
        ROOT / "tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py"
    ).read_text(encoding="utf-8")
    trainer = (
        ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
    ).read_text(encoding="utf-8")
    validator = (
        ROOT / "tasks/validate_noise_final_e4a_direct_augmentation.py"
    ).read_text(encoding="utf-8")
    required_summary = (
        "FORMULA_CI_PRIMARY_COMPARISONS = 3",
        "alpha=FORMULA_CI_FAMILYWISE_ALPHA / FORMULA_CI_PRIMARY_COMPARISONS",
        '"method": "Bonferroni"',
        '"near_paired_top1_vs_matched_shuffled"',
        '"near_risk_net_lambda2_vs_shuffled_positive"',
        '"e4_base_exposure"',
        "source_family_semantic_dose_equal_within_identity",
    )
    missing = [token for token in required_summary if token not in summary]
    if missing:
        raise AssertionError(f"V2 summary lost simultaneous evaluation: {missing}")
    coverage_contract = "e4_base_bank_is_validated_supplier_not_full_row_coverage"
    if coverage_contract not in trainer or coverage_contract not in validator:
        raise AssertionError("E4 base supplier is still misreported as full row coverage")


def main() -> None:
    tests = (
        test_trainer_wires_corrected_e4_base_and_semantic_residual_separately,
        test_sbatch_uses_exactly_one_gpu_without_manual_memory,
        test_sbatch_binds_the_exact_source_manifest,
        test_frozen_artifact_hashes_are_bound_in_summary,
        test_shuffled_receipt_matches_the_imported_four_column_producer,
        test_update_norm_cap_uses_only_the_independent_e4_reference,
        test_summary_uses_familywise_formula_ci_and_honest_e4_base_coverage,
    )
    for test in tests:
        test()
    print(f"[test_noise_e4_signal_preserving_hybrid_v2_sbatch] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
