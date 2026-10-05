"""Static submission contract for the full best-action v7 Slurm job."""
from __future__ import annotations

import hashlib
from pathlib import Path


SBATCH = Path(__file__).with_name("run_noise_corrected_best_v6_full_2gpu.sbatch")
SOURCE_MANIFEST = Path(__file__).with_name("noise_best_v6_source_manifest.sha256")
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_resource_contract_is_exactly_two_gpus_without_manual_memory() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    directives = [line.strip() for line in text.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --gpus=2") == 1
    assert not any("--mem" in line for line in directives)
    assert "#SBATCH --cpus-per-task=16" in directives
    assert "srun " not in text


def test_formal_training_uses_the_repaired_v3_interface_and_no_query_limits() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    common = text.split("COMMON=(", 1)[1].split("\n)", 1)[0]
    assert "--corrective-objective-mode v3_direct" in common
    assert "--direct-contract best_action_v7_corrective_restored" in common
    assert "--corrective-admission strict_top1" in common
    assert "--corrective-margin-floor 0.000005" in common
    assert "--no-arm-invariant-targeted-calibration" in common
    assert "--transfer-target-allocation mass_neutral_monotone" in common
    assert "--materialize-optimizer-update-restoration" in common
    assert "--optimizer-restoration-scope corrective_only" in common
    assert "--reconcile-restored-adamw-first-moment" in common
    assert "--optimizer-restoration-minimum-risk-component-retention 0.90" in common
    assert "--minimum-optimizer-action-attributable-fraction-p10 0.25" in common
    assert "--minimum-optimizer-restoration-target-reached-fraction 0.90" in common
    assert "--continue-after-signal-gate-failure" in common
    assert "--maximum-spectra-per-action-forward 512" in common
    assert "--epochs 4" in common
    assert "--reference-refresh-every-epochs 1" in common
    assert "--bootstrap-resamples 10000" in common
    assert "--development" not in common
    assert "--maximum-corrective-queries" not in common
    assert "--maximum-risk-queries" not in common
    assert "--maximum-robust-queries" not in common
    assert "--maximum-clean-queries" not in common
    assert "--outer-held-eval-queries" not in common
    assert "train_noise_final_e4a_direct_augmentation.py \"${COMMON[@]}\"" not in text


def test_immutable_action_bank_and_three_causal_arms_are_fail_closed() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "noise_corrected_best_v5_canary_fold_0_run_2332784" in text
    assert "SOURCE_RUN=\"${SOURCE_RUN:-" not in text
    for digest in (
        "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
        "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
        "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa",
        "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349",
        "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a",
    ):
        assert digest in text
    assert "len(selected) == 32114" in text
    assert "selected.query_index.nunique() == 3482" in text
    assert "== set(actions.source.astype(str))" in text
    assert "== 297362" in text
    assert 'run_arm_async "$GPU_ZERO" routed_direct &' in text
    assert 'run_arm_async "$GPU_ONE" shuffled_action_control &' in text
    assert 'run_arm "$GPU_ZERO" clean_control' in text
    async_body = text.split("run_arm_async() {", 1)[1].split("\n}", 1)[0]
    assert (
        'exec python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py"'
        in async_body
    )
    assert 'wait -n "$PID_ROUTED" "$PID_SHUFFLED"' not in text
    assert 'worker_is_running()' in text
    assert 'jobs -pr | grep -Fxq -- "$1"' in text
    assert "summarize_noise_corrected_direct_v3_arms.py" in text
    assert '"registered_formal_direct_configuration_verified"' in text
    assert 'SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"' in text
    assert 'cd "$SOURCE_SNAPSHOT"' in text
    assert 'sha256sum "${SNAPSHOT_FILES[@]}" > sha256_manifest.txt' in text
    assert 'sha256sum -c "$SOURCE_MANIFEST"' in text
    assert 'EXPECTED_SOURCE_MANIFEST_SHA256=' in text
    assert 'SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"' in text
    assert 'export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"' in text
    assert 'export PYTHONPATH="$PWD:$PWD/tasks:' not in text
    assert (
        'python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py"'
        in text
    )
    assert (
        'python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_direct_v3_arms.py"'
        in text
    )
    for dependency in (
        "e1_checkpoint_io.py",
        "noise_final_candidate_boundary_core.py",
        "audit_noise_final_positive_guided_matrix.py",
        "audit_noise_final_positive_peak_transfer.py",
        "build_noise_corrected_routed_action_ledger.py",
        "noise_corrected_action_routing.py",
        "audit_noise_peak_gate_candidate_injection.py",
        "noise_final_core.py",
        "noise_final_direct_boundary_v2_core.py",
        "noise_corrected_transfer_objective_v4.py",
        "noise_corrected_action_expansion_v4.py",
        "noise_v3_core.py",
        "noise_corrected_action_routing_v3.py",
        "noise_corrected_directional_update_v4.py",
        "noise_corrected_update_arbitration_v4.py",
        "noise_final_e15_core.py",
        "train_e1_identity.py",
        "train_noise_final_e4a_direct_augmentation.py",
        "train_noise_final_r2_shared_encoder.py",
        "test_noise_corrected_direct_v3_core.py",
        "test_noise_corrected_direct_v3_trainer.py",
        "test_noise_corrected_action_routing_v3.py",
        "test_noise_corrected_routed_ledger_streaming.py",
        "test_noise_corrected_fullgraph_evaluation.py",
        "test_noise_corrected_transfer_objective_v4.py",
        "test_noise_corrected_update_arbitration_v4.py",
    ):
        assert f"tasks/{dependency}" in SOURCE_MANIFEST.read_text(encoding="utf-8")
    for dependency in (
        "dreams/models/dreams/dreams.py",
        "dreams/models/dreams/layers.py",
        "dreams/utils/data.py",
        "dreams/utils/dformats.py",
    ):
        assert dependency in SOURCE_MANIFEST.read_text(encoding="utf-8")


def test_all_python_preflight_executes_inside_the_allocated_sbatch() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n#SBATCH")
    assert 'python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_best_action_v6.py"' in text
    assert (
        'python -u "$SNAPSHOT_ROOT/tasks/test_summarize_noise_corrected_direct_v3_arms.py"'
        in text
    )
    assert 'python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_best_v6_sbatch.py"' in text
    assert "teacher_embedding_or_distillation_target_used" in text


def test_exact_schedule_geometry_is_gated_before_any_model_worker() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    schedule_preflight = text.index("schedule_geometry = v3_schedule_geometry(")
    first_worker = text.index(
        'run_arm_async "$GPU_ZERO" routed_direct &'
    )
    assert schedule_preflight < first_worker
    assert "_validate_registered_best_action_v6_schedule_geometry(" in text
    assert "corrective_batch_size=4" in text
    assert "protective_batch_size=8" in text
    assert "maximum_auxiliary_microbatches_per_step=4" in text
    assert "maximum_protective_microbatches_per_step=4" in text
    assert "maximum_corrective_recycle_factor=4.0" in text
    assert '"schedule_geometry_validated_before_model_load": True' in text


def test_source_manifest_is_complete_and_matches_every_local_file() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert SOURCE_MANIFEST.is_file()
    lines = [line for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 61
    paths: list[str] = []
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
        paths.append(relative)
    assert len(paths) == len(set(paths))
    manifest_digest = hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
    assert f'EXPECTED_SOURCE_MANIFEST_SHA256="{manifest_digest}"' in text


def test_formal_summary_is_single_seed_only_and_parameters_are_rechecked() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert 'assert report["bootstrap_resamples"] == 10000' in text
    assert 'assert report["formula_ci_familywise_hypotheses"] == 4' in text
    assert 'assert isinstance(report["single_seed_gate_pass"], bool)' in text
    assert 'assert report["promote"] is False' in text
    assert '"pending_multiseed", "single_seed_gate_failed"' in text
    assert "best_action_v6_cap_safe_871_to_877_schedule_verified" in text
    assert (
        'assert report["direct_contract"] == '
        '"best_action_v7_corrective_restored"'
    ) in text
    assert 'assert report["training_seed"] == 20260911' in text
    assert "optimizer_update_restoration_contract_valid" in text
    assert "0a7de098701e9ef5f6c3fa2ee62fcc2e1b301dab44878286bd0b61bc31e3e9d9" in text


def test_v7_optimizer_boundary_is_corrective_only_and_state_reconciled() -> None:
    trainer = (REPO_ROOT / "tasks/train_noise_corrected_routed_direct.py").read_text(
        encoding="utf-8"
    )
    arbitrator = (REPO_ROOT / "tasks/noise_corrected_update_arbitration_v4.py").read_text(
        encoding="utf-8"
    )
    assert "projected_noncorrective_action" in trainer
    assert "virtual_noncorrective" in trainer
    assert "arbitrate_corrective_optimizer_updates_by_group(" in trainer
    assert "virtual_combined,\n                                    virtual_noncorrective,\n                                    virtual_risk," in trainer
    assert "virtual_attribution_baseline = virtual_noncorrective" in trainer
    assert "conditional_corrective_grad" in trainer
    assert "reconcile_adamw_first_moments_to_materialized_updates_(" in trainer
    assert '"optimizer_update_restoration_corrective_only"' in trainer
    assert '"protective_gradient_reaches_every_parameter_on_every_active_step"' in trainer
    assert "def arbitrate_corrective_optimizer_updates(" in arbitrator
    assert "combined_updates, noncorrective_baseline_updates" in arbitrator
    assert "def reconcile_adamw_first_moments_to_materialized_updates_(" in arbitrator


def main() -> None:
    tests = [
        test_resource_contract_is_exactly_two_gpus_without_manual_memory,
        test_formal_training_uses_the_repaired_v3_interface_and_no_query_limits,
        test_immutable_action_bank_and_three_causal_arms_are_fail_closed,
        test_all_python_preflight_executes_inside_the_allocated_sbatch,
        test_exact_schedule_geometry_is_gated_before_any_model_worker,
        test_source_manifest_is_complete_and_matches_every_local_file,
        test_formal_summary_is_single_seed_only_and_parameters_are_rechecked,
        test_v7_optimizer_boundary_is_corrective_only_and_state_reconciled,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_best_v6_sbatch] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
