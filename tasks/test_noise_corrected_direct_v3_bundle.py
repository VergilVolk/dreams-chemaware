"""Static fail-closed checks for the complete direct-v3 submission bundle."""
from __future__ import annotations

import ast
from pathlib import Path


TASKS = Path(__file__).resolve().parent
SBATCH = TASKS / "run_noise_corrected_direct_v3_from_formal_routes.sbatch"
ROUTE_SBATCH = TASKS / "run_noise_corrected_v3_routes_fold0.sbatch"
CANARY_SBATCH = TASKS / "run_noise_corrected_direct_v3_canary_2gpu.sbatch"


def _defined_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.Import):
            symbols.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names)
    return symbols


def _assert_direct_local_imports_resolve(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        module_path = TASKS / f"{node.module}.py"
        if not module_path.is_file():
            continue
        available = _defined_symbols(module_path)
        missing = {
            alias.name for alias in node.names
            if alias.name != "*" and alias.name not in available
        }
        if missing:
            raise RuntimeError(
                f"{path.name} imports missing symbols from {module_path.name}: "
                f"{sorted(missing)}"
            )


def test_local_bundle_imports_exist() -> None:
    files = (
        "noise_corrected_action_routing_v3.py",
        "noise_corrected_action_panel.py",
        "noise_corrected_full_action_registry.py",
        "noise_corrected_direct_v3_core.py",
        "noise_corrected_action_expansion_v4.py",
        "noise_corrected_transfer_objective_v4.py",
        "noise_corrected_update_arbitration_v4.py",
        "noise_corrected_directional_update_v4.py",
        "noise_corrected_shuffled_control_v3.py",
        "noise_corrected_fullgraph_evaluation.py",
        "build_noise_corrected_routed_action_ledger.py",
        "audit_noise_corrected_direct_v3_plan.py",
        "train_noise_corrected_routed_direct.py",
        "summarize_noise_corrected_direct_v3_arms.py",
        "summarize_noise_corrected_direct_v3_canary.py",
        "test_noise_corrected_action_routing_v3.py",
        "test_noise_corrected_action_panel.py",
        "test_noise_corrected_full_action_registry.py",
        "test_noise_corrected_routed_ledger_streaming.py",
        "test_noise_corrected_direct_v3_core.py",
        "test_noise_corrected_direct_v3_trainer.py",
        "test_noise_corrected_transfer_objective_v4.py",
        "test_noise_corrected_update_arbitration_v4.py",
        "test_noise_corrected_directional_update_v4.py",
        "test_noise_corrected_fullgraph_evaluation.py",
        "test_summarize_noise_corrected_direct_v3_arms.py",
        "test_summarize_noise_corrected_direct_v3_canary.py",
        "audit_noise_corrected_full_p_router.py",
        "audit_noise_corrected_a4_full_router.py",
        "audit_noise_corrected_n_router.py",
    )
    for name in files:
        path = TASKS / name
        if not path.is_file():
            raise FileNotFoundError(path)
        _assert_direct_local_imports_resolve(path)
    trainer = (TASKS / "train_noise_corrected_routed_direct.py").read_text(
        encoding="utf-8"
    )
    for required in (
        'branch_scale["consistency"] * corr_components["consistency"]',
        '"corrective.embedding_grad_consistency_query_norm"',
        '"corrective.embedding_grad_consistency_action_norm"',
        '"action_and_control_candidate_switch_molecules_retained"',
        '"clean_action_control_topk_edge_union"',
        '"routed_hard_rows_never_truncated_within_molecule"',
        '"active_transfer_fraction_gate_applicable"',
        '"mature_E8_gradient_clip_restored"',
        '"optimizer_counterfactual_action_attribution_measured"',
        '"optimizer_parameter_clones_limited_to_registered_audit_steps"',
        '"head_and_backbone_action_signal_measured_separately"',
        '"corrective_direction_traced_through_risk_clip_and_adamw"',
        '"near_corrected_introduced_risk_reported"',
        '"parameter_group_action_signal_gate_passed"',
        '"registered_formal_v3_configuration_verified"',
        '"formal_query_truncation_disabled"',
        '"maximum_clean_queries": 0',
        '"outer_held_eval_queries": 0',
        '"source_manifest_exact_graph_alignment"',
        '"route_and_training_formula_fold_seed_exactly_aligned"',
        '"shuffled_control_matches_supervision_source_family_and_exact_recipe"',
        '"control_semantics_explicit_and_separate"',
        '"corrective_N_P_A4_global_epoch_mass_equalized_across_queries"',
        '"P_repeated_generation_source_does_not_multiply_family_dose"',
        '"selector_balances_source_then_family_before_recipe_multiplicity"',
        '"robust_harmful_use_query_local_not_sparse_global_equalization"',
        '"sparse_auxiliary_batches_evenly_interleaved"',
        '"sparse_auxiliary_global_duty_compensation_applied": False',
        '"partial_batch_query_mass_scaled_to_registered_size": True',
        '"protective_microbatches_have_equal_epoch_weight": True',
        '"clean_control_skips_no_gradient_action_forwards": True',
        '"corrective_scale_uses_dense_corrective_branches_only": True',
        '"sparse_auxiliary_branches_excluded_from_corrective_scale_denominator": True',
        '"training_epoch_transfer_edge_strata_are_pooled_not_batch_averaged"',
        '"dense_corrective_to_risk_target_exactly_reached"',
        '"action_calibration_caps_not_silently_truncated"',
        '"optimizer_action_attributable_update_fraction_gate_passed"',
        '"legacy_90pct_end_to_end_loss_reproduced"',
        '"mechanism_prioritized_formula_diverse_prefix"',
        '"actual_scheduled_corrective_mechanism_epoch_mass_equalized"',
        '"global_mechanism_coefficient_cap_passed"',
        '"bounded_memory_sequential_action_branch_backprop"',
        '"dense_auxiliary_panels_cannot_multiply_calibrated_step_mass"',
        '"auxiliary_projected_against_corrective_before_fullgraph_risk"',
        '"corrective_direction_preserved_through_inner_semantic_projection"',
        '"semantic_action_branches_have_separate_gradient_ledgers"',
    ):
        if required not in trainer:
            raise RuntimeError(f"direct-v3 trainer misses active path: {required}")


def test_sbatch_is_two_gpu_no_manual_memory_and_complete() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=2",
        "BASE_RUN",
        "tasks/audit_noise_corrected_direct_v3_plan.py",
        "tasks/noise_corrected_shuffled_control_v3.py",
        "formal source/family/exact-recipe shuffle preflight",
        "source_family_shuffle_strata_report",
        'report["cross_query_fraction"] >= 0.95',
        "tasks/test_noise_corrected_direct_v3_bundle.py",
        "tasks/test_noise_corrected_direct_v3_core.py",
        "tasks/test_noise_corrected_action_panel.py",
        "tasks/test_noise_corrected_routed_ledger_streaming.py",
        "tasks/test_noise_corrected_direct_v3_trainer.py",
        "tasks/test_noise_corrected_transfer_objective_v4.py",
        "tasks/test_noise_corrected_update_arbitration_v4.py",
        "tasks/test_noise_corrected_directional_update_v4.py",
        "tasks/test_noise_corrected_fullgraph_evaluation.py",
        "tasks/test_summarize_noise_corrected_direct_v3_arms.py",
        "--corrective-objective-mode v3_direct",
        "--positive-references 4",
        "--lambda-action-safety 0.25",
        "--lambda-v3-corrective-consistency 0.25",
        "--target-v3-consistency-to-transfer-ratio 0.25",
        "--minimum-action-retention-p10 0.50",
        "--minimum-optimizer-action-alignment-p10 0.05",
        "--minimum-optimizer-action-attributable-fraction-p10 0.10",
        "--minimum-corrective-direction-retention-p10 0.10",
        '"recall1_headroom_uses_actual_action_rank1_not_partial_margin"',
        "--optimizer-attribution-steps-per-epoch 16",
        "--maximum-clip-event-fraction 0.10",
        "--grad-clip 1.0",
        "--v3-maximum-protective-microbatches-per-step 4",
        "--v3-maximum-auxiliary-microbatches-per-step 4",
        "--v3-maximum-corrective-recycle-factor 4",
        "--v3-corrective-recycle-full-dose",
        "--minimum-v3-active-transfer-fraction 0.05",
        "--lambda-risk-floor 2.0",
        "--lambda-preserve 5.0",
        "--action-conditioned-positive-references 32",
        "--action-conditioned-negative-molecules 32",
        "--reference-refresh-every-epochs 1",
        "--n-highest-peaks 100",
        "--formula-ci-familywise-hypotheses 4",
        "exact_action_control_candidate_switch_rows_recorded",
        "control_semantics_explicit_and_source_validated",
        'report["training_seed"] == seed',
        "duplicate routed seed result detected",
        "duplicate routed seed model state detected",
        'report["provenance"]["routed_final_model_state_sha256"]',
        'report["provenance"]["shared_source_manifest_sha256"]',
        "20260906:shuffled_action_control",
        "20260906:clean_control",
        "WORKER_ZERO_TASKS",
        "WORKER_ONE_TASKS",
        "run_arm_worker",
        '--arm "$ARM_NAME"',
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"direct-v3 SBATCH misses required contract text: {missing}")
    expected_tasks = {
        f'"{seed}:{arm}"'
        for seed in (20260906, 20260907, 20260908)
        for arm in ("routed_direct", "shuffled_action_control", "clean_control")
    }
    if any(text.count(task) != 1 for task in expected_tasks):
        raise RuntimeError("persistent GPU queues do not cover every formal arm exactly once")
    if "#SBATCH --mem" in text or "#SBATCH --mem-per-cpu" in text:
        raise RuntimeError("direct-v3 SBATCH must not request memory manually")


def test_route_sbatch_is_bounded_two_gpu_and_chains_training() -> None:
    text = ROUTE_SBATCH.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=2",
        "GPU0: mature N construction/routing, then chunked A4",
        "GPU1: chunked complete P66",
        "--query-scope initial_error_boundary",
        "--query-chunk-size 64",
        "--query-chunk-size 16",
        "--query-chunk-size 128",
        "--maximum-corrective-frontier 16",
        "--maximum-harmful-frontier 8",
        "--maximum-robust-frontier 8",
        "monolithic_all_action_embedding_allocation",
        "exact_action_control_candidate_switch_rows_recorded",
        "control_semantics_explicit",
        "tasks/test_noise_final_e10_positive_residual_matrix.py",
        "tasks/test_noise_corrected_full_action_registry.py",
        "tasks/test_noise_corrected_n_router.py",
        "all_66_unique_mature_P_cells_materialized",
        "original_12_cell_P_intensity_matrix_complete_without_duplicates",
        '"P_guided_original": 6',
        "reference_profiles_cached_per_query_policy_direction",
        "missing_peaks_cached_per_query_policy_direction_parameters",
        "exact_action_tensor_deduplication_before_encoder_is_lossless",
        "AUTO_SUBMIT_TRAINING",
        "tasks/run_noise_corrected_direct_v3_from_formal_routes.sbatch",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"direct-v3 route SBATCH misses required contract: {missing}")
    if "#SBATCH --mem" in text or "#SBATCH --mem-per-cpu" in text:
        raise RuntimeError("direct-v3 route SBATCH must not request memory manually")


def test_canary_sbatch_is_bounded_two_gpu_complete_and_observable() -> None:
    text = CANARY_SBATCH.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=2",
        "#SBATCH --time=12:00:00",
        "BASE_RUN",
        "tasks/test_noise_corrected_direct_v3_bundle.py",
        "tasks/test_noise_corrected_direct_v3_core.py",
        "tasks/test_noise_corrected_direct_v3_trainer.py",
        "tasks/test_noise_corrected_action_expansion_v4.py",
        "tasks/test_noise_corrected_transfer_objective_v4.py",
        "tasks/test_noise_corrected_update_arbitration_v4.py",
        "tasks/test_noise_corrected_directional_update_v4.py",
        "tasks/test_noise_corrected_fullgraph_evaluation.py",
        "tasks/test_summarize_noise_corrected_direct_v3_canary.py",
        "--development --epochs 1",
        "--maximum-corrective-queries 512",
        "--maximum-risk-queries 512",
        "--maximum-robust-queries 256",
        "--maximum-clean-queries 4096",
        "--outer-held-eval-queries 4096",
        "--progress-every-steps 25",
        'run_arm "$GPU_ZERO" routed_direct &',
        'run_arm "$GPU_ONE" shuffled_action_control &',
        'wait "$PID_ROUTED"',
        'wait "$PID_SHUFFLED"',
        'run_arm "$GPU_ZERO" clean_control',
        "tasks/summarize_noise_corrected_direct_v3_canary.py",
        'mv "$RUN_ROOT" "$FINAL_ROOT"',
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"direct-v3 canary SBATCH misses contract: {missing}")
    if text.count("#SBATCH --gpus=2") != 1:
        raise RuntimeError("direct-v3 canary must request exactly one two-GPU allocation")
    if "#SBATCH --mem" in text or "#SBATCH --mem-per-cpu" in text:
        raise RuntimeError("direct-v3 canary SBATCH must not request memory manually")
    _assert_direct_local_imports_resolve(
        TASKS / "summarize_noise_corrected_direct_v3_canary.py"
    )


def main() -> None:
    tests = (
        test_local_bundle_imports_exist,
        test_sbatch_is_two_gpu_no_manual_memory_and_complete,
        test_route_sbatch_is_bounded_two_gpu_and_chains_training,
        test_canary_sbatch_is_bounded_two_gpu_complete_and_observable,
    )
    for test in tests:
        test()
    print(f"[test_noise_corrected_direct_v3_bundle] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
