"""Static fail-closed checks for the two-GPU best-actions hybrid job."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_e4_best_actions_hybrid_v1_2gpu.sbatch"
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
VALIDATOR = ROOT / "tasks/validate_noise_final_e4a_direct_augmentation.py"
EVALUATOR = ROOT / "tasks/evaluate_noise_e4_best_actions_injector_v1_final.py"


def main() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=2",
        "complete_panel_historical_e4",
        "--optimizer-boundary-mode action_injector_v1",
        "--injector-target-attributable-fraction 0.25",
        "--injector-minimum-protective-retention 0.90",
        "--injector-maximum-update-norm-ratio 1.50",
        "--direct-transfer-mode symmetric",
        "--rank-reference-mode shared",
        "--positive-stream-weight 0",
        "--guided-noise-policy none",
        "--pmt-arm none",
        "--no-candidate-boundary-loss",
        "--no-refresh-hard-negatives",
        "--no-amp",
        "launch_arm \"$GPU_ZERO\" targeted",
        "launch_arm \"$GPU_ONE\" shuffled",
        "terminate_worker_tree",
        "while IFS= read -r CHILD_PID",
        "executed_source_manifest.sha256",
        "evaluate_noise_e4_best_actions_injector_v1_final.py",
        "test_noise_final_e4a_direct_augmentation.py",
        "test_noise_e4_best_actions_hybrid_v1.py",
        "8ee5b5aa604bda7f1e3cc99396773cebd10749e8820d02c3de048f2865c2509e",
        "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
        "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
        "32114",
        "3482",
        "N_mature",
        "P_guided_original",
        "E10B",
        "E11",
        "E12B",
        "A4_exact",
        "V4_gradient_path",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise AssertionError(f"hybrid SBATCH lost required contracts: {missing}")
    forbidden = (
        "#SBATCH --mem",
        "--initial-student-checkpoint",
        "--materialized-injection-mode multi_action_balanced",
        "--materialized-injection-mode one_best_e4",
        "__MANIFEST_SHA256__",
        "CHILDREN[@]",
    )
    found = [token for token in forbidden if token in text]
    if found:
        raise AssertionError(f"hybrid SBATCH retained forbidden predecessor path: {found}")
    for token in (
        "complete_panel_historical_e4",
        "hybrid_changes_only_action_supplier_and_optimizer_boundary",
        "complete_seven_source_action_panel_used_without_one_best_compression",
        "objective_reference_by_row",
        "audit_historical_e4_action_specific_alignment",
        "action_specific_to_clean_corrective_gradient_gate_passed",
    ):
        if token not in trainer:
            raise AssertionError(f"trainer lacks final hybrid token: {token}")
    if "complete-panel historical E4 contract failed" not in validator:
        raise AssertionError("validator does not recognize the final hybrid")
    for token in (
        'final_shared_encoder_sha256',
        'materialized_injection_mode',
        '_validate_causal_pair',
        'action_exposure_schedule_sha256',
        'all_nonfallback_donors_inside_input_panel',
        'strict_four_pp_full_panel_result_achieved',
        '_registered_metric_strict_or_boundary_failures',
        'official_checkpoint_edge_scores_plus_candidate_comparison_tie_aware_ranks',
    ):
        if token not in evaluator:
            raise AssertionError(f"final evaluator lacks artifact/control gate: {token}")
    print("[test_noise_e4_best_actions_hybrid_v1_sbatch] PASS")


if __name__ == "__main__":
    main()
