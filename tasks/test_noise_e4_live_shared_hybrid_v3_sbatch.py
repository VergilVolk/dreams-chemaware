"""Static contract checks for the sole Hybrid V3 Slurm entrypoint."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_e4_live_shared_hybrid_v3_1gpu.sbatch"
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
MODULE = ROOT / "tasks/noise_e4_live_shared_hybrid_v3.py"
VALIDATOR = ROOT / "tasks/validate_noise_final_e4a_direct_augmentation.py"
SUMMARIZER = ROOT / "tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py"


def test() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=1",
        "--materialized-injection-mode e4_live_shared_v3",
        "--optimizer-boundary-mode separated_e4_action_v3",
        "python -u tasks/test_noise_e4_live_shared_hybrid_v3.py",
        "run_arm targeted",
        "run_arm shuffled",
        'RECOVER_RUN_ID="${RECOVER_RUN_ID:-}"',
        'STAGING_ROOT="data/validation/.noise_e4_live_shared_hybrid_v3_',
        'RECOVERY_LAYOUT="final_in_place"',
        "arm_dir_is_complete()",
        "arm_is_complete()",
        'held_per_query.csv.gz',
        'run_arm_into shuffled "$RECOVERY_SHUFFLED_ROOT"',
        'mv -Tn "$RECOVERY_SHUFFLED_RESULT" "$CANONICAL_SHUFFLED_RESULT"',
        "Reusing completed targeted arm",
        "Reusing completed shuffled arm",
        "Reusing current strict matched summary",
        "summary_schema_is_current()",
        'AUTHORITATIVE_SUMMARY="$RUN_ROOT/summary"',
        'AUTHORITATIVE_SUMMARY="$RUN_ROOT/summary_strict_v3"',
        "strict_full_panel_four_pp_achieved",
        "exact_fraction_is_not_targeted_minus_shuffled_fraction",
        "print_summary_report",
        'if [[ "$RUN_ROOT" != "$FINAL_ROOT" ]]',
        'mv -Tn "$RUN_ROOT" "$FINAL_ROOT"',
        "--hybrid-version v3",
        "--positive-spectra 4 --negative-molecules 8",
        "--direct-transfer-mode symmetric --rank-reference-mode shared",
        "--no-amp",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError(f"V3 sbatch lost required tokens: {missing}")
    forbidden = (
        "#SBATCH --mem",
        "e4_base_semantic_v2",
        "signal_preserving_v2",
        "query_local_e4_semantic_residual",
        "--amp\n",
    )
    present = [value for value in forbidden if value in text]
    if present:
        raise AssertionError(f"V3 sbatch revived a forbidden route: {present}")
    obsolete_recovery = (
        "Recovery requires existing staging and absent final root",
        '[[ -d "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]]',
    )
    if present_obsolete := [
        value for value in obsolete_recovery if value in text
    ]:
        raise AssertionError(
            f"V3 sbatch still requires the obsolete staging-only recovery: "
            f"{present_obsolete}"
        )

    trainer = TRAINER.read_text(encoding="utf-8")
    v3_loop = trainer.split("def train_e4_live_shared_v3_epochs(", 1)[1].split(
        "\ndef main()", 1,
    )[0]
    required_loop = (
        "materialized_live_shared_e4_loss_v3(",
        "injector.capture_action_gradient_",
        "direct_action_loss(",
        "safety_loss(",
        "injector.step_",
    )
    if missing_loop := [value for value in required_loop if value not in v3_loop]:
        raise AssertionError(f"production V3 loop lost a required stream: {missing_loop}")
    if "materialized_query_local_semantic_loss(" in v3_loop:
        raise AssertionError("production V3 loop revived the rejected detached residual")
    if "minimum_e4_projection_retention=(" not in trainer:
        raise AssertionError("trainer did not pass the registered 0.90 E4 retention floor")
    v3_gate_tokens = (
        'promotion_reference = "initialization"',
        '"clean_recall_positive_vs_initial_e8"',
        '"near_risk_net_lambda2_positive_vs_initial_e8"',
        '"all_registered_metrics_nonnegative_vs_initial_e8"',
        '"all_registered_metrics_strict_or_boundary_vs_initial_e8"',
        '"strict_four_pp_recall1_gain_vs_official"',
    )
    if missing_v3_gates := [
        value for value in v3_gate_tokens if value not in trainer
    ]:
        raise AssertionError(
            f"V3 promotion revived the official-only gate: {missing_v3_gates}"
        )

    module = MODULE.read_text(encoding="utf-8")
    objective = module.split("def live_shared_e4_action_objective_v3(", 1)[1].split(
        "\ndef _clone_gradients(", 1,
    )[0]
    live_tokens = (
        'clean = encoded[int(layout["clean"])]',
        'action = encoded[int(layout["action"])]',
        "positives = encoded[positive_indices]",
        "negatives = encoded[negative_indices]",
        "action_rank_terms.append(F.softplus",
    )
    if missing_live := [value for value in live_tokens if value not in objective]:
        raise AssertionError(f"production V3 objective lost live E4 roles: {missing_live}")
    if "action_margin.detach()" in objective or "torch.where(active" in objective:
        raise AssertionError("production V3 objective hard-gated the action rank")
    exact_injection = (
        "compose_safe_exact_corrective_updates_by_group(",
        "target_attributable_fraction=self.target_action_fraction",
        "minimum_protective_component_retention=(",
        'raise RuntimeError("V3 exact action fraction drifted")',
    )
    if missing_exact := [value for value in exact_injection if value not in module]:
        raise AssertionError(
            f"production V3 injector lost exact signal delivery: {missing_exact}"
        )
    if "maximum_action_fraction" in module or "action_update_never_amplified" in module:
        raise AssertionError("production V3 revived the ceiling-only signal-loss bug")
    validator = VALIDATOR.read_text(encoding="utf-8")
    if 'injection_mode not in {"e4_base_semantic_v2", "e4_live_shared_v3"}' not in validator:
        raise AssertionError("validator revived the obsolete V3 identity-dose field")
    summarizer = SUMMARIZER.read_text(encoding="utf-8")
    final_gate_tokens = (
        'strict_fail_shuffle = _registered_metric_strict_or_boundary_failures(',
        '"all_registered_metrics_strict_or_boundary_vs_shuffled"',
        '"registered_metric_strict_failures_vs_matched_shuffled"',
        '"targeted_preservation_vs_initial_e8_ge_0_995"',
        '"shuffled_preservation_vs_initial_e8_ge_0_995"',
        '"causal_decision_role": "ignored"',
        '"different_query_actions_are_mean_reduced_before_optimizer"',
        '"exact_fraction_is_not_targeted_minus_shuffled_fraction"',
    )
    if missing_final_gates := [
        value for value in final_gate_tokens if value not in summarizer
    ]:
        raise AssertionError(
            f"V3 final causal decision lost strict shuffled gates: {missing_final_gates}"
        )


def main() -> None:
    test()
    print("[test_noise_e4_live_shared_hybrid_v3_sbatch] PASS")


if __name__ == "__main__":
    main()
