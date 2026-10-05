"""Static fail-closed audit of the dynamic direct implementation boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = {
        "core": ROOT / "tasks/noise_final_dynamic_direct_core.py",
        "preflight": ROOT / "tasks/audit_noise_final_dynamic_direct_preflight.py",
        "ledger": ROOT / "tasks/build_noise_final_dynamic_direct_ledger.py",
        "schedule": ROOT / "tasks/build_noise_final_dynamic_direct_schedule.py",
        "replay": ROOT / "tasks/audit_noise_final_dynamic_direct_replay.py",
        "sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_preflight.sbatch",
        "replay_sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_replay.sbatch",
        "m1_sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_m1.sbatch",
        "m2": ROOT / "tasks/build_noise_final_dynamic_direct_m2_crossfit.py",
        "m2_sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_m2.sbatch",
        "trainer": ROOT / "tasks/train_noise_final_dynamic_direct_phase_a.py",
        "summary": ROOT / "tasks/summarize_noise_final_dynamic_direct_phase_a.py",
        "big_sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_e4_phase_a_final.sbatch",
        "contract": ROOT / "docs/NOISE_FINAL_DYNAMIC_CONDITIONAL_DIRECT_FINETUNING_CONTRACT_20260904.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    for name in ("core", "preflight", "ledger", "schedule", "replay", "m2", "trainer", "summary"):
        ast.parse(source[name])
    gates = {
        "gpu_requested": all("#SBATCH --gpus=1" in source[name] for name in ("sbatch", "replay_sbatch", "m1_sbatch", "m2_sbatch")),
        "no_explicit_memory_request": all("#SBATCH --mem" not in source[name] for name in ("sbatch", "replay_sbatch", "m1_sbatch", "m2_sbatch")),
        "unique_output": "SLURM_JOB_ID" in source["sbatch"],
        "no_post_outcome_cell_selection": all(
            token not in source["ledger"]
            for token in ("passing_cells", "best_fixed_cell", "oracle_per_query")
        ),
        "N_all_nine_cells": "validate_n_cells" in source["ledger"],
        "P_all_twenty_one_cells": all(
            token in source["ledger"]
            for token in ("P_INTENSITY_FAMILIES", "P_TRANSFER_FAMILIES", "expected_cells = 30")
        ),
        "outer_held_removed_before_P_fit": (
            "outer_train_query = np.flatnonzero(formula_fold != args.outer_fold)" in source["ledger"]
        ),
        "formula_crossfit": "train = (folds != fold) & (folds != outer_fold)" in source["ledger"],
        "no_raw_P_outcomes_in_training_columns": "raw_P_outcomes_published\": False" in source["ledger"],
        "L0_geometry_exact": "exact clean geometry used to define L0/L1 action labels" in source["preflight"],
        "formula_identity_family_exposure_in_sampler": "stratified_action_schedule" in source["core"],
        "epoch_cell_cycling": all(
            token in source["core"] + source["schedule"]
            for token in ("epoch-cycling", "epoch_cell_cycling", "epoch_schedule_index")
        ),
        "static_matches_conditional_query_dose": all(
            token in source["schedule"]
            for token in ("static_dynamic_mass_max_abs_difference", "static_matches_dynamic_query_dose")
        ),
        "hierarchical_training_mass": "formula_identity_query_equal_weights" in source["trainer"],
        "explicit_positive_no_op_mass": all(
            token in source["core"] + source["ledger"]
            for token in ("no_op_weight", "explicit no-op mass collapsed", "action/no-op mass")
        ),
        "one_schedule_for_all_phase_a_arms": all(
            token in source["schedule"]
            for token in ("PHASE_A_ARMS", "membership_sha256", "one_schedule_for_all_arms")
        ),
        "full_ledger_replayed_before_schedule": (
            "training_actions.csv.gz" in source["replay"]
            and "epoch_schedule.csv.gz" not in source["replay"]
            and "optimizer_steps\": 0" in source["replay"]
        ),
        "current_geometry_full_candidate_replay": all(
            token in source["replay"]
            for token in ("score_vector", "paired_advantage", "exact_current_geometry", "full_candidate_scoring")
        ),
        "current_geometry_crossfit_with_ablations": all(
            token in source["m2"]
            for token in ("cell_only", "permuted_clean", "positive_brier_gain_full_vs_cell_only",
                          "risk_probability", "current_geometry_labels_only")
        ),
        "utility_not_inverse_frequency": (
            "Dataset-abundance correction is deferred" in source["core"]
            and "1.0 / counts" not in source["core"].split("def build_action_weights", 1)[1].split("def validate_n_cells", 1)[0]
        ),
        "sampling_not_double_weighted": "never a second time through sampling probability" in source["core"],
        "P2b_absent": "P2b_score" not in "\n".join(source.values()),
        "contract_direct_primary": "直接微调是主线" in source["contract"],
        "mature_e4_requires_fresh_replay": all(
            token in source["preflight"] + source["big_sbatch"]
            for token in ("mature_e4_current_replay", "pass_to_multifold", "current_geometry")
        ),
        "current_embedding_features_used": (
            "current_geometry_embeddings.npz" in source["replay"]
            and "current_geometry_embeddings.npz" in source["m2"]
        ),
        "direct_shared_full_list_training": all(
            token in source["trainer"]
            for token in ("molecule_logits", "unfreeze_last_blocks", "shared_query_reference_encoder",
                          "full_candidate_molecule_list_training", "inference_clean_spectrum_only")
        ),
        "four_paired_arms": all(
            token in source["big_sbatch"]
            for token in ("clean_continuation", "matched_random", "static_target", "dynamic_np")
        ),
        "one_big_gpu_job_without_memory_request": (
            "#SBATCH --gpus=1" in source["big_sbatch"]
            and "#SBATCH --mem" not in source["big_sbatch"]
            and "SLURM_JOB_ID" in source["big_sbatch"]
        ),
        "gpu_smoke_precedes_full_training": (
            "--smoke" in source["big_sbatch"]
            and source["big_sbatch"].index("--smoke")
            < source["big_sbatch"].index("# Four paired arms")
        ),
        "paired_formula_multiplicity_summary": all(
            token in source["summary"] for token in (
                "signflip_p", "holm_adjusted_p", "formula_cluster_ci",
                "dynamic_beats_matched_random_formula_ci_positive",
                "dynamic_beats_static_target_formula_ci_positive",
            )
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-direct implementation audit failed: {gates}")
    report = {
        "status": "noise_final_dynamic_direct_implementation_audit_passed",
        "gates": gates,
        "scope": "mature-E4 replay, current-geometry crossfit, explicit no-op schedule, four-arm direct shared-encoder training",
        "claim_limit": "Static implementation audit; performance is established only by the completed Phase-A outputs.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
