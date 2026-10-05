"""Static fail-closed audit for direct candidate-boundary fine-tuning."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = {
        "core": ROOT / "tasks/noise_final_candidate_boundary_core.py",
        "test": ROOT / "tasks/test_noise_final_candidate_boundary_core.py",
        "trainer": ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py",
        "manifest": ROOT / "tasks/build_noise_final_e4_pmt_manifest.py",
        "summary": ROOT / "tasks/summarize_noise_final_e4_deb.py",
        "nist": ROOT / "tasks/evaluate_noise_shared_nist20_pairwise.py",
        "sbatch": ROOT / "tasks/run_noise_final_e4_deb_phase_a.sbatch",
        "resume": ROOT / "tasks/run_noise_final_e4_deb_resume_2331540.sbatch",
        "coverage": ROOT / "tasks/audit_noise_final_e4_deb_action_coverage.py",
        "coverage_test": ROOT / "tasks/test_noise_final_e4_deb_action_coverage.py",
        "coverage_sbatch": ROOT / "tasks/run_noise_final_e4_deb_coverage_2331540.sbatch",
    }
    source: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source[name] = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(source[name])
    required = {
        "core": [
            "target_i - control_i", "control_i.detach()", "positive =",
            "topk_negatives", "lambda_counterfactual", "candidate_safety_objective",
        ],
        "trainer": [
            "edge_margin_matrices", "calibrate_candidate_boundary_gradients",
            "formula_stratified_microbatches", "effective_boundary_action_weight",
            "refresh_candidate_references", "graph_multimetric_summary",
            "action_exposure_schedule_sha256",
            "action_safety_formula_strata_identical",
        ],
        "manifest": [
            '"--corrective-query-scope"', 'choices=("all", "errors")',
            'corrective["clean_rank"].astype(int).ne(1)',
        ],
        "summary": [
            'treatment = "candidate_boundary"',
            '("clean_duplicate", "matched_random")',
            "all_secondary_metrics_nonnegative_vs_",
            "external_nist20_supplementary",
            "never a promotion gate",
        ],
        "nist": [
            "SpecRetrievalValidation", "pooled_pairwise_auroc",
            "exact_paper_replication", "legacy callback",
            "candidate_boundary_vs_", "clean-control-checkpoint",
            "load_noise_inference_model", "NOISE_SHARED_STATUS",
        ],
        "sbatch": [
            "#SBATCH --gpus=1", "candidate_boundary", "--refresh-hard-negatives",
            "--corrective-query-scope errors", "fold_0_run_2331467",
            "test_noise_final_candidate_boundary_core.py",
        ],
        "resume": [
            "#SBATCH --gpus=1", "fold_0_run_2331540",
            "summary_resume_${SLURM_JOB_ID}",
            "summarize_noise_final_e4_deb.py",
        ],
        "coverage": [
            "optimizer_steps", "action_covered_error_queries",
            "selected_fraction_of_action_covered_error_queries",
            "P_unique_beyond_N", "outcomes_used_only_for_diagnostic_coverage",
        ],
        "coverage_sbatch": [
            "#SBATCH --gpus=1", "fold_0_run_2331540",
            "audit_noise_final_e4_deb_action_coverage.py",
            "test_noise_final_e4_deb_action_coverage.py",
            "coverage_audit_${SLURM_JOB_ID}",
        ],
    }
    missing = {
        name: [token for token in tokens if token not in source[name]]
        for name, tokens in required.items()
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise RuntimeError(f"E4-DEB implementation audit failed: {missing}")
    if any(
        "#SBATCH --mem" in source[name]
        for name in ("sbatch", "resume", "coverage_sbatch")
    ):
        raise RuntimeError("E4-DEB sbatch must not request explicit memory")
    if "train_noise_final" in source["resume"]:
        raise RuntimeError("job-2331540 resume must not retrain any encoder")
    forbidden_trainer = (
        "candidate_boundary_scalar_teacher_used\": True",
        "candidate_boundary_control_symmetrically_ranked\": True",
    )
    if any(token in source["trainer"] for token in forbidden_trainer):
        raise RuntimeError("E4-DEB trainer reintroduced a forbidden scalar/symmetric path")
    forbidden_nist_imports = (
        "from shared_dreams_inference import",
        "dreams.models.chem_aware",
    )
    if any(token in source["nist"] for token in forbidden_nist_imports):
        raise RuntimeError("NIST20 evaluator depends on optional ChemAware inference code")
    print("[audit_noise_final_e4_deb_implementation] PASS")


if __name__ == "__main__":
    main()
