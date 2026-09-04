"""Stdlib-only implementation audit for CPG0/CPG shared-encoder architecture."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "contract": ROOT / "docs/NOISE_FINAL_CPG_CONTRACT_20260903.md",
    "core": ROOT / "tasks/noise_final_cpg_core.py",
    "builder": ROOT / "tasks/build_noise_final_cpg0_residual_teacher.py",
    "core_test": ROOT / "tasks/test_noise_final_cpg_core.py",
    "builder_test": ROOT / "tasks/test_noise_final_cpg0_residual_teacher.py",
    "validator": ROOT / "tasks/validate_noise_final_cpg0_residual_teacher.py",
    "sbatch": ROOT / "tasks/run_noise_final_cpg0_residual_teacher.sbatch",
}


def function_source(tree: ast.Module, source: str, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise RuntimeError(f"missing audited function: {name}")


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source = {key: path.read_text(encoding="utf-8") for key, path in FILES.items()}
    trees = {
        key: ast.parse(source[key], filename=str(FILES[key]))
        for key in ("core", "builder", "core_test", "builder_test", "validator")
    }
    clean_loss = function_source(trees["core"], source["core"], "clean_candidate_residual_loss")
    builder_main = function_source(trees["builder"], source["builder"], "main")
    sbatch = source["sbatch"]
    forbidden_builder_tokens = (
        "passing_cells", "best_fixed_cell", "oracle_recoverable", "new_beyond_pn",
        "corrected_queries", "outcome_audit_only",
    )
    gates = {
        "clean_loss_has_no_trainable_comparator_detach": ".detach(" not in clean_loss,
        "clean_loss_updates_query_and_candidates": (
            "current_query" in clean_loss and "current_candidates" in clean_loss
        ),
        "teacher_is_vector_not_scalar_only": (
            "paired_candidate_residual" in source["core"]
            and "candidate_residuals.h5" in source["builder"]
        ),
        "all_fixed_grids_are_explicit": all(
            token in source["builder"] for token in (
                "N_GRID", "P_INTENSITY_FAMILIES", "P_INTENSITY_DOSES",
                "P_TRANSFER_FAMILIES", "P_TRANSFER_DOSES",
            )
        ),
        "no_outcome_selected_cell_filter": not any(
            token in source["builder"].lower() for token in forbidden_builder_tokens
        ),
        "held_formula_filtered_before_action_replay": (
            builder_main.find("eligible_queries = np.flatnonzero(folds != args.outer_fold)")
            < builder_main.find("for row in r0.itertuples")
        ),
        "candidate_library_not_filtered_by_holdout": (
            "reachable.update(map(int, graph.pair_candidate_row))" in builder_main
        ),
        "signed_harmful_actions_retained": "harmful_signed_residuals_retained" in builder_main,
        "all_gates_are_positive_assertions": (
            '"P3_not_consumed": True' in builder_main
            and '"P3_consumed": False' not in builder_main
            and "require_positive_gates(report[\"gates\"])" in builder_main
        ),
        "completed_compute_survives_late_failure": (
            "failed_complete" in builder_main and "compute_complete.json" in builder_main
        ),
        "partial_compute_is_never_deleted": (
            "failed_partial" in builder_main and "shutil.rmtree(staging" not in builder_main
        ),
        "atomic_output_publication": (
            "tempfile.mkdtemp" in builder_main and "staging.replace(args.output_dir)" in builder_main
        ),
        "job_requests_one_gpu": "#SBATCH --gpus=1" in sbatch,
        "job_uses_unique_output": "run_${SLURM_JOB_ID}" in sbatch,
        "tests_run_before_model_builder": (
            sbatch.find("test_noise_final_cpg_core.py")
            < sbatch.find("build_noise_final_cpg0_residual_teacher.py")
        ),
        "validator_runs_after_builder": (
            sbatch.find("validate_noise_final_cpg0_residual_teacher.py")
            > sbatch.find("build_noise_final_cpg0_residual_teacher.py")
        ),
        "contract_separates_action_headroom_from_embedding_gain": (
            "No action headroom is reported as learned embedding" in source["contract"]
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"CPG implementation audit failed: {gates}")
    print(json.dumps({
        "status": "noise_final_cpg_implementation_audit_passed",
        "files": {key: str(path.relative_to(ROOT)) for key, path in FILES.items()},
        "gates": gates,
        "note": "stdlib audit only; torch numerical tests execute inside the GPU sbatch before model loading",
    }, indent=2))


if __name__ == "__main__":
    main()
