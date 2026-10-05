"""Static fail-closed contracts for the Jacobian action audit and sbatch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tasks/audit_chemaware_jacobian_intersection_actions.py"
SBATCH = ROOT / "tasks/run_chemaware_jacobian_intersection_actions.sbatch"


def main() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "official_same_formula_boundary",
        "input_jacobian_variable",
        "chemistry_and_positive_local_gain_required",
        "base_peak_and_max_normalization_fixed",
        "controls_match_edit_count_factor_intensity_and_abs_jacobian",
        "direction_control_uses_same_peak_slots",
        "confirmation_computed_only_after_discovery_pass",
        "embedding_evaluation_fold_inspected",
        "reserve_fold_inspected",
        "formal_training_authorized",
        "perturbed_spectra_encoded",
    )
    missing = [value for value in required if value not in source]
    assert not missing, missing
    assert "SETTINGS = tuple(" in source
    assert "log_dose: float" in source

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --gpus=" not in sbatch.replace("#SBATCH --gpus=1", "")
    assert "--mem" not in sbatch and "--mem-per-cpu" not in sbatch
    assert "run_${SLURM_JOB_ID}" in sbatch
    assert "--preflight-only" in sbatch
    assert "python -u tasks/audit_chemaware_jacobian_intersection_actions.py" in sbatch
    print("PASS: ChemAware Jacobian-intersection audit and one-GPU sbatch contracts")


if __name__ == "__main__":
    main()
