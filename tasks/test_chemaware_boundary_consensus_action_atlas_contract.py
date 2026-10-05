"""Static fail-closed contracts for the B1 audit and Slurm entrypoint."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tasks/audit_chemaware_boundary_consensus_action_atlas.py"
SBATCH = ROOT / "tasks/run_chemaware_boundary_consensus_action_atlas.sbatch"


def main() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    ast.parse(source)
    required_scientific_contracts = (
        "same_formula_negatives_only",
        "official_dreams_boundary_frozen_across_arms",
        "multi_negative_consensus",
        "weak_evidence_abstains",
        "precursor_region_excluded",
        "base_peak_and_max_normalization_fixed",
        "candidate_role_reversed",
        "intensity_rank_permuted",
        "direction_reversed",
        "pair_logratio_scale_invariant_action",
        "direction_reversed_same_peak_control",
        "formal_training_authorized",
        "delta_micro_auc",
        "delta_macro_auc",
    )
    missing = [value for value in required_scientific_contracts if value not in source]
    assert not missing, missing
    assert "RECALL_K = (1, 5, 10, 20, 50)" in source

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --gpus=" not in sbatch.replace("#SBATCH --gpus=1", "")
    assert "--mem" not in sbatch and "--mem-per-cpu" not in sbatch
    assert (
        "python -u tasks/audit_chemaware_boundary_consensus_action_atlas.py" in sbatch
    )
    assert "--preflight-only" in sbatch
    assert "run_${SLURM_JOB_ID}" in sbatch
    print("PASS: ChemAware B1 action-atlas and one-GPU sbatch contracts")


if __name__ == "__main__":
    main()
