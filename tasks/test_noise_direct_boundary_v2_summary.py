"""Small tests for the direct-boundary-v2 promotion statistics and sbatch."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from summarize_noise_direct_boundary_v2 import corrected_formula_ci


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    formulas = np.asarray([f"F{index // 4}" for index in range(80)])
    old = np.full(80, 2, dtype=int)
    new = np.ones(80, dtype=int)
    interval = corrected_formula_ci(
        old, new, formulas, resamples=2000, seed=7, family_size=3,
    )
    assert interval["mean"] == 1.0 and interval["ci_low"] == 1.0
    assert interval["bonferroni_family_size"] == 3

    sbatch = (ROOT / "tasks/run_noise_final_direct_boundary_v2_phase_a.sbatch").read_text(
        encoding="utf-8",
    )
    required = (
        "#SBATCH --gpus=1", "fold_${FOLD}_run_${SLURM_JOB_ID}",
        "--corrective-query-scope errors", "--candidate-boundary-version v2_molecule_max",
        "--batch-actions 9", "--refresh-hard-negatives",
        "SEEDS=(20260906 20260907 20260908)",
        "summarize_noise_direct_boundary_v2.py",
    )
    for token in required:
        assert token in sbatch, token
    print("[test_noise_direct_boundary_v2_summary] PASS tests=2")


if __name__ == "__main__":
    main()
