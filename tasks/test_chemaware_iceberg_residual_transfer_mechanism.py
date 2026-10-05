"""CPU contracts for the post-hoc residual-transfer mechanism audit."""

from pathlib import Path

import numpy as np

from audit_chemaware_iceberg_residual_transfer_mechanism import (
    candidate_score_vectors,
    centered_residual,
    cosine_each_active,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ptr = np.asarray([0, 3, 5])
    official = np.asarray([0.4, 0.3, 0.1, 0.6, 0.2])
    student = np.asarray([0.5, 0.25, 0.05, 0.55, 0.25])
    residual = centered_residual(student, official, ptr)
    assert abs(float(np.sum(residual[:3]))) < 1e-12
    assert abs(float(np.sum(residual[3:]))) < 1e-12
    target = np.asarray([0.1, -0.05, -0.05, -0.1, 0.1])
    cosine = cosine_each_active(residual, target, ptr, np.asarray([True, True]))
    assert cosine.shape == (2,)
    assert np.all(np.isfinite(cosine))

    body = {
        "query_row": np.asarray([10]),
        "query_ptr": np.asarray([0, 2]),
        "molecule_ptr": np.asarray([0, 2, 3]),
        "molecule_label": np.asarray([1, 0]),
        "pair_candidate_row": np.asarray([20, 21, 30]),
    }
    rows = np.asarray([10, 20, 21, 30])
    old = np.asarray([[1.0, 0.0], [0.7, 0.3], [0.9, 0.1], [0.4, 0.6]])
    new = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
    score = candidate_score_vectors(body, rows, old, new)
    assert set(score) == {"official", "both", "query_only", "reference_only"}
    assert all(value.shape == (2,) for value in score.values())

    sbatch = (
        ROOT / "tasks/run_chemaware_iceberg_residual_transfer_mechanism_audit.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem=" not in sbatch
    assert "#SBATCH --mem-per-cpu=" not in sbatch
    assert "transfer_mechanism_audit.json" in sbatch
    print("PASS: ChemAware residual-transfer mechanism contracts")


if __name__ == "__main__":
    main()
