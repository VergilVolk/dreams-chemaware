"""Static contracts for the bounded one-GPU ChemAware residual Phase A."""

from pathlib import Path

import numpy as np

from summarize_chemaware_iceberg_residual_shared_phase_a import pairwise_metrics


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    text = (ROOT / "tasks/run_chemaware_iceberg_residual_shared_phase_a.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --mem=" not in text
    assert "#SBATCH --mem-per-cpu=" not in text
    assert "LABELS=(clean_duplicate correct_alpha050 structure_alpha050 peak_alpha050)" in text
    assert "ALPHAS=(0.0 0.50 0.50 0.50)" in text
    assert "build_chemaware_iceberg_corrective_residual_ledger.py" in text
    assert "audit_chemaware_direct_chemical_prior_utility.py" in text
    assert "--action-ledger-dir \"$LEDGER_DIR\"" in text
    assert 'ledger["strict_corrective_actions"] == 272' in text
    assert 'row["arms"]["correct"]["introduced"] == 0' in text
    assert "--inner-fold 3 --outer-fold 4" in text
    assert "--references-per-molecule 2" in text
    assert "summarize_chemaware_iceberg_residual_shared_phase_a.py" in text
    treatment = {
        "new_rank": np.asarray([1, 2, 1, 3]),
        "candidate_count": np.asarray([4, 4, 5, 5]),
        "formula": np.asarray(["A", "B", "C", "D"]),
    }
    comparator = {
        "new_rank": np.asarray([2, 2, 3, 4]),
        "candidate_count": treatment["candidate_count"],
        "formula": treatment["formula"],
    }
    paired = pairwise_metrics(treatment, comparator, 7, 10_000)
    assert paired["delta_recall1"] == 0.5
    assert paired["treatment_wins_top1"] == 2
    assert paired["comparator_wins_top1"] == 0
    assert paired["delta_mrr"] > 0
    assert paired["delta_macro_auc"] > 0
    assert paired["delta_micro_auc"] > 0
    print("PASS: ChemAware ICEBERG residual one-GPU Phase-A contracts")


if __name__ == "__main__":
    main()
