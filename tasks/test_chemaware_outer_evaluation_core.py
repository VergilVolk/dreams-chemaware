from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_outer_evaluation_core import (  # noqa: E402
    auc_metrics,
    binary_auc,
    evaluate_selections,
    formula_cluster_ci,
    paired_comparison,
    retrieval_metrics,
)


def main() -> None:
    ptr = np.asarray([0, 3, 6], dtype=np.int64)
    score = np.asarray([0.8, 0.7, 0.6, 0.7, 0.8, 0.6])
    label = np.asarray([False, True, False, True, False, False])
    selections = {
        "correct": np.asarray([1, -1]),
        "harmful": np.asarray([-1, 1]),
    }
    baseline, ranks, proposed = evaluate_selections(ptr, score, label, selections)
    assert baseline.tolist() == [2, 2]
    assert ranks["correct"].tolist() == [1, 2]
    assert ranks["harmful"].tolist() == [2, 2]
    metric = retrieval_metrics(baseline, ranks["correct"])
    assert metric["corrected_at_1"] == 1 and metric["introduced_at_1"] == 0
    assert metric["delta_recall1"] == 0.5
    assert binary_auc(np.asarray([0.0, 1.0, 1.0]), np.asarray([False, True, False])) == 0.75
    auc = auc_metrics(ptr, proposed["correct"], label)
    assert auc["micro_auc"] > 0.5 and auc["macro_auc_queries"] == 2
    formula = np.asarray(["A", "B"])
    ci = formula_cluster_ci(formula, ranks["correct"], baseline, draws=200, seed=7)
    assert ci[0] >= 0
    comparison = paired_comparison(ranks["correct"], baseline, formula, draws=200, seed=8)
    assert comparison["delta_recall1"] == 0.5
    print("PASS: ChemAware sealed-outer evaluation metric contracts")


if __name__ == "__main__":
    main()
