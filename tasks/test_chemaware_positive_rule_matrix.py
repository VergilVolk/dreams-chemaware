"""Small contracts for rule-matrix evaluation semantics."""

from __future__ import annotations

import numpy as np

from pilot_chemaware_positive_rule_matrix import fit_positive_matrix


def main() -> None:
    xtx = np.eye(2)
    xty = np.asarray([[1.0, -2.0], [-3.0, 4.0]])
    matrix = fit_positive_matrix(xtx, xty, ridge=1.0)
    assert np.array_equal(matrix, np.asarray([[0.5, 0.0], [0.0, 2.0]], dtype=np.float32))

    # A tied pair has AUC credit 0.5 but remains a strict Top-1 failure.  Keep
    # both semantics explicit because sparse chemical scores tie frequently.
    positive = 0.0
    negative = np.asarray([0.0, -1.0, 1.0])
    auc = np.mean(positive > negative) + 0.5 * np.mean(positive == negative)
    strict_rank = 1 + np.sum(negative >= positive)
    assert np.isclose(auc, 0.5)
    assert strict_rank == 3

    print("PASS: positive rule-matrix metric contracts")


if __name__ == "__main__":
    main()
