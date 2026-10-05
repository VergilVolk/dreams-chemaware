"""Pure statistical contracts for empirical structure-rule mining."""
from __future__ import annotations

import numpy as np

from mine_chemaware_structure_conditioned_rules import (
    bh_qvalues, matched_formula_differences, nearest_detect, sign_flip_pvalue,
)


def main() -> None:
    detected = nearest_detect(
        np.asarray([18.0104, 44.0, 100.0]),
        np.asarray([18.0106, 43.0, 100.01]),
        np.asarray([0.001, 0.1, 0.02]),
    )
    assert detected.tolist() == [True, False, True]
    observation = np.asarray([
        [1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ])
    predicate = np.asarray([1, 0, 1, 0], dtype=bool)
    formula = np.asarray(["A", "A", "B", "B"])
    groups, difference, positive, negative = matched_formula_differences(
        observation, predicate, formula, np.arange(4),
    )
    assert groups.tolist() == ["A", "B"]
    assert np.allclose(difference, [[1, 0], [1, 0]])
    assert positive == negative == 2
    assert sign_flip_pvalue(np.ones(20), 1, 10_000) < 0.001
    qvalue = bh_qvalues(np.asarray([0.001, 0.01, 0.5]))
    assert np.allclose(qvalue, [0.003, 0.015, 0.5])
    print("structure-conditioned empirical rule contracts passed")


if __name__ == "__main__":
    main()
