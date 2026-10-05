"""Small invariants for routed-vs-control formula-cluster inference."""
from __future__ import annotations

import numpy as np

from summarize_noise_corrected_routed_direct_arms import formula_cluster_ci


def main() -> None:
    formulas = np.asarray(["A", "A", "B", "C", "C"], dtype=str)
    values = np.asarray([1, 0, 1, -1, 1], dtype=float)
    left = formula_cluster_ci(formulas, values, 1_000, 17)
    right = formula_cluster_ci(formulas, values, 1_000, 17)
    assert left == right
    assert left["formula_clusters"] == 3
    assert np.isclose(left["delta_pp"], 40.0)

    positive = formula_cluster_ci(
        np.asarray(["A", "B", "C", "D"], dtype=str),
        np.ones(4, dtype=float), 500, 23,
    )
    assert positive["ci_low_pp"] == 100.0
    assert positive["ci_high_pp"] == 100.0
    print("[noise corrected routed summary tests] PASS=6")


if __name__ == "__main__":
    main()
