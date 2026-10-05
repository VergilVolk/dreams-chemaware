"""CPU contracts for pairwise ChemAware PSD metric updates."""
from __future__ import annotations

import numpy as np

from chemaware_pairwise_psd_core import (
    compose_psd_update,
    cross_split_coordinate_consensus,
    cross_split_spectral_consensus,
    formula_balanced_pair_operator,
    unit_rows,
)


def main() -> None:
    rng = np.random.default_rng(117)
    query = unit_rows(rng.normal(size=(120, 6)))
    positive = unit_rows(query + 0.3 * rng.normal(size=query.shape))
    negative = unit_rows(rng.normal(size=query.shape))
    formula = np.repeat(np.asarray([f"F{i}" for i in range(30)]), 4)
    first, first_report = formula_balanced_pair_operator(
        query[:60], positive[:60], negative[:60], formula[:60],
    )
    second, second_report = formula_balanced_pair_operator(
        query[60:], positive[60:], negative[60:], formula[60:],
    )
    assert first.shape == (6, 6) and np.allclose(first, first.T)
    assert first_report["formulas"] == second_report["formulas"] == 15
    consensus, report = cross_split_spectral_consensus(first, second)
    assert consensus.shape == (6, 6) and np.allclose(consensus, consensus.T, atol=1e-6)
    assert np.linalg.eigvalsh(consensus).min() >= -1.00001
    base = np.eye(6, dtype=np.float32)
    transform, transform_report = compose_psd_update(base, consensus, 0.9)
    gram_metric = transform @ transform.T
    assert np.linalg.eigvalsh(gram_metric).min() > 0.09
    assert transform_report["metric_min_eigenvalue"] > 0.09
    unchanged, unchanged_report = compose_psd_update(base, consensus, 0.0)
    assert np.array_equal(unchanged, base) and unchanged_report["eta"] == 0.0
    assert report["first_order_alignment_first"] >= -1e-8
    assert report["first_order_alignment_second"] >= -1e-8
    coordinate, coordinate_report = cross_split_coordinate_consensus(first, second)
    assert coordinate.shape == first.shape and np.allclose(coordinate, coordinate.T, atol=1e-6)
    assert coordinate_report["fixed_rule_coordinate_basis"]
    assert 0 <= coordinate_report["reproduced_upper_triangle_fraction"] <= 1
    assert coordinate_report["first_order_alignment_first"] >= -1e-8
    assert coordinate_report["first_order_alignment_second"] >= -1e-8
    try:
        compose_psd_update(base, consensus, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-strict PSD update bound was accepted")
    print("PASS: 17 ChemAware pairwise-PSD contracts")


if __name__ == "__main__":
    main()
