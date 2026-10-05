"""CPU contracts for conditional Fisher ChemAware feature maps."""
from __future__ import annotations

import numpy as np

from chemaware_conditional_fisher_core import (
    fit_conditional_fisher_map,
    permute_identity_formulas,
)
from chemaware_shrinkage_whitening_core import apply_whitener


def main() -> None:
    rng = np.random.default_rng(91)
    formulas = np.repeat(np.asarray([f"F{i}" for i in range(30)]), 4)
    identities = np.asarray([f"I{i}" for i in range(len(formulas))])
    identity_rows = np.repeat(identities, 5)
    formula_rows = np.repeat(formulas, 5)
    identity_signal = rng.normal(size=(len(identities), 1))
    centroid = np.column_stack((
        3.0 * identity_signal[:, 0],
        0.05 * rng.normal(size=len(identities)),
        rng.normal(size=len(identities)),
    ))
    noise = np.column_stack((
        0.05 * rng.normal(size=len(identity_rows)),
        2.0 * rng.normal(size=len(identity_rows)),
        0.3 * rng.normal(size=len(identity_rows)),
    ))
    feature = np.repeat(centroid, 5, axis=0) + noise
    mean, transform, report = fit_conditional_fisher_map(
        feature, identity_rows, formula_rows, 0.10, 2,
    )
    assert transform.shape == (3, 2)
    assert report["eligible_multi_identity_formulas"] == 30
    assert report["replicated_identities"] == 120
    mapped = apply_whitener(feature, mean, transform)
    assert mapped.shape == (len(feature), 2) and np.isfinite(mapped).all()
    gram = mapped @ mapped.T
    assert np.linalg.eigvalsh(gram).min() > -1e-4
    assert abs(float(transform[0, 0])) > abs(float(transform[1, 0]))
    _, global_transform, global_report = fit_conditional_fisher_map(
        feature, identity_rows, formula_rows, 0.10, 3, between_mode="global",
    )
    assert global_transform.shape == (3, 3)
    assert global_report["between_mode"] == "global"
    permuted, permutation_report = permute_identity_formulas(formulas, 13)
    assert permutation_report["formula_counts_preserved_exactly"]
    assert np.array_equal(np.unique(permuted, return_counts=True)[1], np.full(30, 4))
    assert np.mean(permuted != formulas) > 0.8
    try:
        fit_conditional_fisher_map(feature, identity_rows, formula_rows, 0.1, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range rank was accepted")
    print("PASS: 9 ChemAware conditional-Fisher contracts")


if __name__ == "__main__":
    main()
