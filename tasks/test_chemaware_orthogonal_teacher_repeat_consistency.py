from __future__ import annotations

import numpy as np

from audit_chemaware_orthogonal_teacher_repeat_consistency import (
    agreement,
    pair_indices,
    permutation_agreement,
    repeat_panel,
)


def main() -> None:
    pool = np.arange(9, dtype=np.int64)
    identity = np.asarray(["a", "a", "a", "b", "b", "c", "d", "d", "d"])
    panel = repeat_panel(pool, identity, identities=2, spectra_per_identity=2, seed=7)
    chosen, count = np.unique(identity[panel], return_counts=True)
    assert len(chosen) == 2
    assert np.array_equal(count, np.asarray([2, 2]))

    values = np.asarray([1, 1, 0, 0, 1])
    groups = np.asarray(["a", "a", "b", "b", "b"])
    left, right = pair_indices(groups)
    assert len(left) == 4
    assert agreement(values, left, right) == 0.5

    formula = np.asarray(["f", "f", "f", "f", "f"])
    result = permutation_agreement(
        values, formula, left, right, permutations=200, seed=11,
    )
    assert result["repeat_pairs"] == 4
    assert result["observed_agreement"] == 0.5
    assert 0 <= result["empirical_p_null_ge_observed"] <= 1
    print("PASS: ChemAware orthogonal-teacher repeat-consistency contracts")


if __name__ == "__main__":
    main()
