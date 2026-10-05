"""CPU contracts for exact cross-view candidate-pair evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_crossview_boundary_core import crossview_pair_boundary_proof  # noqa: E402
from build_chemaware_crossview_boundary_native_triplets import binary_formula_folds  # noqa: E402


def main() -> None:
    formula = np.asarray(["A", "B", "A", "C", "D", "E"])
    split = binary_formula_folds(formula, 3407)
    assert split[0] == split[2]
    assert set(split.tolist()) == {0, 1}

    candidates = np.asarray([["T", "F"], ["T", "F"], ["T", "F"]])
    valid = np.ones((3, 2), dtype=bool)
    utility = np.zeros((3, 3, 2), dtype=np.float64)
    utility[0, :, 0] = [999.0, 3.0, 3.0]
    utility[0, :, 1] = [-999.0, 1.0, 1.0]
    utility[1:, :, :] = 1.0
    proof = crossview_pair_boundary_proof(
        candidates, valid, utility,
        np.asarray(["I", "I", "I"]),
        np.asarray(["T", "T", "T"]),
        np.asarray(["F", "F", "F"]),
        absolute_threshold=1.0, aggregation="q25", min_context=1,
    )
    assert proof["eligible"].tolist() == [True, True, True]
    assert proof["boundary"][0, 0] == 2.0
    assert proof["specificity"][0] == 2.0
    # The held query's extreme utilities are never used in its own proof.
    assert proof["truth_score"][0, 0] == 3.0
    assert proof["false_score"][0, 0] == 1.0

    missing = valid.copy(); missing[1:, 1] = False
    no_false_context = crossview_pair_boundary_proof(
        candidates, missing, utility,
        np.asarray(["I", "I", "I"]),
        np.asarray(["T", "T", "T"]),
        np.asarray(["F", "F", "F"]),
        min_context=1,
    )
    assert not no_false_context["eligible"][0]

    rotated = crossview_pair_boundary_proof(
        candidates, valid, utility,
        np.asarray(["I", "I", "I"]),
        np.asarray(["T", "T", "T"]),
        np.asarray(["F", "F", "F"]),
        primary_arm=1, min_context=1,
    )
    assert not np.any(rotated["eligible"])

    # A baseline candidate is represented by the exact zero-action utility,
    # not dropped merely because challenger tables omit the baseline slot.
    baseline_utility = np.zeros_like(utility)
    baseline_utility[0, :, 1] = -2.0
    baseline = crossview_pair_boundary_proof(
        candidates, valid, baseline_utility,
        np.asarray(["I", "I", "I"]),
        np.asarray(["T", "T", "T"]),
        np.asarray(["F", "F", "F"]),
        baseline_identity=np.asarray(["T", "T", "T"]),
        absolute_threshold=1.0, min_context=1,
    )
    assert np.all(baseline["eligible"])
    assert np.allclose(baseline["boundary"][0], 2.0)
    print("PASS: ChemAware exact cross-view candidate-boundary contracts")


if __name__ == "__main__":
    main()
