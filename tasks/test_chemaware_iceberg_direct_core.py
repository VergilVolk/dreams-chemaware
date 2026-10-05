from __future__ import annotations

import numpy as np
import torch

from audit_chemaware_iceberg_teacher_headroom import QueryRecord, select_balanced
from chemaware_iceberg_direct_core import (
    action_prediction_indices, assert_formula_disjoint,
    identity_balanced_positions, molecule_scores_from_pairs, rescue_positions,
    stable_formula_folds,
)


def record(index: int, identity: str, rank: int) -> QueryRecord:
    return QueryRecord(
        query_index=index, query_row=index, formula=f"F{index % 5}", identity=identity,
        official_rank=rank, official_margin=float(index) / 100,
        candidate_molecules=(0, 1), candidate_rows=(0, 1), smiles=("C", "CC"),
        collision_energy=20.0, precursor_mz=100.0, adduct="[M+H]+", instrument="Orbitrap",
    )


def main() -> None:
    # Regression for the former cross-stratum duplicate-identity bug.
    records = [record(0, "shared", 2), record(1, "error2", 2),
               record(2, "shared", 1), record(3, "correct2", 1),
               record(4, "correct3", 1)]
    selected = select_balanced(records, 4)
    assert len({value.identity for value in selected}) == 4

    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    assert np.array_equal(action_prediction_indices(ptr, "correct_synthetic"), [0, 3])
    assert np.array_equal(action_prediction_indices(ptr, "peak_permuted"), [0, 3])
    assert np.array_equal(action_prediction_indices(ptr, "candidate_swapped"), [2, 4])

    formulas = np.asarray(["A", "B", "C", "D", "E", "F"])
    folds = stable_formula_folds(formulas, 5, 17)
    assert_formula_disjoint(formulas, folds, 1, 4)
    official = np.asarray([2, 1, 3, 2, 1, 2])
    teacher = np.asarray([1, 1, 1, 2, 1, 1])
    positions = rescue_positions(official, teacher, folds, 1, 4)
    assert np.all(official[positions] != 1) and np.all(teacher[positions] == 1)

    balanced = identity_balanced_positions(
        np.asarray([0, 1, 2, 3]), np.asarray(["x", "x", "y", "z"]), 11,
    )
    assert len(balanced) == 3

    scores = torch.tensor([0.1, 0.4, 0.2, 0.3], requires_grad=True)
    molecule = molecule_scores_from_pairs(scores, np.asarray([0, 2, 3, 4]))
    assert torch.allclose(molecule, torch.tensor([0.4, 0.2, 0.3]))
    molecule.sum().backward()
    assert torch.allclose(scores.grad, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    print("[test_chemaware_iceberg_direct_core] PASS")


if __name__ == "__main__":
    main()
