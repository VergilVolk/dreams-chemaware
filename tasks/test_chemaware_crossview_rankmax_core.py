from __future__ import annotations

import numpy as np
import torch

from chemaware_crossview_rankmax_core import (
    boundary_coordinate_gradients,
    boundary_symmetric_operator_gradients,
    gradient_coherence_report,
    gradient_transfer_report,
    identity_balanced_weights,
    identity_risk_route,
    matched_error_identities,
    rankmax_top1_loss,
)


def test_boundary_coordinate_gradients() -> None:
    fixed = np.asarray([[0.2, 0.4, 0.1], [0.6, 0.2, 0.5]])
    product = np.asarray([
        [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
        [[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]],
    ])
    valid = np.ones_like(fixed, dtype=bool)
    gradient, negative = boundary_coordinate_gradients(
        fixed, product, valid, np.asarray([0, 2]), 0.5,
    )
    assert np.array_equal(negative, np.asarray([1, 0]))
    assert np.allclose(gradient, np.asarray([[-1.0, -1.5], [4.5, 5.0]]))


def test_boundary_symmetric_operator_gradients() -> None:
    query = np.asarray([[1.0, 2.0]])
    positive = np.asarray([[3.0, 5.0]])
    negative = np.asarray([[1.0, 1.0]])
    observed = boundary_symmetric_operator_gradients(query, positive, negative, 0.5)
    expected = 0.5 * np.asarray([[[2.0, 4.0], [4.0, 8.0]]])
    assert np.allclose(observed.reshape(1, 2, 2), expected)


def test_gradient_coherence_report() -> None:
    gradient = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0],
    ])
    identity = np.asarray(["a", "a", "b", "c"])
    report = gradient_coherence_report(
        gradient, identity, np.asarray([True, True, True, False]),
    )
    assert report["queries"] == 3 and report["identities"] == 2
    assert report["mean_pairwise_cosine"] > 0.99
    assert report["negative_pair_fraction"] == 0.0


def test_gradient_transfer_report() -> None:
    gradient = np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.2], [-1.0, 0.0]])
    identity = np.asarray(["a", "b", "c", "d"])
    report = gradient_transfer_report(
        gradient, identity,
        np.asarray([True, True, False, False]),
        np.asarray([False, False, True, True]),
    )
    assert report["source_identities"] == 2 and report["target_identities"] == 2
    assert report["negative_target_fraction"] == 0.5


def main() -> None:
    test_boundary_coordinate_gradients()
    test_boundary_symmetric_operator_gradients()
    test_gradient_coherence_report()
    test_gradient_transfer_report()
    identity = np.asarray(["A", "A", "B", "B", "C", "C"])
    scope = np.ones(6, dtype=bool)
    base = np.asarray([2, 1, 2, 1, 2, 2])
    correct = np.asarray(["A", "", "B", "X", "", ""])
    control = np.asarray(["", "", "B", "", "C", ""])
    selected, audit = identity_risk_route(identity, scope, base, correct, control)
    assert selected.tolist() == ["A"]
    assert audit["A"]["correct_risk"] == 1
    weight = identity_balanced_weights(identity, np.isin(identity, ["A", "C"]))
    assert np.isclose(weight.sum(), 1.0)
    assert np.isclose(weight[identity == "A"].sum(), 0.5)
    assert np.isclose(weight[identity == "C"].sum(), 0.5)
    matched = matched_error_identities(
        np.asarray(["A"]), identity, np.asarray(["F1", "F1", "F2", "F2", "F3", "F3"]),
        scope, base > 1, np.asarray([-.10, .2, -.11, .3, -.5, -.4]),
    )
    assert matched.tolist() == ["B"]
    safe = rankmax_top1_loss(
        torch.tensor([.5]), torch.tensor([[.1, .2]]), torch.tensor([1.]), .05,
    )
    unsafe = rankmax_top1_loss(
        torch.tensor([.1]), torch.tensor([[.2, .0]]), torch.tensor([1.]), .05,
    )
    assert float(safe) == 0.0 and float(unsafe) > 0.0
    print("PASS: ChemAware cross-view Rankmax core contracts")


if __name__ == "__main__":
    main()
