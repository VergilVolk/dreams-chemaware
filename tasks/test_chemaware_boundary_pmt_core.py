"""Executable contracts for ChemAware B-PMT primitives."""

import numpy as np
import torch

from chemaware_boundary_pmt_core import (
    active_margin_transfer_loss,
    active_transfer_weights,
    inherited_clean_margin_target,
    margin_bin_formula_derangement,
    margin_bin_formula_derangement_indices,
    strict_action_advantage,
)


def main() -> None:
    old = np.asarray([-0.1, 0.01, 0.2, 0.0])
    correct = np.asarray([0.1, 0.06, 0.4, -0.1])
    controls = np.asarray([[0.0, 0.03, 0.1, -0.2], [-0.05, 0.02, 0.3, -0.3]])
    advantage = strict_action_advantage(old, correct, controls)
    assert np.allclose(advantage, [0.1, 0.03, 0.1, 0.0])
    weight = active_transfer_weights(
        old, advantage, np.ones(4), activation_margin=0.05, advantage_cap=0.05
    )
    assert np.allclose(weight, [1.0, 0.6, 0.0, 0.0])
    initial = torch.tensor([-0.1, 0.01])
    gain = torch.tensor([0.1, 0.03])
    target = inherited_clean_margin_target(initial, gain, 0.5, 0.05)
    assert torch.allclose(target, torch.tensor([-0.075, 0.025]))
    current = initial.clone().requires_grad_(True)
    loss = active_margin_transfer_loss(current, target, torch.tensor([1.0, 0.0]), 0.05)
    loss.backward()
    assert current.grad[0] < 0 and current.grad[1] == 0
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    permuted = margin_bin_formula_derangement(
        values,
        np.asarray([-0.1, -0.05, 0.0, 0.05]),
        np.asarray(["A", "B", "C", "D"]),
        seed=7,
        bins=2,
    )
    assert np.allclose(np.sort(values), np.sort(permuted))
    source = margin_bin_formula_derangement_indices(
        np.asarray([-0.1, -0.05, 0.0, 0.05]),
        np.asarray(["A", "B", "C", "D"]),
        seed=7,
        bins=2,
    )
    assert np.all(np.asarray(["A", "B", "C", "D"])[source] != np.asarray(["A", "B", "C", "D"]))
    print("PASS: ChemAware B-PMT mathematical contracts")


if __name__ == "__main__":
    main()
