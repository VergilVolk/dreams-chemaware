"""CPU contracts for the Torch direct-prior objective."""

from __future__ import annotations

import numpy as np
import torch

from chemaware_direct_prior_objective import direct_prior_loss_and_gradient
from chemaware_direct_prior_torch import (
    candidate_centered_residual_loss,
    select_official_boundary_batch,
)


def main() -> None:
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    official = np.asarray([0.4, 0.3, 0.1, 0.6, 0.2], dtype=np.float64)
    prior = np.asarray([0.2, -0.05, -0.15, 0.1, -0.1], dtype=np.float64)
    student = np.asarray([0.43, 0.28, 0.09, 0.57, 0.23], dtype=np.float64)

    expected_loss, expected_gradient = direct_prior_loss_and_gradient(
        student, official, prior, ptr, np.asarray([True, True]),
        alpha=0.5, huber_delta=0.02,
    )
    value = torch.tensor(student, dtype=torch.float64, requires_grad=True)
    loss = candidate_centered_residual_loss(
        value,
        torch.tensor(official, dtype=torch.float64),
        torch.tensor(prior, dtype=torch.float64),
        ptr,
        alpha=0.5,
        huber_delta=0.02,
    )
    loss.backward()
    assert abs(float(loss.detach()) - expected_loss) < 1e-12
    assert np.allclose(value.grad.detach().numpy(), expected_gradient, atol=1e-12)
    assert abs(float(value.grad[:3].sum())) < 1e-12
    assert abs(float(value.grad[3:].sum())) < 1e-12

    clean = torch.tensor(student, dtype=torch.float64, requires_grad=True)
    clean_loss = candidate_centered_residual_loss(
        clean,
        torch.tensor(official, dtype=torch.float64),
        torch.tensor(prior, dtype=torch.float64),
        ptr,
        alpha=0.0,
        huber_delta=0.02,
    )
    clean_loss.backward()
    assert float(clean_loss.detach()) == 0.0
    assert torch.count_nonzero(clean.grad) == 0

    shifted = torch.tensor(student + np.repeat([3.0, -4.0], [3, 2]), dtype=torch.float64)
    shifted_loss = candidate_centered_residual_loss(
        shifted,
        torch.tensor(official, dtype=torch.float64),
        torch.tensor(prior, dtype=torch.float64),
        ptr,
        alpha=0.5,
        huber_delta=0.02,
    )
    assert abs(float(shifted_loss) - float(loss.detach())) < 1e-12

    # The official top reference is mandatory even when RNG would otherwise
    # select a different spectrum.  The second reference remains stochastic.
    body = {
        "query_row": np.asarray([10]),
        "query_ptr": np.asarray([0, 2]),
        "molecule_ptr": np.asarray([0, 3, 5]),
        "molecule_label": np.asarray([1, 0]),
        "pair_candidate_row": np.asarray([20, 21, 22, 30, 31]),
    }
    rows = np.asarray([10, 20, 21, 22, 30, 31])
    row_position = {int(row): index for index, row in enumerate(rows)}
    official_embedding = np.asarray(
        [[1.0, 0.0], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]],
        dtype=np.float32,
    )
    batch = select_official_boundary_batch(
        body, np.asarray([0]), row_position, official_embedding, 2,
        np.random.default_rng(7),
    )
    selected_positions = batch["reference_cache"][batch["reference_edge"]]
    first = selected_positions[batch["reference_ptr"][0] : batch["reference_ptr"][1]]
    second = selected_positions[batch["reference_ptr"][1] : batch["reference_ptr"][2]]
    assert row_position[21] in first
    assert row_position[31] in second
    print("PASS: ChemAware Torch candidate-centred residual contracts")


if __name__ == "__main__":
    main()
