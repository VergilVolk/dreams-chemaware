from __future__ import annotations

import numpy as np
import torch

from chemaware_candidate_residual_distillation_core import (
    event_calibrated_rankmax_loss,
    logmeanexp_segments,
    teacher_boundary_events,
)


def main() -> None:
    events = teacher_boundary_events(
        baseline_rank=np.asarray([2, 1, 3, 2]),
        proposal_rank=np.asarray([[1, 2], [2, 1], [2, 3], [1, 2]]),
        baseline_candidate=np.asarray([1, 0, 2, 1]),
        proposed_candidate=np.asarray([[0, 2], [1, 2], [1, 2], [2, 0]]),
        selected_slot=np.asarray([0, 0, 0, -1]),
        positive_candidate=np.asarray([0, 0, 0, 0]),
    )
    assert events.role.tolist() == [1, 2, 0, 0]
    assert events.positive.tolist() == [0, 0, -1, -1]
    assert events.negative.tolist() == [1, 1, -1, -1]

    pair = torch.tensor([0.2, 0.2, 0.5, 0.1, 0.5], requires_grad=True)
    molecule = logmeanexp_segments(pair, np.asarray([0, 2, 5]), 0.05)
    expected_second = 0.05 * torch.logsumexp(
        torch.tensor([0.5, 0.1, 0.5]) / 0.05, dim=0,
    ) - 0.05 * torch.log(torch.tensor(3.0))
    assert torch.allclose(molecule, torch.stack((torch.tensor(0.2), expected_second)), atol=1e-6)
    molecule.sum().backward()
    assert pair.grad is not None and torch.all(pair.grad > 0)

    official = torch.tensor([
        [0.40, 0.51, -10.0],  # correction needs 0.005 - (-0.11) = 0.115
        [0.60, 0.55, -10.0],  # protection retains 0.05 - 0.002 = 0.048
        [0.30, 0.40, 0.20],   # unsupported: exact zero weight
    ])
    student = official.clone().requires_grad_(True)
    valid = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    loss, audit = event_calibrated_rankmax_loss(
        student, official, valid,
        positive=torch.tensor([0, 0, -1]),
        negative=torch.tensor([1, 1, -1]),
        role=torch.tensor([1, 2, 0]),
        weight=torch.ones(3),
    )
    assert torch.allclose(audit["required_residual"][:2], torch.tensor([0.115, -0.002]), atol=1e-6)
    assert float(loss) > 0
    loss.backward()
    assert student.grad is not None
    assert student.grad[0, 0] < 0 and student.grad[0, 1] > 0
    assert torch.all(student.grad[2] == 0)

    satisfied = official.clone()
    satisfied[0, 0] += 0.12
    zero, _ = event_calibrated_rankmax_loss(
        satisfied, official, valid,
        positive=torch.tensor([0, 0, -1]),
        negative=torch.tensor([1, 1, -1]),
        role=torch.tensor([1, 2, 0]),
        weight=torch.ones(3),
    )
    assert float(zero) == 0.0
    print("PASS: ChemAware event-calibrated candidate-residual contracts")


if __name__ == "__main__":
    main()
