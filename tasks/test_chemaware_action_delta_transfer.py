"""CPU contracts for qualified-action effect transfer."""
from __future__ import annotations

import numpy as np
import torch

from chemaware_action_delta_transfer_core import (
    action_delta_target,
    clean_inherits_action_delta_loss,
    optimizer_descent_geometry,
    role_calibrated_dose,
)


def fixtures():
    clean = torch.tensor([1.0, 0.0], requires_grad=True)
    action = torch.tensor([0.8, 0.6], requires_grad=True)
    reference = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    return clean, action, reference, np.asarray([0, 2, 4], dtype=np.int64)


def test_only_live_clean_receives_chemical_gradient() -> None:
    clean, action, reference, ptr = fixtures()
    loss, _ = clean_inherits_action_delta_loss(
        clean, clean.detach(), action, reference, ptr, alpha=0.5,
    )
    loss.backward()
    assert clean.grad is not None and float(torch.linalg.vector_norm(clean.grad)) > 0
    assert action.grad is None
    assert reference.grad is None


def test_exact_target_reproduction_is_zero() -> None:
    clean, action, reference, ptr = fixtures()
    target = action_delta_target(clean.detach(), action.detach(), reference.detach(), ptr)
    # For this two-dimensional fixture, using the action query itself exactly
    # reproduces the full-dose action score displacement.
    loss, values = clean_inherits_action_delta_loss(
        action.detach(), clean.detach(), action.detach(), reference.detach(), ptr,
        alpha=1.0,
    )
    assert torch.allclose(values["student_delta"], target, atol=1e-7)
    assert float(loss) == 0.0


def test_clean_duplicate_target_is_exact_zero() -> None:
    clean, _action, reference, ptr = fixtures()
    target = action_delta_target(clean.detach(), clean.detach(), reference.detach(), ptr)
    assert torch.equal(target, torch.zeros_like(target))


def test_candidate_common_offset_is_unidentifiable() -> None:
    clean, action, reference, ptr = fixtures()
    target = action_delta_target(clean.detach(), action.detach(), reference.detach(), ptr)
    shifted = target + 7.0
    assert torch.allclose(shifted - shifted.mean(), target, atol=1e-7)


def test_role_dose_is_data_calibrated_and_unsupported_is_zero() -> None:
    role = np.asarray([1, 1, 2, 2, 0, 4])
    gain = np.asarray([0.10, 0.30, 0.05, 0.40, 2.0, 2.0])
    dose = role_calibrated_dose(role, gain, 1, 2)
    assert np.allclose(dose[:2], 1.0)
    assert np.isclose(dose[2], 0.25)
    assert np.isclose(dose[3], 1.0)
    assert np.array_equal(dose[4:], np.zeros(2, dtype=np.float32))


def test_real_adamw_descent_geometry_is_group_resolved() -> None:
    backbone = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    head = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    optimizer = torch.optim.AdamW([
        {"params": [backbone], "lr": 1e-3},
        {"params": [head], "lr": 2e-3},
    ], weight_decay=0.0)
    loss = torch.sum(backbone ** 2) + 0.5 * torch.sum(head ** 2)
    loss.backward()
    before = [backbone.detach().clone(), head.detach().clone()]
    optimizer.step()
    geometry = optimizer_descent_geometry(before, [backbone], [head])
    for group in ("all", "backbone", "head"):
        assert geometry[group]["postclip_gradient_norm"] > 0
        assert geometry[group]["descent_update_norm"] > 0
        assert geometry[group]["gradient_update_cosine"] > 0
        assert geometry[group]["first_order_descent"] > 0


if __name__ == "__main__":
    test_only_live_clean_receives_chemical_gradient()
    test_exact_target_reproduction_is_zero()
    test_clean_duplicate_target_is_exact_zero()
    test_candidate_common_offset_is_unidentifiable()
    test_role_dose_is_data_calibrated_and_unsupported_is_zero()
    test_real_adamw_descent_geometry_is_group_resolved()
    print("PASS: 6 qualified ChemAware action-delta transfer contracts")
