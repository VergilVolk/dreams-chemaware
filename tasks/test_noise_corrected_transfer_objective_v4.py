"""CPU tests for v4's mass-neutral transfer-target allocator."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from noise_corrected_direct_v3_core import _family_equal_weighted_edges
from noise_corrected_transfer_objective_v4 import (
    v3_hard_capped_transfer_delta,
    v4_mass_neutral_transfer_delta,
)


def fixture() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    delta = torch.tensor([
        [0.02, 0.08, 0.40],
        [0.03, 0.20, 0.80],
        [0.01, 0.12, 0.50],
        [0.04, 0.30, 0.60],
    ], dtype=torch.float64)
    mask = torch.tensor([
        [True, True, True],
        [True, True, True],
        [True, True, False],
        [True, True, True],
    ])
    groups = [
        "N::source_a|family_a", "N::source_a|family_a",
        "P::source_b|family_b", "P::source_b|family_b",
    ]
    return delta, mask, groups


def test_preserves_hard_cap_mass_inside_every_leaf() -> None:
    delta, mask, groups = fixture()
    old = v3_hard_capped_transfer_delta(delta, mask, hard_cap=0.10)
    new = v4_mass_neutral_transfer_delta(delta, mask, groups, hard_cap=0.10)
    for leaf in sorted(set(groups)):
        rows = torch.tensor([value == leaf for value in groups])
        assert torch.allclose(old[rows].sum(), new[rows].sum(), atol=1e-12)


def test_retains_more_strong_edge_resolution_without_inverting_order() -> None:
    delta, mask, groups = fixture()
    old = v3_hard_capped_transfer_delta(delta, mask, hard_cap=0.10)
    new = v4_mass_neutral_transfer_delta(delta, mask, groups, hard_cap=0.10)
    assert torch.unique(new[new > 0]).numel() > torch.unique(old[old > 0]).numel()
    for leaf in sorted(set(groups)):
        rows = torch.tensor([value == leaf for value in groups])
        x = delta[rows][mask[rows]].flatten()
        y = new[rows][mask[rows]].flatten()
        order = torch.argsort(x)
        assert torch.all(torch.diff(y[order]) >= -1e-12)
    assert float(new.max()) <= 0.20 + 1e-12


def test_linear_huber_transfer_gradient_mass_is_preserved() -> None:
    delta, mask, groups = fixture()
    old = v3_hard_capped_transfer_delta(delta, mask, hard_cap=0.10)
    new = v4_mass_neutral_transfer_delta(delta, mask, groups, hard_cap=0.10)

    def gradient(allocated: torch.Tensor) -> torch.Tensor:
        clean = torch.zeros(delta.shape[1], dtype=torch.float64, requires_grad=True)
        target = clean.detach().unsqueeze(0) + 0.50 * allocated
        edge_loss = F.smooth_l1_loss(
            clean.unsqueeze(0).expand_as(target), target,
            beta=0.10, reduction="none",
        )
        loss = _family_equal_weighted_edges(
            edge_loss, mask.to(edge_loss.dtype), groups,
        )
        return torch.autograd.grad(loss, clean)[0]

    # Individual candidate edges become distinguishable, but the aggregate
    # transfer push remains identical in v3's linear Huber region.
    assert torch.allclose(gradient(old).sum(), gradient(new).sum(), atol=1e-12)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_transfer_objective_v4] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
