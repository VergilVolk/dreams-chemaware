"""CPU contracts for ChemAware action-to-clean gradient routing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_action_transfer_core import (  # noqa: E402
    action_routed_clean_pair_loss,
    detached_gradients,
    first_order_metric_gain,
    gradient_cosine,
    molecule_max_scores,
    routed_listwise_loss,
    straight_through_value,
    subtract_gradients,
)


def test_molecule_scoring_matches_deployment_max_over_reference_rule() -> None:
    query = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([
        [0.40, 0.0], [0.80, 0.0],  # positive molecule: max 0.80
        [0.70, 0.0],               # first negative
        [0.20, 0.0], [0.60, 0.0],  # second negative: max 0.60
    ])
    scores = molecule_max_scores(query, candidates, [0, 2, 3, 5])
    assert torch.allclose(scores, torch.tensor([0.80, 0.70, 0.60]))


def test_straight_through_uses_action_forward_and_clean_backward() -> None:
    clean = torch.tensor([1.0, 2.0], requires_grad=True)
    action = torch.tensor([4.0, 8.0], requires_grad=True)
    routed = straight_through_value(action, clean)
    assert torch.equal(routed.detach(), action.detach())
    routed.sum().backward()
    assert torch.equal(clean.grad, torch.ones_like(clean))
    assert action.grad is None


def test_query_only_route_blocks_candidate_gradients() -> None:
    clean = torch.tensor([0.6, 0.8], requires_grad=True)
    action = torch.tensor([0.8, 0.6], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    result = routed_listwise_loss(
        clean, action, candidates, [0, 1, 2],
        route="action_forward_clean_backward", temperature=0.2,
    )
    result.loss.backward()
    assert clean.grad is not None and float(clean.grad.norm()) > 0
    assert action.grad is None
    assert candidates.grad is None
    assert torch.allclose(result.scores.detach(), candidates.detach() @ action.detach())


def test_raw_shared_route_reproduces_candidate_gradient_leakage() -> None:
    clean = torch.tensor([0.6, 0.8], requires_grad=True)
    action = torch.tensor([0.8, 0.6], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    result = routed_listwise_loss(
        clean, action, candidates, [0, 1, 2],
        route="raw_shared_action", temperature=0.2,
    )
    result.loss.backward()
    assert clean.grad is None
    assert action.grad is not None and float(action.grad.norm()) > 0
    assert candidates.grad is not None and float(candidates.grad.norm()) > 0


def test_action_routed_pair_selects_with_action_but_updates_only_clean_query() -> None:
    clean = torch.tensor([0.0, 1.0], requires_grad=True)
    action = torch.tensor([1.0, 0.0], requires_grad=True)
    candidates = torch.tensor([
        [0.9, 0.0], [0.1, 0.8],  # positive molecule
        [0.8, 0.0], [0.0, 0.7],  # negatives
    ], requires_grad=True)
    result = action_routed_clean_pair_loss(
        clean, action, candidates, [0, 2, 3, 4], temperature=0.2,
    )
    assert result.positive_reference == 0
    assert result.negative_reference == 2
    result.loss.backward()
    assert clean.grad is not None and float(clean.grad.norm()) > 0
    assert action.grad is None
    assert candidates.grad is None


def test_naive_action_minus_clean_gradient_can_worsen_clean_margin() -> None:
    """A better action has smaller CE magnitude, reversing the naive contrast."""
    theta = torch.tensor(0.0, requires_grad=True)
    clean_margin = theta
    action_margin = theta + 2.0
    clean_loss = torch.nn.functional.softplus(-clean_margin)
    action_loss = torch.nn.functional.softplus(-action_margin)

    clean_gradient = detached_gradients(clean_loss, [theta], retain_graph=True)
    action_gradient = detached_gradients(action_loss, [theta], retain_graph=True)
    metric_gradient = detached_gradients(clean_margin, [theta])
    naive_difference = subtract_gradients(action_gradient, clean_gradient)

    assert float(action_loss) < float(clean_loss)
    assert first_order_metric_gain(naive_difference, metric_gradient) < 0


def test_action_forward_clean_backward_delivers_action_geometry_to_clean_path() -> None:
    clean = torch.tensor([0.0, 1.0], requires_grad=True)
    action = torch.tensor([1.0, 0.0], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    routed = routed_listwise_loss(
        clean, action, candidates, [0, 1, 2],
        route="action_forward_clean_backward", temperature=1.0,
    )
    clean_only = routed_listwise_loss(
        clean, action, candidates, [0, 1, 2],
        route="clean_query_only", temperature=1.0,
    )
    routed_gradient = detached_gradients(routed.loss, [clean], retain_graph=True)
    clean_gradient = detached_gradients(clean_only.loss, [clean])

    # Forward losses differ because one sees action geometry and one sees clean.
    assert float(routed.loss) < float(clean_only.loss)
    # Both gradients improve the same positive-vs-negative direction, while the
    # action-forward confidence changes their magnitude.
    assert math.isclose(
        gradient_cosine(routed_gradient, clean_gradient), 1.0,
        rel_tol=1e-6, abs_tol=1e-6,
    )
    assert math.isfinite(float(routed.margin))


def test_invalid_pointer_is_rejected_before_training() -> None:
    try:
        molecule_max_scores(torch.ones(2), torch.ones(2, 2), [0, 1, 3])
    except ValueError as error:
        assert "span" in str(error)
    else:
        raise AssertionError("invalid molecule pointer was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware action-transfer contracts")
