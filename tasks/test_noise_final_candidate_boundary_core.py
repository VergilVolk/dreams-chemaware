"""Fast numerical tests for candidate-edge direct fine-tuning."""
from __future__ import annotations

import torch

from noise_final_candidate_boundary_core import (
    candidate_boundary_objective, candidate_safety_objective,
)


def test_candidate_edges_are_not_scalar_collapsed() -> None:
    clean = [torch.tensor([[-0.10, 0.02], [-0.05, 0.08]], requires_grad=True)]
    target = [torch.tensor([[0.10, 0.03], [0.04, 0.09]], requires_grad=True)]
    control = [torch.tensor([[-0.02, 0.04], [-0.01, 0.10]], requires_grad=True)]
    result = candidate_boundary_objective(
        clean, target, control, rank_margin=0.05, rank_temperature=0.10,
        advantage_temperature=0.02, hard_temperature=0.10, topk_negatives=2,
        lambda_clean=1.0, lambda_target=1.0, lambda_counterfactual=1.0,
        lambda_full_clean=0.1,
    )
    result.loss.backward()
    assert result.effective_queries == 1
    assert clean[0].grad is not None and target[0].grad is not None
    assert not torch.isclose(clean[0].grad[0, 0], clean[0].grad[0, 1])
    assert control[0].grad is None or float(control[0].grad.abs().sum()) == 0.0


def test_harmful_edges_have_zero_corrective_gradient() -> None:
    clean = [torch.tensor([[0.01, 0.01]], requires_grad=True)]
    target = [torch.tensor([[-0.03, -0.04]], requires_grad=True)]
    control = [torch.tensor([[0.02, 0.03]], requires_grad=True)]
    result = candidate_boundary_objective(
        clean, target, control, rank_margin=0.05, rank_temperature=0.10,
        advantage_temperature=0.02, hard_temperature=0.10, topk_negatives=2,
        lambda_clean=1.0, lambda_target=1.0, lambda_counterfactual=1.0,
        lambda_full_clean=0.0,
    )
    result.loss.backward()
    assert result.effective_queries == 0
    assert target[0].grad is None or float(target[0].grad.abs().sum()) == 0.0
    assert control[0].grad is None or float(control[0].grad.abs().sum()) == 0.0


def test_counterfactual_logistic_has_no_point01_dead_zone() -> None:
    clean = [torch.tensor([[-0.02]], requires_grad=True)]
    target = [torch.tensor([[0.04]], requires_grad=True)]
    control = [torch.tensor([[0.00]], requires_grad=True)]
    result = candidate_boundary_objective(
        clean, target, control, rank_margin=0.05, rank_temperature=0.10,
        advantage_temperature=0.02, hard_temperature=0.10, topk_negatives=1,
        lambda_clean=0.0, lambda_target=0.0, lambda_counterfactual=1.0,
        lambda_full_clean=0.0,
    )
    result.loss.backward()
    assert float(target[0].grad.abs().sum()) > 0
    assert control[0].grad is None or float(control[0].grad.abs().sum()) == 0.0


def test_candidate_safety_catches_one_switch() -> None:
    current = [torch.tensor([[0.20, -0.01]], requires_grad=True)]
    initial = [torch.tensor([[0.18, 0.04]])]
    loss, metrics = candidate_safety_objective(
        current, initial, slack=0.005, topk_negatives=2,
    )
    loss.backward()
    assert metrics["candidate_safety_active_fraction"] == 0.5
    assert current[0].grad is not None
    assert float(current[0].grad[0, 0]) == 0.0
    assert float(current[0].grad[0, 1]) < 0.0


def main() -> None:
    tests = [
        test_candidate_edges_are_not_scalar_collapsed,
        test_harmful_edges_have_zero_corrective_gradient,
        test_counterfactual_logistic_has_no_point01_dead_zone,
        test_candidate_safety_catches_one_switch,
    ]
    for test in tests:
        test()
    print(f"[test_noise_final_candidate_boundary_core] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
