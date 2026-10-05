"""Numerical invariants for evaluator-aligned direct-boundary training."""
from __future__ import annotations

import torch

from noise_final_direct_boundary_v2_core import (
    calibrate_action_scale,
    direct_boundary_objective,
    direct_action_to_clean_invariance,
    direct_target_boundary_objective,
    molecule_max_margin,
    molecule_max_margin_from_scores,
)


def objective(clean, actions, controls, weights, **overrides):
    options = {
        "rank_margin": 0.05,
        "rank_temperature": 0.10,
        "advantage_temperature": 0.02,
        "topk_negatives": 2,
        "action_safety_slack": 0.01,
        "lambda_clean": 1.0,
        "lambda_corrective_clean": 1.0,
        "lambda_action_rank": 0.0,
        "lambda_counterfactual": 0.0,
        "lambda_action_safety": 0.0,
        "lambda_margin_transfer": 0.0,
    }
    options.update(overrides)
    return direct_boundary_objective(clean, actions, controls, weights, **options)


def test_molecule_max_matches_retrieval_and_ignores_weak_positive_references() -> None:
    positive = torch.tensor([0.90, -0.70, 0.10], requires_grad=True)
    negatives = [torch.tensor([0.20, 0.80]), torch.tensor([0.30, 0.40, 0.35])]
    margin = molecule_max_margin_from_scores(positive, negatives)
    assert torch.allclose(margin, torch.tensor([0.10, 0.50]))
    margin.sum().backward()
    assert torch.allclose(positive.grad, torch.tensor([2.0, 0.0, 0.0]))

    query = torch.tensor([1.0, 0.0])
    positive_embeddings = torch.tensor([[0.9, 0.1], [-0.7, 0.0]])
    negative_embeddings = [torch.tensor([[0.8, 0.2], [0.2, 0.8]])]
    assert torch.allclose(
        molecule_max_margin(query, positive_embeddings, negative_embeddings),
        torch.tensor([0.1]),
    )


def test_qualified_action_adds_clean_boundary_pressure() -> None:
    clean_without = torch.tensor([-0.05, 0.02], requires_grad=True)
    no_action = objective([clean_without], [[]], [[]], [torch.empty(0)])
    no_action.loss.backward()
    base_gradient = clean_without.grad.detach().clone()

    clean_with = torch.tensor([-0.05, 0.02], requires_grad=True)
    action = torch.tensor([0.10, 0.08], requires_grad=True)
    control = torch.tensor([0.00, 0.00], requires_grad=True)
    with_action = objective([clean_with], [[action]], [[control]], [torch.tensor([1.0])])
    with_action.loss.backward()
    assert with_action.effective_queries == 1
    assert float(clean_with.grad.abs().sum()) > float(base_gradient.abs().sum())


def test_easier_action_does_not_reduce_clean_gradient() -> None:
    def clean_gradient(action_values):
        clean = torch.tensor([-0.05, 0.02], requires_grad=True)
        action = torch.tensor(action_values, requires_grad=True)
        control = torch.tensor([0.00, 0.00], requires_grad=True)
        result = objective([clean], [[action]], [[control]], [torch.tensor([1.0])])
        result.loss.backward()
        return clean.grad.detach().clone()

    moderate = clean_gradient([0.04, 0.04])
    easy = clean_gradient([0.40, 0.40])
    assert torch.allclose(moderate, easy, atol=1e-6, rtol=1e-6)


def test_action_duplication_does_not_multiply_query_dose() -> None:
    def losses_and_gradient(copies: int):
        clean = torch.tensor([-0.04, 0.01], requires_grad=True)
        actions = [torch.tensor([0.08, 0.06], requires_grad=True) for _ in range(copies)]
        controls = [torch.tensor([0.00, 0.00], requires_grad=True) for _ in range(copies)]
        result = objective(
            [clean], [actions], [controls], [torch.ones(copies)],
            lambda_clean=0.0, lambda_action_rank=0.5,
            lambda_counterfactual=0.2, lambda_action_safety=0.3,
        )
        result.loss.backward()
        return result, clean.grad.detach().clone()

    one, one_gradient = losses_and_gradient(1)
    ten, ten_gradient = losses_and_gradient(10)
    assert torch.allclose(one.loss, ten.loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(one_gradient, ten_gradient, atol=1e-7, rtol=1e-6)


def test_family_equal_weight_prevents_large_recipe_family_dilution() -> None:
    def value(copies: int):
        clean = torch.tensor([-0.04, 0.01], requires_grad=True)
        family_a = [torch.tensor([0.08, 0.06], requires_grad=True) for _ in range(copies)]
        family_b = [torch.tensor([0.02, 0.03], requires_grad=True)]
        actions = family_a + family_b
        controls = [torch.zeros(2, requires_grad=True) for _ in actions]
        groups = [["large"] * copies + ["small"]]
        result = objective(
            [clean], [actions], [controls], [torch.ones(len(actions))],
            action_group=groups, lambda_clean=0.0,
        )
        result.loss.backward()
        return result.loss.detach(), clean.grad.detach().clone()
    one = value(1)
    ten = value(10)
    assert torch.allclose(one[0], ten[0], atol=1e-7, rtol=1e-6)
    assert torch.allclose(one[1], ten[1], atol=1e-7, rtol=1e-6)


def test_margin_delta_transfer_updates_clean_not_control() -> None:
    clean = torch.tensor([-0.05, 0.00], requires_grad=True)
    action = torch.tensor([0.10, 0.08], requires_grad=True)
    control = torch.tensor([0.00, 0.01], requires_grad=True)
    result = objective(
        [clean], [[action]], [[control]], [torch.tensor([1.0])],
        lambda_clean=0.0, lambda_corrective_clean=0.0,
        lambda_margin_transfer=1.0, margin_transfer_fraction=0.5,
    )
    result.loss.backward()
    assert float(result.margin_transfer) > 0
    assert clean.grad is not None and bool(torch.all(clean.grad < 0))
    assert action.grad is None or float(action.grad.abs().sum()) == 0.0
    assert control.grad is None or float(control.grad.abs().sum()) == 0.0


def test_unqualified_action_is_safety_only() -> None:
    clean = torch.tensor([0.10, 0.08], requires_grad=True)
    harmful = torch.tensor([-0.10, -0.02], requires_grad=True)
    control = torch.tensor([0.05, 0.04], requires_grad=True)
    result = objective(
        [clean], [[harmful]], [[control]], [torch.tensor([0.0])],
        lambda_clean=0.0, lambda_corrective_clean=1.0,
        lambda_action_rank=1.0, lambda_counterfactual=1.0,
        lambda_action_safety=1.0,
    )
    result.loss.backward()
    assert result.effective_queries == 0 and result.unqualified_actions == 1
    assert float(result.corrective_clean_rank) == 0.0
    assert float(result.action_rank) == 0.0 and float(result.counterfactual) == 0.0
    assert float(result.action_safety) > 0.0
    assert clean.grad is None or float(clean.grad.abs().sum()) == 0.0
    assert harmful.grad is not None and float(harmful.grad.abs().sum()) > 0.0
    assert control.grad is None or float(control.grad.abs().sum()) == 0.0


def test_control_branch_is_detached() -> None:
    clean = torch.tensor([-0.05], requires_grad=True)
    action = torch.tensor([0.04], requires_grad=True)
    control = torch.tensor([0.00], requires_grad=True)
    result = objective(
        [clean], [[action]], [[control]], [torch.tensor([1.0])],
        lambda_clean=0.0, lambda_corrective_clean=0.0,
        lambda_action_rank=0.0, lambda_counterfactual=1.0,
    )
    result.loss.backward()
    assert action.grad is not None and float(action.grad.abs().sum()) > 0.0
    assert control.grad is None or float(control.grad.abs().sum()) == 0.0


def test_calibration_can_downscale_an_oversized_action_branch() -> None:
    scale = calibrate_action_scale(
        action_gradient_norm=10.0,
        safety_gradient_norm=2.0,
        target_action_to_safety_ratio=1.0,
        scale_cap=16.0,
    )
    assert scale == 0.2


def test_inactive_safety_gradient_does_not_disable_actions() -> None:
    scale = calibrate_action_scale(
        action_gradient_norm=10.0,
        safety_gradient_norm=0.0,
        target_action_to_safety_ratio=1.0,
    )
    assert scale == 1.0


def test_target_only_action_multiplicity_is_query_equal() -> None:
    def value(copies: int):
        clean = torch.tensor([-0.05, 0.02], requires_grad=True)
        action = [torch.tensor([0.08, 0.06], requires_grad=True) for _ in range(copies)]
        result = direct_target_boundary_objective(
            [clean], [action], rank_margin=0.05, rank_temperature=0.10,
            topk_negatives=2, action_safety_slack=0.01,
            lambda_clean=0.1, lambda_action_conditioned_clean=1.0,
            lambda_action_rank=1.0, lambda_action_safety=0.25,
        )
        result.loss.backward()
        return result.loss.detach(), clean.grad.detach().clone()
    one_loss, one_grad = value(1)
    ten_loss, ten_grad = value(10)
    assert torch.allclose(one_loss, ten_loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(one_grad, ten_grad, atol=1e-7, rtol=1e-6)


def test_action_to_clean_invariance_has_explicit_one_way_gradient() -> None:
    clean = torch.tensor([1.0, 0.0], requires_grad=True)
    action = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = direct_action_to_clean_invariance([clean], [[action]])
    loss.backward()
    assert clean.grad is not None and float(clean.grad.abs().sum()) > 0
    assert action.grad is None


def test_action_to_clean_invariance_is_query_equal() -> None:
    clean_one = torch.tensor([0.8, 0.2], requires_grad=True)
    action_one = torch.tensor([0.2, 0.8], requires_grad=True)
    one = direct_action_to_clean_invariance([clean_one], [[action_one]])
    one.backward()
    clean_many = torch.tensor([0.8, 0.2], requires_grad=True)
    actions_many = [torch.tensor([0.2, 0.8], requires_grad=True) for _ in range(10)]
    many = direct_action_to_clean_invariance([clean_many], [actions_many])
    many.backward()
    assert torch.allclose(one, many)
    assert torch.allclose(clean_one.grad, clean_many.grad)


def test_target_and_transfer_are_family_equal() -> None:
    def target_value(copies: int):
        clean = torch.tensor([-0.05, 0.02], requires_grad=True)
        family_a = [torch.tensor([0.08, 0.06], requires_grad=True) for _ in range(copies)]
        family_b = [torch.tensor([-0.02, 0.03], requires_grad=True)]
        actions = family_a + family_b
        groups = ["N"] * copies + ["P"]
        result = direct_target_boundary_objective(
            [clean], [actions], rank_margin=0.05, rank_temperature=0.10,
            topk_negatives=2, action_safety_slack=0.01,
            lambda_clean=0.0, lambda_action_conditioned_clean=0.0,
            lambda_action_rank=1.0, lambda_action_safety=0.25,
            action_group=[groups],
        )
        transfer_clean = torch.tensor([0.8, 0.2], requires_grad=True)
        vectors = [torch.tensor([0.2, 0.8], requires_grad=True) for _ in range(copies)]
        vectors.append(torch.tensor([0.6, 0.4], requires_grad=True))
        transfer = direct_action_to_clean_invariance(
            [transfer_clean], [vectors], action_group=[groups],
        )
        return result.loss.detach(), transfer.detach()
    one = target_value(1)
    ten = target_value(10)
    assert torch.allclose(one[0], ten[0], atol=1e-7, rtol=1e-6)
    assert torch.allclose(one[1], ten[1], atol=1e-7, rtol=1e-6)


def main() -> None:
    tests = [
        test_molecule_max_matches_retrieval_and_ignores_weak_positive_references,
        test_qualified_action_adds_clean_boundary_pressure,
        test_easier_action_does_not_reduce_clean_gradient,
        test_action_duplication_does_not_multiply_query_dose,
        test_family_equal_weight_prevents_large_recipe_family_dilution,
        test_margin_delta_transfer_updates_clean_not_control,
        test_unqualified_action_is_safety_only,
        test_control_branch_is_detached,
        test_calibration_can_downscale_an_oversized_action_branch,
        test_inactive_safety_gradient_does_not_disable_actions,
        test_target_only_action_multiplicity_is_query_equal,
        test_action_to_clean_invariance_has_explicit_one_way_gradient,
        test_action_to_clean_invariance_is_query_equal,
        test_target_and_transfer_are_family_equal,
    ]
    for test in tests:
        test()
    print(f"[test_noise_final_direct_boundary_v2_core] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
