"""Independent scientific tests for E4 live-shared Hybrid V3."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_e4_live_shared_hybrid_v3 import (  # noqa: E402
    SeparatedE4ActionInjectorV3,
    build_query_equal_action_plan_v3,
    live_shared_e4_action_objective_v3,
)


def _preservation_anchor(
    encoded: torch.Tensor, layout: dict[str, object],
) -> torch.Tensor:
    """Mirror E4 exactly: preserve clean and references, never the action."""
    indices = [
        int(layout["clean"]),
        *map(int, layout["positive"]),
        *map(int, layout["negative"]),
    ]
    return encoded[indices].detach().clone()


def _objective(
    encoded: torch.Tensor,
    layouts: list[dict[str, object]],
    anchors: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    return live_shared_e4_action_objective_v3(
        encoded,
        layouts,
        anchors,
        [0.0] * len(layouts),
        rank_margin=0.05,
        temperature=0.10,
        margin_floor_slack=0.005,
        lambda_clean_rank=1.0,
        lambda_aug_rank=1.0,
        lambda_consistency=0.25,
        # This test isolates the shared ranking/consistency graph. A fixed
        # preservation anchor must not be able to mask a detached reference.
        lambda_margin_floor=0.0,
        lambda_preserve=0.0,
    )


def test_query_equal_schedule_never_mixes_same_query_actions() -> None:
    # Sixteen is the observed formal-panel maximum, and is exactly the four
    # views/query/epoch x four epochs coverage boundary.
    queries = [10] * 16 + [11] * 2 + [12] * 7 + [13] * 1
    action_ids = [f"a{index}" for index in range(len(queries))]
    families = [f"f{index % 3}" for index in range(len(queries))]
    plan = build_query_equal_action_plan_v3(
        action_ids,
        queries,
        families,
        epochs=4,
        views_per_query_per_epoch=4,
        actions_per_step=3,
        optimizer_steps_per_epoch=20,
        seed=71,
    )
    seen: set[int] = set()
    for epoch in plan.epoch_steps:
        counts = {query: 0 for query in set(queries)}
        for batch in epoch:
            batch_queries = [queries[index] for index in batch]
            if len(batch_queries) != len(set(batch_queries)):
                raise AssertionError("same-query actions were averaged in one step")
            for query in batch_queries:
                counts[query] += 1
            seen.update(batch)
        if set(counts.values()) != {4}:
            raise AssertionError(f"query dose is unequal: {counts}")
    if seen != set(range(len(action_ids))):
        raise AssertionError("coverage-first query cycles lost an action in the toy panel")
    report = plan.audit_manifest()
    if report["same_query_actions_never_share_an_optimizer_step"] is not True:
        raise AssertionError("schedule manifest contradicts the realized plan")
    if report["all_unique_actions_exposed"] is not True:
        raise AssertionError("the maximum-size query did not expose all actions")


def test_v3_objective_equals_complete_historical_e4_formula_and_gradient() -> None:
    torch.manual_seed(73)
    raw = torch.randn(10, 5, dtype=torch.float64)
    encoded = torch.nn.functional.normalize(raw, dim=1).detach().requires_grad_(True)
    manual_encoded = encoded.detach().clone().requires_grad_(True)
    layouts = [
        {"clean": 0, "action": 1, "positive": [2, 3], "negative": [4]},
        {"clean": 5, "action": 6, "positive": [7], "negative": [8, 9]},
    ]
    anchors = [
        torch.nn.functional.normalize(torch.randn(4, 5, dtype=torch.float64), dim=1),
        torch.nn.functional.normalize(torch.randn(4, 5, dtype=torch.float64), dim=1),
    ]
    inherited = [0.2, -0.1]
    observed, _ = live_shared_e4_action_objective_v3(
        encoded, layouts, anchors, inherited,
        rank_margin=0.05, temperature=0.10, margin_floor_slack=0.005,
        lambda_clean_rank=1.0, lambda_aug_rank=1.0,
        lambda_consistency=0.25, lambda_margin_floor=2.0,
        lambda_preserve=5.0,
    )

    clean_rank = []
    action_rank = []
    consistency = []
    floor = []
    preserve = []
    for item, anchor, inherited_margin in zip(layouts, anchors, inherited):
        clean = manual_encoded[int(item["clean"])]
        action = manual_encoded[int(item["action"])]
        positive = manual_encoded[list(map(int, item["positive"]))]
        negative = manual_encoded[list(map(int, item["negative"]))]
        clean_margin = torch.max(positive @ clean) - torch.max(negative @ clean)
        action_margin = torch.max(positive @ action) - torch.max(negative @ action)
        clean_rank.append(torch.nn.functional.softplus((0.05 - clean_margin) / 0.10))
        action_rank.append(torch.nn.functional.softplus((0.05 - action_margin) / 0.10))
        consistency.append(1.0 - torch.sum(clean * action))
        floor.append(torch.nn.functional.relu(inherited_margin - 0.005 - clean_margin))
        current = torch.cat((clean.unsqueeze(0), positive, negative), dim=0)
        preserve.append((1.0 - torch.sum(current * anchor, dim=1)).mean())
    expected = (
        torch.stack(clean_rank).mean()
        + torch.stack(action_rank).mean()
        + 0.25 * torch.stack(consistency).mean()
        + 2.0 * torch.stack(floor).mean()
        + 5.0 * torch.stack(preserve).mean()
    )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1e-12)
    observed_gradient = torch.autograd.grad(observed, encoded)[0]
    expected_gradient = torch.autograd.grad(expected, manual_encoded)[0]
    torch.testing.assert_close(
        observed_gradient, expected_gradient, rtol=0.0, atol=1e-12,
    )


def test_full_live_shared_e4_loss_reaches_all_four_roles() -> None:
    # The positive and negative rows below are the active max references, so
    # their non-zero gradients directly test the production max-margin path.
    encoded = torch.tensor([
        [0.0, 1.0],    # clean
        [0.8, 0.6],    # action
        [1.0, 0.0],    # positive
        [-0.8, 0.6],   # negative
    ], dtype=torch.float64, requires_grad=True)
    layout = {"clean": 0, "action": 1, "positive": [2], "negative": [3]}
    anchors = [_preservation_anchor(encoded, layout)]
    loss, report = _objective(
        encoded,
        [layout],
        anchors,
    )
    loss.backward()
    role_norms = {
        "clean": encoded.grad[0].norm().item(),
        "action": encoded.grad[1].norm().item(),
        "positive": encoded.grad[2].norm().item(),
        "negative": encoded.grad[3].norm().item(),
    }
    if any(value <= 0 for value in role_norms.values()):
        raise AssertionError(f"one live shared E4 role lost gradient: {role_norms}")
    if report["action_reference_gradient_detached"] is not False:
        raise AssertionError("V3 silently detached positive/negative references")
    if report["action_aug_rank_active_fraction"] != 1.0:
        raise AssertionError("V3 gated the action rank")
    if report["complete_historical_e4_terms_present"] is not True:
        raise AssertionError("V3 omitted an E4 loss term")


def test_action_rank_remains_live_after_margin_is_satisfied() -> None:
    encoded = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ], dtype=torch.float64, requires_grad=True)
    layout = {"clean": 0, "action": 1, "positive": [2], "negative": [3]}
    loss, _ = live_shared_e4_action_objective_v3(
        encoded,
        [layout],
        [_preservation_anchor(encoded, layout)],
        [0.0],
        rank_margin=0.05,
        temperature=0.10,
        margin_floor_slack=0.005,
        lambda_clean_rank=0.0,
        lambda_aug_rank=1.0,
        lambda_consistency=0.0,
        lambda_margin_floor=0.0,
        lambda_preserve=0.0,
    )
    loss.backward()
    if encoded.grad[1].norm().item() <= 0:
        raise AssertionError("satisfied action anchor was hard-gated")
    if encoded.grad[2].norm().item() <= 0 or encoded.grad[3].norm().item() <= 0:
        raise AssertionError("satisfied action lost its live triplet references")


def test_targeted_and_shuffled_differ_before_optimizer_injection() -> None:
    torch.manual_seed(79)
    model = torch.nn.Linear(3, 2, bias=False).double()
    clean = torch.tensor([[0.0, 1.0, 0.2]], dtype=torch.float64)
    positive = torch.tensor([[1.0, 0.0, 0.1]], dtype=torch.float64)
    negative = torch.tensor([[-1.0, 0.0, 0.1]], dtype=torch.float64)

    def gradient(action: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        spectra = torch.cat((clean, action, positive, negative), dim=0)
        encoded = torch.nn.functional.normalize(model(spectra), dim=1)
        layout = {"clean": 0, "action": 1, "positive": [2], "negative": [3]}
        loss, _ = live_shared_e4_action_objective_v3(
            encoded,
            [layout],
            [_preservation_anchor(encoded, layout)],
            [0.0],
            rank_margin=0.05,
            temperature=0.10,
            margin_floor_slack=0.005,
            lambda_clean_rank=1.0,
            lambda_aug_rank=1.0,
            lambda_consistency=0.25,
            lambda_margin_floor=0.0,
            lambda_preserve=0.0,
        )
        loss.backward()
        return torch.cat([parameter.grad.flatten() for parameter in model.parameters()])

    targeted = gradient(torch.tensor([[1.0, 0.0, 0.2]], dtype=torch.float64))
    shuffled = gradient(torch.tensor([[-0.4, 0.7, 0.2]], dtype=torch.float64))
    if targeted.norm().item() <= 0 or shuffled.norm().item() <= 0:
        raise AssertionError("targeted/shuffled pre-injection gradient is zero")
    if torch.equal(targeted, shuffled):
        raise AssertionError("action content vanished before the injector")
    cosine = torch.nn.functional.cosine_similarity(
        targeted.unsqueeze(0), shuffled.unsqueeze(0), dim=1,
    ).item()
    if cosine >= 1.0 - 1e-10:
        raise AssertionError(f"targeted/shuffled directions are indistinguishable: {cosine}")


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    parameters = list(model.parameters())
    return torch.optim.AdamW([
        {"params": parameters[:1], "lr": 1e-3, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": parameters[1:], "lr": 2e-4, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])


def test_zero_action_is_bitwise_ordinary_e4_for_64_steps() -> None:
    torch.manual_seed(83)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    oracle = copy.deepcopy(model)
    optimizer = _optimizer(model)
    oracle_optimizer = _optimizer(oracle)
    bridge = SeparatedE4ActionInjectorV3(optimizer, list(model.parameters()))
    for step in range(64):
        x = torch.tensor([
            [1.0 + 0.01 * step, -0.5, 0.25],
            [-0.2, 0.8 - 0.005 * step, 0.4],
        ], dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        (model(x).sum() * 0.0).backward()
        bridge.capture_action_gradient_(allow_zero=True)
        optimizer.zero_grad(set_to_none=True)
        (model(x).square().mean() * (1 + step % 3)).backward()

        oracle_optimizer.zero_grad(set_to_none=True)
        (oracle(x).square().mean() * (1 + step % 3)).backward()
        torch.nn.utils.clip_grad_norm_(list(oracle.parameters()), 1.0)
        oracle_optimizer.step()
        oracle_optimizer.zero_grad(set_to_none=True)

        receipt = bridge.step_(maximum_gradient_norm=1.0)
        if receipt.zero_action_exact_e4_equivalence is not True:
            raise AssertionError("zero-action receipt is not exact E4")
        for observed, expected in zip(model.parameters(), oracle.parameters()):
            if not torch.equal(observed, expected):
                raise AssertionError(f"zero action changed E4 at step {step}")
        for observed, expected in zip(model.parameters(), oracle.parameters()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                if not torch.equal(
                    optimizer.state[observed][key], oracle_optimizer.state[expected][key]
                ):
                    raise AssertionError(f"E4 optimizer state changed at step {step}: {key}")


def test_action_state_is_independent_and_reaches_exact_fraction() -> None:
    torch.manual_seed(89)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    oracle = copy.deepcopy(model)
    optimizer = _optimizer(model)
    oracle_optimizer = _optimizer(oracle)
    parameters = list(model.parameters())
    bridge = SeparatedE4ActionInjectorV3(
        optimizer,
        parameters,
        target_action_fraction=0.25,
        minimum_e4_projection_retention=0.90,
        maximum_update_norm_ratio=1.50,
    )
    x = torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.8, 0.4]], dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    (1000.0 * model(x)[0].square().sum()).backward()
    bridge.capture_action_gradient_()
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    base_gradients = [parameter.grad.detach().clone() for parameter in parameters]

    oracle_optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(oracle.parameters(), base_gradients):
        parameter.grad = gradient.clone()
    torch.nn.utils.clip_grad_norm_(list(oracle.parameters()), 1.0)
    oracle_optimizer.step()
    oracle_optimizer.zero_grad(set_to_none=True)

    receipt = bridge.step_(maximum_gradient_norm=1.0)
    if receipt.action_active is not True:
        raise AssertionError("non-zero live-shared action was not processed")
    if any(
        abs(value - 0.25) > 1e-6
        for value in receipt.final_action_fraction_by_group.values()
    ):
        raise AssertionError("action stream did not reach the exact 0.25 boundary")
    if any(value + 1e-6 < 0.90 for value in receipt.e4_projection_retention_by_group.values()):
        raise AssertionError("action update consumed the E4 protective direction")
    if any(value > 1.50 + 1e-6 for value in receipt.final_to_e4_update_norm_ratio_by_group.values()):
        raise AssertionError("exact composition exceeded the E4 norm ceiling")
    # The live AdamW state must equal the independently executed E4-only state;
    # only parameter materialization may include the separately owned action.
    for observed, expected in zip(model.parameters(), oracle.parameters()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[observed][key],
                oracle_optimizer.state[expected][key],
                rtol=0.0,
                atol=0.0,
            )
    for live, shadow in zip(parameters, bridge.action_parameters):
        if live.data_ptr() == shadow.data_ptr():
            raise AssertionError("E4 and action optimizer states share parameter storage")
        live_state = optimizer.state[live]
        action_state = bridge.action_optimizer.state[shadow]
        if live_state["exp_avg"].data_ptr() == action_state["exp_avg"].data_ptr():
            raise AssertionError("E4 and action first moments share storage")
    if any(float(group["weight_decay"]) != 0.0
           for group in bridge.action_optimizer.param_groups):
        raise AssertionError("action optimizer contains non-action weight decay")


def test_weak_natural_action_update_is_filled_to_exact_fraction() -> None:
    """Regression for V3's former max-25%/never-amplify signal-loss bug."""
    torch.manual_seed(93)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    optimizer = _optimizer(model)
    bridge = SeparatedE4ActionInjectorV3(
        optimizer,
        list(model.parameters()),
        target_action_fraction=0.25,
        minimum_e4_projection_retention=0.90,
        maximum_update_norm_ratio=1.50,
    )
    # Force the independently materialised natural action displacement far
    # below the E4 displacement.  A ceiling-only implementation would silently
    # retain this tiny fraction and still call it non-zero.
    for group in bridge.action_optimizer.param_groups:
        group["lr"] = float(group["lr"]) * 1e-4
    x = torch.tensor([[0.8, -0.3, 0.2], [-0.1, 0.7, 0.5]], dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    model(x)[0].square().sum().backward()
    bridge.capture_action_gradient_()
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    receipt = bridge.step_(maximum_gradient_norm=1.0)
    if any(value >= 0.01 for value in receipt.natural_action_fraction_by_group.values()):
        raise AssertionError("regression fixture did not create a weak natural action")
    if any(
        abs(value - 0.25) > 1e-6
        for value in receipt.final_action_fraction_by_group.values()
    ):
        raise AssertionError("weak action signal was not filled to exact 0.25")
    if not any(value > 1.0 for value in receipt.retained_action_scale_by_group.values()):
        raise AssertionError("weak action signal was not amplified")


def test_zero_action_after_action_history_is_ordinary_e4() -> None:
    """Catch the V2 failure where an earlier action poisoned later zero steps."""
    torch.manual_seed(97)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    optimizer = _optimizer(model)
    bridge = SeparatedE4ActionInjectorV3(optimizer, list(model.parameters()))
    x = torch.tensor([[0.7, -0.2, 0.1], [-0.1, 0.9, 0.3]], dtype=torch.float64)

    for cycle in range(8):
        optimizer.zero_grad(set_to_none=True)
        ((cycle + 1.0) * model(x)[0].square().sum()).backward()
        bridge.capture_action_gradient_()
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        bridge.step_(maximum_gradient_norm=1.0)

        # Build the exact ordinary-E4 counterfactual from the current combined
        # parameters and the live E4-only optimizer state.
        oracle = copy.deepcopy(model)
        oracle_optimizer = _optimizer(oracle)
        oracle_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        optimizer.zero_grad(set_to_none=True)
        (model(x).sum() * 0.0).backward()
        bridge.capture_action_gradient_(allow_zero=True)
        optimizer.zero_grad(set_to_none=True)
        ((cycle + 2.0) * model(x).square().mean()).backward()

        oracle_optimizer.zero_grad(set_to_none=True)
        ((cycle + 2.0) * oracle(x).square().mean()).backward()
        torch.nn.utils.clip_grad_norm_(list(oracle.parameters()), 1.0)
        oracle_optimizer.step()
        oracle_optimizer.zero_grad(set_to_none=True)

        receipt = bridge.step_(maximum_gradient_norm=1.0)
        if receipt.zero_action_exact_e4_equivalence is not True:
            raise AssertionError("post-action zero step was not declared exact E4")
        for observed, expected in zip(model.parameters(), oracle.parameters()):
            if not torch.equal(observed, expected):
                raise AssertionError(
                    f"action history altered a later zero-action E4 step at cycle {cycle}"
                )


def main() -> None:
    tests = (
        test_query_equal_schedule_never_mixes_same_query_actions,
        test_v3_objective_equals_complete_historical_e4_formula_and_gradient,
        test_full_live_shared_e4_loss_reaches_all_four_roles,
        test_action_rank_remains_live_after_margin_is_satisfied,
        test_targeted_and_shuffled_differ_before_optimizer_injection,
        test_zero_action_is_bitwise_ordinary_e4_for_64_steps,
        test_action_state_is_independent_and_reaches_exact_fraction,
        test_weak_natural_action_update_is_filled_to_exact_fraction,
        test_zero_action_after_action_history_is_ordinary_e4,
    )
    for test in tests:
        test()
    print(f"[test_noise_e4_live_shared_hybrid_v3] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
