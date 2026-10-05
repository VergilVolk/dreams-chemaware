"""Numerical and scheduling invariants for direct-v3 action injection."""
from __future__ import annotations

from collections import Counter

import torch

from noise_corrected_direct_v3_core import (
    balanced_action_step_plan,
    cap_safe_corrective_repartition,
    corrective_recycle_scale,
    corrective_action_objective,
    harmful_boundary_objective,
    robust_action_objective,
    symmetric_live_action_consistency,
    v3_schedule_geometry,
)


def test_balanced_plan_has_no_protect_only_tail_or_coverage_loss() -> None:
    corrective = [f"c{index}" for index in range(7)]
    protective = [f"p{index}" for index in range(23)]
    plan = balanced_action_step_plan(corrective, protective, seed=17)
    assert len(plan) == len(corrective)
    assert sorted(step.corrective for step in plan) == corrective
    flattened = [item for step in plan for item in step.protective]
    assert len(flattened) == len(protective) and set(flattened) == set(protective)
    counts = [len(step.protective) for step in plan]
    assert max(counts) - min(counts) <= 1
    assert all(step.corrective.startswith("c") for step in plan)
    assert all(step.corrective_scale == 1.0 for step in plan)


def test_balanced_plan_is_seed_deterministic() -> None:
    one = balanced_action_step_plan(list(range(5)), list(range(19)), seed=9)
    two = balanced_action_step_plan(list(range(5)), list(range(19)), seed=9)
    three = balanced_action_step_plan(list(range(5)), list(range(19)), seed=10)
    assert one == two
    assert one != three


def test_balanced_plan_can_repeat_actions_without_increasing_epoch_dose() -> None:
    corrective = [f"c{index}" for index in range(3)]
    protective = [f"p{index}" for index in range(100)]
    plan = balanced_action_step_plan(
        corrective,
        protective,
        seed=4,
        maximum_protective_microbatches_per_step=4,
    )
    assert len(plan) == 25
    assert max(len(step.protective) for step in plan) <= 4
    flattened = [item for step in plan for item in step.protective]
    assert len(flattened) == len(protective) and set(flattened) == set(protective)
    for batch in corrective:
        assert abs(sum(
            step.corrective_scale for step in plan if step.corrective == batch
        ) - 1.0) < 1e-12


def test_full_dose_recycling_keeps_every_optimizer_step_action_strength() -> None:
    plan = balanced_action_step_plan(
        ["c0", "c1"],
        list(range(17)),
        seed=5,
        maximum_protective_microbatches_per_step=2,
    )
    assert len(plan) == 9
    recycle = len(plan) / 2
    for batch in ("c0", "c1"):
        normalized = sum(
            corrective_recycle_scale(step, full_dose=False)
            for step in plan if step.corrective == batch
        )
        full = sum(
            corrective_recycle_scale(
                step, full_dose=True, full_dose_epoch_multiplier=recycle,
            )
            for step in plan if step.corrective == batch
        )
        assert abs(normalized - 1.0) < 1e-12
        assert abs(full - recycle) < 1e-12
    assert abs(sum(
        corrective_recycle_scale(
            step, full_dose=True, full_dose_epoch_multiplier=recycle,
        )
        for step in plan
    ) - len(plan)) < 1e-12


def test_balanced_plan_respects_auxiliary_minimum_steps() -> None:
    plan = balanced_action_step_plan(
        list(range(5)), list(range(3)), seed=9,
        maximum_protective_microbatches_per_step=4,
        minimum_steps=11,
    )
    assert len(plan) == 11
    assert {step.corrective for step in plan} == set(range(5))
    assert sum(len(step.protective) for step in plan) == 3


def test_v6_schedule_geometry_is_exactly_cap_safe_before_training() -> None:
    geometry = v3_schedule_geometry(
        corrective_queries=3482,
        robust_queries=39165,
        harmful_queries=16955,
        protective_queries=65286,
        corrective_batch_size=4,
        protective_batch_size=8,
        maximum_auxiliary_microbatches_per_step=4,
        maximum_protective_microbatches_per_step=4,
        maximum_corrective_recycle_factor=4.0,
    )
    assert geometry.original_corrective_batches == 871
    assert geometry.robust_batches == 9792
    assert geometry.harmful_batches == 4239
    assert geometry.auxiliary_batches == 14031
    assert geometry.protective_batches == 8161
    assert geometry.auxiliary_required_optimizer_steps == 3508
    assert geometry.protective_required_optimizer_steps == 2041
    assert geometry.required_optimizer_steps == 3508
    assert geometry.original_corrective_recycle_factor == 3508 / 871
    assert geometry.original_corrective_recycle_factor > 4.0
    assert geometry.cap_safe_corrective_batches == 877
    assert geometry.corrective_batches_added_by_cap_safe_repartition == 6
    assert geometry.effective_corrective_recycle_factor == 4.0
    assert geometry.minimum_cap_safe_corrective_batch_size == 3
    assert geometry.maximum_cap_safe_corrective_batch_size == 4
    assert geometry.minimum_size_corrective_batches == 26
    assert geometry.maximum_size_corrective_batches == 851


def test_v6_cap_safe_repartition_is_even_order_preserving_and_four_dose() -> None:
    natural = [list(range(left, min(left + 4, 3482))) for left in range(0, 3482, 4)]
    repaired = cap_safe_corrective_repartition(natural, 877)
    assert [value for batch in repaired for value in batch] == list(range(3482))
    assert Counter(map(len, repaired)) == {3: 26, 4: 851}
    smaller_positions = [index for index, batch in enumerate(repaired) if len(batch) == 3]
    assert smaller_positions[0] < 877 // 10
    assert smaller_positions[-1] > 9 * 877 // 10
    assert max(
        right - left for left, right in zip(
            smaller_positions, smaller_positions[1:],
        )
    ) - min(
        right - left for left, right in zip(
            smaller_positions, smaller_positions[1:],
        )
    ) <= 1

    plan = balanced_action_step_plan(
        repaired,
        list(range(8161)),
        seed=20260908,
        maximum_protective_microbatches_per_step=4,
        minimum_steps=3508,
    )
    assert len(plan) == 3508
    assert max(len(step.protective) for step in plan) <= 4
    assert sorted(value for step in plan for value in step.protective) == list(range(8161))
    batch_exposures = Counter(tuple(step.corrective) for step in plan)
    assert set(batch_exposures.values()) == {4}
    assert all(
        corrective_recycle_scale(
            step, full_dose=True, full_dose_epoch_multiplier=4.0,
        ) == 1.0
        for step in plan
    )


def test_cap_safe_geometry_has_no_tolerance_leak_and_fails_if_queries_cannot_split() -> None:
    natural = [[0, 1, 2, 3], [4, 5, 6]]
    assert cap_safe_corrective_repartition(natural, 2) == natural

    def geometry(required_steps: int):
        return v3_schedule_geometry(
            corrective_queries=3482,
            robust_queries=required_steps * 16,
            harmful_queries=0,
            protective_queries=0,
            corrective_batch_size=4,
            protective_batch_size=8,
            maximum_auxiliary_microbatches_per_step=4,
            maximum_protective_microbatches_per_step=4,
            maximum_corrective_recycle_factor=4.0,
        )

    assert geometry(3484).cap_safe_corrective_batches == 871
    assert geometry(3484).effective_corrective_recycle_factor == 4.0
    assert geometry(3485).cap_safe_corrective_batches == 872
    assert geometry(3485).effective_corrective_recycle_factor < 4.0
    try:
        v3_schedule_geometry(
            corrective_queries=1,
            robust_queries=17,
            harmful_queries=0,
            protective_queries=0,
            corrective_batch_size=4,
            protective_batch_size=8,
            maximum_auxiliary_microbatches_per_step=1,
            maximum_protective_microbatches_per_step=1,
            maximum_corrective_recycle_factor=4.0,
        )
    except RuntimeError as error:
        assert "one complete query per batch" in str(error)
    else:
        raise AssertionError("impossible cap-safe schedule did not fail closed")


def _corrective(copies: int = 1):
    clean = torch.tensor([-0.05, 0.00], requires_grad=True)
    actions = [torch.tensor([0.10, 0.08], requires_grad=True) for _ in range(copies)]
    controls = [torch.tensor([0.00, 0.01], requires_grad=True) for _ in range(copies)]
    result = corrective_action_objective(
        [clean], [actions], [controls], [["N"] * copies],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=2,
        margin_transfer_fraction=0.5,
        margin_transfer_cap=0.10,
        lambda_margin_transfer=1.0,
        lambda_payload_rank=0.25,
    )
    return clean, actions, controls, result


def test_corrective_has_clean_transfer_and_real_action_gradient() -> None:
    clean, actions, controls, result = _corrective()
    result.loss.backward()
    assert float(result.margin_transfer) > 0
    assert float(result.payload_rank) > 0
    assert clean.grad is not None and float(clean.grad.abs().sum()) > 0
    assert actions[0].grad is not None and float(actions[0].grad.abs().sum()) > 0
    assert controls[0].grad is None or float(controls[0].grad.abs().sum()) == 0
    assert result.active_transfer_edges == 2
    assert result.capped_transfer_edges == 1
    assert result.considered_transfer_edges == 2


def test_control_semantics_are_stratified_under_one_conservative_ceiling() -> None:
    clean = torch.tensor([0.0], requires_grad=True)
    actions = [
        torch.tensor([0.20], requires_grad=True),
        torch.tensor([0.20], requires_grad=True),
    ]
    controls = [
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([0.19], requires_grad=True),
    ]
    result = corrective_action_objective(
        [clean], [actions], [controls],
        [["P::E10B|projection", "N::N_mature|candidate"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=1,
        margin_transfer_fraction=0.5,
        margin_transfer_cap=1.0,
        lambda_margin_transfer=1.0,
        lambda_payload_rank=0.0,
        action_control_semantic=[[
            "wrong_identity_direction", "matched_neutral",
        ]],
    )
    diagnostics = result.transfer_edge_diagnostics
    assert diagnostics["control_semantic=wrong_identity_direction"]["active"] == 1
    assert diagnostics["control_semantic=wrong_identity_direction"]["clean_limited"] == 1
    assert diagnostics["control_semantic=matched_neutral"]["active"] == 1
    assert diagnostics["control_semantic=matched_neutral"]["control_limited"] == 1
    assert diagnostics["family=projection"]["active"] == 1
    assert diagnostics["family=candidate"]["active"] == 1
    assert diagnostics["source_family=E10B|projection"]["considered"] == 1
    assert diagnostics["source_family=N_mature|candidate"]["considered"] == 1
    # The bad P control cannot make the inherited clean target exceed the
    # action-vs-clean improvement.
    assert diagnostics["mechanism=P"]["capped"] == 0


def test_corrective_safety_floor_repairs_action_without_moving_clean_or_control() -> None:
    clean = torch.tensor([0.08, 0.04], requires_grad=True)
    action = torch.tensor([0.01, -0.03], requires_grad=True)
    control = torch.tensor([-0.02, -0.04], requires_grad=True)
    result = corrective_action_objective(
        [clean], [[action]], [[control]], [["N::N_mature|candidate"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=2,
        margin_transfer_fraction=0.5,
        margin_transfer_cap=0.10,
        lambda_margin_transfer=0.0,
        lambda_payload_rank=0.0,
        action_safety_slack=0.005,
        lambda_payload_safety=1.0,
    )
    result.loss.backward()
    assert float(result.payload_safety.detach()) > 0
    assert result.active_safety_edges == 2
    assert action.grad is not None and float(action.grad.abs().sum()) > 0
    assert clean.grad is None or float(clean.grad.abs().sum()) == 0
    assert control.grad is None or float(control.grad.abs().sum()) == 0


def test_action_candidate_switch_is_not_dropped_by_clean_topk() -> None:
    # Clean considers edge 0 hardest, whereas the action makes edge 1 hardest.
    # The direct objective must retain both candidate boundaries when top-k=1.
    clean = torch.tensor([0.01, 0.90], requires_grad=True)
    action = torch.tensor([0.80, -0.10], requires_grad=True)
    control = torch.tensor([0.02, 0.70], requires_grad=True)
    result = corrective_action_objective(
        [clean], [[action]], [[control]], [["N::candidate"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=1,
        margin_transfer_fraction=0.5,
        margin_transfer_cap=0.10,
        lambda_margin_transfer=0.0,
        lambda_payload_rank=1.0,
    )
    result.loss.backward()
    assert result.considered_transfer_edges == 2
    assert action.grad is not None
    assert float(action.grad[0].abs()) > 0
    assert float(action.grad[1].abs()) > 0


def test_e8_symmetric_consistency_updates_both_live_views_only() -> None:
    clean = torch.tensor([1.0, 0.0], requires_grad=True)
    action = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = symmetric_live_action_consistency(
        [clean], [[action]], [["P::E10B|union"]],
    )
    loss.backward()
    assert float(loss.detach()) == 1.0
    assert clean.grad is not None and float(clean.grad.abs().sum()) > 0
    assert action.grad is not None and float(action.grad.abs().sum()) > 0


def test_corrective_action_duplication_does_not_multiply_dose() -> None:
    one = _corrective(1)[-1]
    many = _corrective(20)[-1]
    assert torch.allclose(one.loss, many.loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(one.margin_transfer, many.margin_transfer, atol=1e-7, rtol=1e-6)
    assert torch.allclose(one.payload_rank, many.payload_rank, atol=1e-7, rtol=1e-6)


def test_corrective_family_reduction_is_equal_not_path_weighted() -> None:
    def value(copies: int) -> torch.Tensor:
        clean = torch.tensor([-0.05, 0.00], requires_grad=True)
        n_actions = [torch.tensor([0.10, 0.08], requires_grad=True) for _ in range(copies)]
        p_action = torch.tensor([0.02, 0.03], requires_grad=True)
        actions = n_actions + [p_action]
        controls = [torch.zeros(2, requires_grad=True) for _ in actions]
        result = corrective_action_objective(
            [clean], [actions], [controls], [["N"] * copies + ["P"]],
            rank_margin=0.05,
            rank_temperature=0.10,
            topk_negatives=2,
            margin_transfer_fraction=0.5,
            margin_transfer_cap=0.10,
            lambda_margin_transfer=1.0,
            lambda_payload_rank=0.25,
        )
        return result.loss
    assert torch.allclose(value(1), value(20), atol=1e-7, rtol=1e-6)


def test_mechanism_blocks_are_equal_despite_more_p_source_families() -> None:
    def value(p_leaf_copies: int) -> torch.Tensor:
        clean = torch.tensor([-0.05], requires_grad=True)
        n_action = torch.tensor([0.10], requires_grad=True)
        p_actions = [
            torch.tensor([0.02], requires_grad=True) for _ in range(p_leaf_copies)
        ]
        actions = [n_action] + p_actions
        controls = [torch.tensor([0.0], requires_grad=True) for _ in actions]
        groups = ["N::N_mature|candidate"] + [
            f"P::E{index}|projection" for index in range(p_leaf_copies)
        ]
        return corrective_action_objective(
            [clean], [actions], [controls], [groups],
            rank_margin=0.05,
            rank_temperature=0.10,
            topk_negatives=1,
            margin_transfer_fraction=0.5,
            margin_transfer_cap=0.10,
            lambda_margin_transfer=0.0,
            lambda_payload_rank=1.0,
        ).loss
    assert torch.allclose(value(1), value(12), atol=1e-7, rtol=1e-6)


def test_same_p_family_repeated_across_generations_does_not_gain_votes() -> None:
    def value(repeated_sources: int) -> torch.Tensor:
        clean = torch.tensor([-0.05], requires_grad=True)
        projection = torch.tensor([0.10], requires_grad=True)
        recurrent = [
            torch.tensor([-0.02], requires_grad=True)
            for _ in range(repeated_sources)
        ]
        actions = [projection] + recurrent
        controls = [torch.tensor([0.0], requires_grad=True) for _ in actions]
        groups = ["P::E10B|consensus_projection"] + [
            f"P::generation{index}|recurrent_union_mix"
            for index in range(repeated_sources)
        ]
        return corrective_action_objective(
            [clean], [actions], [controls], [groups],
            rank_margin=0.05,
            rank_temperature=0.10,
            topk_negatives=1,
            margin_transfer_fraction=0.5,
            margin_transfer_cap=0.10,
            lambda_margin_transfer=0.0,
            lambda_payload_rank=1.0,
        ).loss
    assert torch.allclose(value(1), value(7), atol=1e-7, rtol=1e-6)


def test_inverse_multiplicity_query_weights_make_identities_equal() -> None:
    def objective(actions, weights=None):
        clean = [torch.tensor([-0.05], requires_grad=True) for _ in actions]
        action = [[torch.tensor([value], requires_grad=True)] for value in actions]
        control = [[torch.tensor([0.0], requires_grad=True)] for _ in actions]
        return corrective_action_objective(
            clean, action, control, [["P"] for _ in actions],
            rank_margin=0.05,
            rank_temperature=0.10,
            topk_negatives=1,
            margin_transfer_fraction=0.5,
            margin_transfer_cap=0.10,
            lambda_margin_transfer=0.0,
            lambda_payload_rank=1.0,
            query_weight=weights,
        ).loss
    identity_equal = objective([0.10, -0.05])
    duplicated_first_identity = objective([0.10, 0.10, -0.05], [0.75, 0.75, 1.5])
    assert torch.allclose(identity_equal, duplicated_first_identity, atol=1e-7, rtol=1e-6)


def test_robust_updates_action_view_not_clean_view() -> None:
    clean = torch.tensor([0.08, 0.04], requires_grad=True)
    action = torch.tensor([0.02, -0.02], requires_grad=True)
    result = robust_action_objective(
        [clean], [[action]], [["A4"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=2,
        safety_slack=0.01,
        lambda_payload_rank=0.25,
        lambda_safety_floor=0.25,
    )
    result.loss.backward()
    assert action.grad is not None and float(action.grad.abs().sum()) > 0
    assert clean.grad is None or float(clean.grad.abs().sum()) == 0


def test_harmful_content_selects_clean_gradient_and_is_never_imitated() -> None:
    clean = torch.tensor([0.02, 0.01], requires_grad=True)
    harmful = torch.tensor([-0.10, -0.08], requires_grad=True)
    baseline = torch.tensor([0.08, 0.06])
    result = harmful_boundary_objective(
        [clean], [[harmful]], [baseline], [["E10B"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=2,
        harmful_damage_slack=0.01,
        baseline_floor_slack=0.01,
        lambda_boundary_rank=1.0,
        lambda_baseline_floor=1.0,
    )
    result.loss.backward()
    assert result.active_damage_edges == 2
    assert clean.grad is not None and float(clean.grad.abs().sum()) > 0
    assert harmful.grad is None or float(harmful.grad.abs().sum()) == 0


def test_non_damaging_action_has_zero_harmful_branch() -> None:
    clean = torch.tensor([0.02, 0.01], requires_grad=True)
    safe = torch.tensor([0.10, 0.08], requires_grad=True)
    result = harmful_boundary_objective(
        [clean], [[safe]], [torch.tensor([0.02, 0.01])], [["P"]],
        rank_margin=0.05,
        rank_temperature=0.10,
        topk_negatives=2,
        harmful_damage_slack=0.01,
        baseline_floor_slack=0.01,
        lambda_boundary_rank=1.0,
        lambda_baseline_floor=1.0,
    )
    result.loss.backward()
    assert result.active_damage_edges == 0
    assert float(result.loss.detach()) == 0.0
    assert clean.grad is not None and float(clean.grad.abs().sum()) == 0.0
    assert safe.grad is None


def test_v4_mass_neutral_transfer_changes_resolution_not_total_clean_push() -> None:
    def run(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        clean = torch.tensor([-0.10, -0.05, 0.00], dtype=torch.float64, requires_grad=True)
        actions = [
            torch.tensor([0.30, 0.15, 0.08], dtype=torch.float64),
            torch.tensor([0.70, 0.25, 0.03], dtype=torch.float64),
        ]
        controls = [torch.full((3,), -1.0, dtype=torch.float64) for _ in actions]
        result = corrective_action_objective(
            [clean], [actions], [controls],
            [["N::source|family", "N::source|family"]],
            rank_margin=0.05,
            rank_temperature=0.10,
            topk_negatives=3,
            margin_transfer_fraction=0.5,
            margin_transfer_cap=0.10,
            lambda_margin_transfer=1.0,
            lambda_payload_rank=0.0,
            transfer_target_allocation=mode,
            maximum_transfer_cap_factor=2.0,
        )
        result.loss.backward()
        return result.margin_transfer.detach(), clean.grad.detach()

    old_loss, old_gradient = run("hard_cap")
    new_loss, new_gradient = run("mass_neutral_monotone")
    assert not torch.allclose(old_loss, new_loss)
    assert not torch.allclose(old_gradient, new_gradient)
    assert torch.allclose(old_gradient.sum(), new_gradient.sum(), atol=1e-12)


def main() -> None:
    tests = [
        test_balanced_plan_has_no_protect_only_tail_or_coverage_loss,
        test_balanced_plan_is_seed_deterministic,
        test_balanced_plan_can_repeat_actions_without_increasing_epoch_dose,
        test_full_dose_recycling_keeps_every_optimizer_step_action_strength,
        test_balanced_plan_respects_auxiliary_minimum_steps,
        test_v6_schedule_geometry_is_exactly_cap_safe_before_training,
        test_v6_cap_safe_repartition_is_even_order_preserving_and_four_dose,
        test_cap_safe_geometry_has_no_tolerance_leak_and_fails_if_queries_cannot_split,
        test_corrective_has_clean_transfer_and_real_action_gradient,
        test_control_semantics_are_stratified_under_one_conservative_ceiling,
        test_corrective_safety_floor_repairs_action_without_moving_clean_or_control,
        test_action_candidate_switch_is_not_dropped_by_clean_topk,
        test_e8_symmetric_consistency_updates_both_live_views_only,
        test_corrective_action_duplication_does_not_multiply_dose,
        test_corrective_family_reduction_is_equal_not_path_weighted,
        test_mechanism_blocks_are_equal_despite_more_p_source_families,
        test_same_p_family_repeated_across_generations_does_not_gain_votes,
        test_inverse_multiplicity_query_weights_make_identities_equal,
        test_robust_updates_action_view_not_clean_view,
        test_harmful_content_selects_clean_gradient_and_is_never_imitated,
        test_non_damaging_action_has_zero_harmful_branch,
        test_v4_mass_neutral_transfer_changes_resolution_not_total_clean_push,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_direct_v3_core] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
