"""Focused invariants for direct-v3 trainer reference wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from types import SimpleNamespace

from train_noise_corrected_routed_direct import (
    BoundaryExample,
    REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY,
    REGISTERED_V3_EXACT_CONFIGURATION,
    REGISTERED_V3_FLOAT_CONFIGURATION,
    RoutedAction,
    augment_action_conditioned_references,
    _bounded_calibration_scale,
    _bounded_dense_auxiliary_scale,
    _formula_stratified_batches,
    _finalize_transfer_edge_breakdown,
    _gradient_cosine,
    _gradient_direction_retention,
    _batch_cardinality_scale,
    _global_mechanism_weights,
    _interleaved_single_exposure_auxiliary_schedules,
    _identity_stratified_batches,
    _legacy_90pct_end_to_end_loss_reproduced,
    _legacy_90pct_signal_boundary_observed,
    _mechanism_block,
    _optimizer_parameter_group_positions,
    _preserve_corrective_against_auxiliary,
    _prioritize_mechanism_coverage,
    _registered_report_configuration,
    _single_exposure_auxiliary_schedule,
    _take_gradient_positions,
    _union_initial_negative_molecules,
    _validate_registered_formal_v3_configuration,
    _validate_registered_best_action_v6_schedule_geometry,
    _virtual_adamw_descent_updates,
    _weighted_mechanism_mass,
    select_references,
    select_references_from_pair_scores,
    v3_action_panel_batch_loss,
)


class TinyGraph:
    query_row = np.asarray([0], dtype=np.int64)

    def query_block(self, query: int):
        assert query == 0
        return (
            slice(0, 5),
            np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
            np.asarray([0, 2, 4, 5], dtype=np.int64),
            0,
        )


class TinySpectrumStore:
    def __init__(self, values: dict[int, np.ndarray]):
        self.values = values

    def one(self, row: int) -> torch.Tensor:
        return torch.as_tensor(self.values[int(row)], dtype=torch.float32)


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(spectra[:, 0, :] @ self.weight, dim=1)


def test_query_local_corrective_stops_only_reference_gradient() -> None:
    values = {
        10: np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float32),
        11: np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        12: np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    }
    action = np.asarray([[[0.8, 0.6], [0.0, 0.0]]], dtype=np.float32)
    control = np.asarray([[[0.6, 0.8], [0.0, 0.0]]], dtype=np.float32)
    example = BoundaryExample(
        query_index=0,
        query_row=10,
        identity="id0",
        formula="F0",
        actions=(RoutedAction(
            "a0", 0, "N::N_mature|candidate_gradient",
        ),),
        positive_rows=(11,),
        negative_rows=((12,),),
    )
    base = dict(
        amp=False,
        maximum_spectra_per_action_forward=8,
        rank_margin=0.05,
        temperature=0.10,
        topk_negatives=1,
        margin_transfer_fraction=0.5,
        margin_transfer_cap=0.1,
        lambda_margin_transfer=1.0,
        lambda_action_rank=0.25,
        action_safety_slack=0.005,
        lambda_action_safety=0.25,
        lambda_v3_corrective_consistency=0.25,
        transfer_target_allocation="hard_cap",
        maximum_transfer_cap_factor=2.0,
    )
    anchor = {
        row: vector[0] / np.linalg.norm(vector[0])
        for row, vector in values.items()
    }
    shared_args = SimpleNamespace(
        **base, corrective_gradient_locality="shared",
    )
    _, shared, _ = v3_action_panel_batch_loss(
        TinyEmbeddingModel(), TinySpectrumStore(values), [example],
        action, control, anchor, torch.device("cpu"), shared_args,
        supervision_kind="corrective", diagnose_embedding_gradients=True,
    )
    local_args = SimpleNamespace(
        **base, corrective_gradient_locality="query_action_only",
    )
    _, local, _ = v3_action_panel_batch_loss(
        TinyEmbeddingModel(), TinySpectrumStore(values), [example],
        action, control, anchor, torch.device("cpu"), local_args,
        supervision_kind="corrective", diagnose_embedding_gradients=True,
    )
    assert shared["embedding_grad_transfer_reference_norm"] > 0
    assert shared["embedding_grad_payload_reference_norm"] > 0
    assert local["embedding_grad_transfer_reference_norm"] == 0
    assert local["embedding_grad_payload_reference_norm"] == 0
    assert local["embedding_grad_transfer_query_norm"] > 0
    assert local["embedding_grad_payload_action_norm"] > 0
    assert local["embedding_grad_consistency_query_norm"] > 0
    assert local["embedding_grad_consistency_action_norm"] > 0


def test_all_registered_sources_map_to_three_mechanisms() -> None:
    assert {
        source: _mechanism_block(source)
        for source in (
            "N_mature", "V4_gradient_path", "P_guided_original", "E10B", "E11",
            "E12B", "A4_exact",
        )
    } == {
        "N_mature": "N", "V4_gradient_path": "N", "P_guided_original": "P",
        "E10B": "P", "E11": "P", "E12B": "P", "A4_exact": "A4",
    }


def test_vectorized_pair_score_reference_selection_matches_direct_dot() -> None:
    graph = TinyGraph()
    rows = np.arange(6, dtype=np.int64)
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.6, 0.4],
    ], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = {int(row): int(row) for row in rows}
    pair_scores = embeddings[[1, 2, 3, 4, 5]] @ embeddings[0]
    direct = select_references(
        graph, 0, embeddings, index, positives=2, negatives=2, per_negative=2,
    )
    cached = select_references_from_pair_scores(
        graph, 0, pair_scores, positives=2, negatives=2, per_negative=2,
    )
    assert direct == cached


def test_identity_stratified_batches_cover_once_and_spread_repeated_views() -> None:
    examples = [
        BoundaryExample(
            query_index=index,
            query_row=index,
            identity=("repeat" if index < 5 else f"id{index}"),
            formula=f"F{index}",
            actions=(),
            positive_rows=(1,),
            negative_rows=((2,),),
        )
        for index in range(12)
    ]
    batches = _identity_stratified_batches(
        examples, 4, np.random.default_rng(9),
    )
    flattened = [example for batch in batches for example in batch]
    assert sorted(example.query_index for example in flattened) == list(range(12))
    # While at least four identities remain, no batch needs two views from the
    # repeated identity.
    assert all(
        sum(example.identity == "repeat" for example in batch) <= 1
        for batch in batches[:2]
    )


def test_formula_stratified_calibration_batches_are_formula_diverse() -> None:
    examples = [
        BoundaryExample(
            query_index=index,
            query_row=index,
            identity=f"id{index}",
            formula=f"F{index % 5}",
            actions=(),
            positive_rows=(1,),
            negative_rows=((2,),),
        )
        for index in range(15)
    ]
    batches = _formula_stratified_batches(
        examples, 4, np.random.default_rng(19),
    )
    assert sorted(
        example.query_index for batch in batches for example in batch
    ) == list(range(15))
    assert all(
        len({example.formula for example in batch}) == len(batch)
        for batch in batches
    )


def test_global_mechanism_weights_cancel_cross_query_p_coverage() -> None:
    actions = pd.DataFrame([
        {"query_index": 0, "source": "N_mature"},
        {"query_index": 0, "source": "E10B"},
        {"query_index": 1, "source": "E11"},
        {"query_index": 1, "source": "A4_exact"},
        {"query_index": 2, "source": "E12B"},
    ])
    coefficients, report = _global_mechanism_weights(
        actions, {0: 1.0, 1: 1.0, 2: 1.0},
    )
    assert report["balanced"] is True
    assert np.isclose(coefficients["P"], 1 / 3)
    assert np.isclose(coefficients["N"], 1.0)
    assert np.isclose(coefficients["A4"], 1.0)
    assert np.allclose(list(report["effective_epoch_mass"].values()), 1.0)


def test_calibration_prefix_prioritizes_rare_mechanisms_without_rebatching() -> None:
    def example(index: int, mechanism: str) -> BoundaryExample:
        source = {"N": "N_mature", "P": "E10B", "A4": "A4_exact"}[mechanism]
        return BoundaryExample(
            query_index=index,
            query_row=index,
            identity=f"id{index}",
            formula=f"F{index}",
            actions=(RoutedAction(
                f"a{index}", index, f"{mechanism}::{source}|family",
            ),),
            positive_rows=(1,),
            negative_rows=((2,),),
        )
    batches = [
        [example(0, "P")],
        [example(1, "P")],
        [example(2, "P")],
        [example(3, "N")],
        [example(4, "A4")],
    ]
    prioritized = _prioritize_mechanism_coverage(batches)
    first_three = {
        action.group.split("::", 1)[0]
        for batch in prioritized[:3]
        for item in batch for action in item.actions
    }
    assert first_three == {"N", "P", "A4"}
    assert {id(batch) for batch in prioritized} == {id(batch) for batch in batches}


def test_actual_schedule_mass_preserves_global_mechanism_balance() -> None:
    def example(index: int, mechanisms: tuple[str, ...]) -> BoundaryExample:
        coefficients = {"P": 1 / 3, "N": 1.0, "A4": 1.0}
        sources = {"P": "E10B", "N": "N_mature", "A4": "A4_exact"}
        return BoundaryExample(
            query_index=index,
            query_row=index,
            identity=f"id{index}",
            formula=f"F{index}",
            actions=tuple(
                RoutedAction(
                    f"{mechanism}{index}", index,
                    f"{mechanism}::{sources[mechanism]}|family",
                    mechanism_weight=coefficients[mechanism],
                )
                for mechanism in mechanisms
            ),
            positive_rows=(1,), negative_rows=((2,),), epoch_weight=1.0,
        )
    examples = [
        example(0, ("N", "P")),
        example(1, ("A4", "P")),
        example(2, ("P",)),
    ]
    mass = _weighted_mechanism_mass(examples, scale=2.0)
    assert np.allclose([mass["N"], mass["P"], mass["A4"]], 2.0)


def test_sparse_auxiliary_schedule_is_single_exposure_without_duty_amplification() -> None:
    batches = [[object()], [object()], [object()]]
    schedule = _single_exposure_auxiliary_schedule(
        batches, 17, np.random.default_rng(29),
    )
    flattened = [batch for step in schedule for batch in step]
    assert len(schedule) == 17
    assert len(flattened) == len(batches)
    assert {id(batch) for batch in flattened} == {id(batch) for batch in batches}
    assert sum(bool(step) for step in schedule) == len(batches)
    active_positions = [index for index, step in enumerate(schedule) if step]
    assert active_positions != list(range(len(batches)))
    assert max(active_positions) - min(active_positions) >= 17 // len(batches)


def test_sparse_semantic_panels_are_interleaved_without_avoidable_overlap() -> None:
    robust = [[object()] for _ in range(3)]
    harmful = [[object()] for _ in range(2)]
    schedules = _interleaved_single_exposure_auxiliary_schedules(
        {"robust": robust, "harmful": harmful},
        17,
        np.random.default_rng(31),
    )
    combined = [
        len(schedules["robust"][index]) + len(schedules["harmful"][index])
        for index in range(17)
    ]
    assert max(combined) == 1
    assert sum(bool(value) for value in combined) == 5
    for name, original in (("robust", robust), ("harmful", harmful)):
        flattened = [batch for step in schedules[name] for batch in step]
        assert {id(batch) for batch in flattened} == {id(batch) for batch in original}


def test_dense_semantic_panels_keep_coverage_bounded_load_and_bounded_mass() -> None:
    for robust_count, harmful_count, corrective_steps in (
        (10, 7, 3), (21, 3, 5), (8, 8, 4), (1, 1, 1),
    ):
        robust = [[object()] for _ in range(robust_count)]
        harmful = [[object()] for _ in range(harmful_count)]
        steps = max(corrective_steps, int(np.ceil(
            (robust_count + harmful_count) / 4,
        )))
        schedules = _interleaved_single_exposure_auxiliary_schedules(
            {"robust": robust, "harmful": harmful},
            steps,
            np.random.default_rng(37),
        )
        combined = [
            len(schedules["robust"][index])
            + len(schedules["harmful"][index])
            for index in range(steps)
        ]
        assert max(combined) <= 4
        assert max(combined) - min(combined) <= 1
        for name, batches in (("robust", robust), ("harmful", harmful)):
            flattened = [batch for step in schedules[name] for batch in step]
            assert {id(batch) for batch in flattened} == {id(batch) for batch in batches}
            assert len(flattened) == len(batches)
            scale = _bounded_dense_auxiliary_scale(len(batches), steps)
            assert np.isclose(scale * len(batches), min(len(batches), steps))


def test_partial_batch_mean_is_scaled_to_registered_query_mass() -> None:
    assert np.isclose(_batch_cardinality_scale([1, 2, 3, 4], 4), 1.0)
    assert np.isclose(_batch_cardinality_scale([1], 4), 0.25)
    try:
        _batch_cardinality_scale([1, 2, 3, 4, 5], 4)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized batch passed registered cardinality gate")


def test_transfer_edge_breakdown_uses_pooled_counts_not_batch_means() -> None:
    report = _finalize_transfer_edge_breakdown({
        "all": {
            "considered": 10,
            "action_better_clean": 5,
            "action_better_control": 4,
            "active": 3,
            "clean_limited": 6,
            "control_limited": 4,
            "capped": 2,
        },
        "family=projection": {
            "considered": 4,
            "active": 1,
        },
    })
    assert report["all"]["active"] == 3
    assert np.isclose(report["all"]["active_fraction"], 0.3)
    assert np.isclose(report["family=projection"]["active_fraction"], 0.25)


def test_calibration_caps_fail_closed_instead_of_silently_underdosing() -> None:
    value, truncated = _bounded_calibration_scale(
        3.0, 4.0, "test", require_exact=True,
    )
    assert value == 3.0 and truncated is False
    try:
        _bounded_calibration_scale(5.0, 4.0, "test", require_exact=True)
    except RuntimeError as error:
        assert "silently under-dosed" in str(error)
    else:
        raise AssertionError("active calibration silently accepted a truncated scale")
    value, truncated = _bounded_calibration_scale(
        5.0, 4.0, "inactive control", require_exact=False,
    )
    assert value == 4.0 and truncated is True


def test_reference_refresh_unions_molecules_without_duplicating_same_molecule() -> None:
    def example(negative_rows, positive_rows=(1, 2)):
        return BoundaryExample(
            query_index=0,
            query_row=0,
            identity="q",
            formula="F",
            actions=(),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
        )

    initial = [example(((3,), (5,)))]
    # Row 4 is another spectrum from the same candidate molecule as row 3.
    current = [example(((4,),), positive_rows=(1,))]
    merged = _union_initial_negative_molecules(TinyGraph(), initial, current)
    assert merged[0].positive_rows == (1, 2)
    assert merged[0].negative_rows == ((4, 3), (5,))


def test_action_conditioned_reference_adds_exact_switch_with_shared_boundary() -> None:
    graph = TinyGraph()
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1], [0.8, 0.2],
        [0.7, 0.3], [0.2, 0.8],
        [0.6, 0.4],
    ], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    example = BoundaryExample(
        query_index=0,
        query_row=0,
        identity="q",
        formula="F",
        actions=(RoutedAction("a", 0, "N::N_mature|candidate"),),
        positive_rows=(1,),
        negative_rows=((5,),),
    )
    rows = pd.DataFrame([{
        "action_id": "a", "query_index": 0,
        "action_positive_row": 2,
        "action_hard_negative_molecule_index": 1,
        "action_hard_negative_row": 4,
        "control_positive_row": 1,
        "control_hard_negative_molecule_index": 2,
        "control_hard_negative_row": 5,
    }])
    augmented, report = augment_action_conditioned_references(
        graph, [example], rows, embeddings, {index: index for index in range(6)},
        maximum_positive_references=2,
        maximum_negative_molecules=2,
        references_per_negative=2,
    )
    assert augmented[0].positive_rows == (1, 2)
    assert augmented[0].negative_rows == ((5,), (4, 3))
    assert report["action_conditioned_candidate_switch_queries"] == 1


def test_action_conditioned_reference_merges_new_row_for_existing_molecule() -> None:
    graph = TinyGraph()
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1], [0.8, 0.2],
        [0.7, 0.3], [0.2, 0.8],
        [0.6, 0.4],
    ], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    example = BoundaryExample(
        query_index=0,
        query_row=0,
        identity="q",
        formula="F",
        actions=(RoutedAction("a", 0, "N::N_mature|candidate"),),
        positive_rows=(1,),
        # Molecule one is already present, but only through row three.
        negative_rows=((3,),),
    )
    rows = pd.DataFrame([{
        "action_id": "a", "query_index": 0,
        "action_positive_row": 2,
        "action_hard_negative_molecule_index": 1,
        "action_hard_negative_row": 4,
        "control_positive_row": 1,
        "control_hard_negative_molecule_index": 1,
        "control_hard_negative_row": 3,
    }])
    augmented, report = augment_action_conditioned_references(
        graph, [example], rows, embeddings, {index: index for index in range(6)},
        maximum_positive_references=2,
        maximum_negative_molecules=1,
        # The exact routed row must survive even when this ordinary clean-row
        # supplement limit is already full.
        references_per_negative=1,
    )
    assert augmented[0].positive_rows == (1, 2)
    assert augmented[0].negative_rows == ((3, 4),)
    assert report["action_conditioned_candidate_switch_queries"] == 1
    assert report["negative_molecules_added_maximum"] == 0
    assert report["negative_spectrum_rows_added_maximum"] == 1
    assert report["routed_hard_rows_never_truncated_within_molecule"] is True


def test_virtual_adamw_update_matches_real_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.02,
    )
    for gradient in (
        torch.tensor([0.3, -0.4], dtype=torch.float64),
        torch.tensor([-0.2, 0.1], dtype=torch.float64),
    ):
        expected = _virtual_adamw_descent_updates(
            optimizer, [parameter], [gradient],
        )[0]
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()
        optimizer.step()
        actual = before - parameter.detach()
        assert expected is not None
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-10)
        optimizer.zero_grad(set_to_none=True)

    combined = _virtual_adamw_descent_updates(
        optimizer, [parameter], [torch.tensor([0.2, -0.3], dtype=torch.float64)],
    )[0]
    risk_only = _virtual_adamw_descent_updates(
        optimizer, [parameter], [torch.tensor([0.1, -0.1], dtype=torch.float64)],
    )[0]
    assert combined is not None and risk_only is not None
    assert float(torch.linalg.vector_norm(combined - risk_only)) > 0

    # The production model is FP32 and many individual parameter updates are
    # close to its quantization scale.  The audit must reproduce the realized
    # parameter write, not only an algebraically equivalent unrounded update.
    generator = torch.Generator().manual_seed(20260907)
    fp32_parameter = torch.nn.Parameter(torch.randn(65536, generator=generator))
    fp32_optimizer = torch.optim.AdamW(
        [fp32_parameter], lr=2e-6, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=1e-4, foreach=True,
    )
    for _ in range(3):
        gradient = torch.randn(65536, generator=generator) * 1e-3
        expected = _virtual_adamw_descent_updates(
            fp32_optimizer, [fp32_parameter], [gradient],
        )[0]
        before = fp32_parameter.detach().clone()
        fp32_parameter.grad = gradient.clone()
        fp32_optimizer.step()
        actual = before - fp32_parameter.detach()
        assert expected is not None
        relative_error = float(torch.linalg.vector_norm(actual - expected)) / max(
            float(torch.linalg.vector_norm(actual)), 1e-30,
        )
        assert relative_error <= 1e-6
        fp32_optimizer.zero_grad(set_to_none=True)


def test_optimizer_parameter_groups_partition_action_gradient_ledger() -> None:
    head = torch.nn.Parameter(torch.tensor([1.0]))
    backbone = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    parameters = [backbone, head]
    optimizer = torch.optim.AdamW([
        {"params": [head], "group_name": "head"},
        {"params": [backbone], "group_name": "backbone"},
    ], lr=1e-3)
    positions = _optimizer_parameter_group_positions(optimizer, parameters)
    assert positions == {"head": [1], "backbone": [0]}
    gradients = [torch.tensor([0.2, 0.3]), torch.tensor([0.4])]
    assert _take_gradient_positions(gradients, positions["head"])[0] is gradients[1]
    assert _take_gradient_positions(gradients, positions["backbone"])[0] is gradients[0]


def test_inner_semantic_projection_cannot_erase_corrective_direction() -> None:
    corrective = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([-3.0, 4.0])]
    combined, projected_auxiliary, report = _preserve_corrective_against_auxiliary(
        corrective, auxiliary,
    )
    assert torch.allclose(projected_auxiliary[0], torch.tensor([0.0, 4.0]))
    assert torch.allclose(combined[0], torch.tensor([1.0, 4.0]))
    assert report["conflict"] is True
    assert report["corrective_direction_preserved"] is True
    assert np.isclose(report["corrective_direction_preservation_ratio"], 1.0)

    agreeing, kept, agreeing_report = _preserve_corrective_against_auxiliary(
        corrective, [torch.tensor([2.0, 3.0])],
    )
    assert torch.allclose(kept[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(agreeing[0], torch.tensor([3.0, 3.0]))
    assert agreeing_report["conflict"] is False
    assert agreeing_report["corrective_direction_preservation_ratio"] > 1.0

    grouped, grouped_auxiliary, grouped_report = (
        _preserve_corrective_against_auxiliary(
            [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
            [torch.tensor([-2.0, 0.0]), torch.tensor([3.0, 0.0])],
            {"backbone": [0], "head": [1]},
        )
    )
    assert torch.allclose(grouped_auxiliary[0], torch.zeros(2))
    assert torch.allclose(grouped_auxiliary[1], torch.tensor([3.0, 0.0]))
    assert torch.allclose(grouped[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(grouped[1], torch.tensor([4.0, 0.0]))
    assert all(
        value["corrective_direction_preserved"]
        for value in grouped_report["parameter_groups"].values()
    )


def test_gradient_cosine_is_bounded_and_marks_inactive_branches() -> None:
    assert np.isclose(
        _gradient_cosine(
            [torch.tensor([1.0, 0.0]), None],
            [torch.tensor([0.5, 0.5]), None],
        ),
        1.0 / np.sqrt(2.0),
    )
    assert _gradient_cosine(
        [torch.zeros(2), None], [torch.ones(2), None],
    ) is None
    assert np.isclose(
        _gradient_direction_retention(
            [torch.tensor([0.2, 3.0])], [torch.tensor([1.0, 0.0])],
        ),
        0.2,
    )
    assert _gradient_direction_retention(
        [torch.ones(2)], [torch.zeros(2)],
    ) is None


def test_legacy_90pct_boundary_is_reported_without_tightening_registered_gate() -> None:
    safe_group = {
        "pcgrad_clip_action_retention_p10": 0.100001,
        "optimizer_action_attributable_update_fraction_p10": 0.100001,
        "corrective_direction_retention_after_risk_and_clip_p10": 0.100001,
        "optimizer_action_attributable_corrective_alignment_p10": 0.050001,
    }

    def reproduced(**updates) -> bool:
        report = {
            "optimizer_action_attributable_update_fraction_p10": 0.100001,
            "corrective_direction_retention_after_risk_and_clip_p10": 0.100001,
            "optimizer_action_attributable_corrective_alignment_p10": 0.050001,
            "parameter_group_action_signal": {
                "head": dict(safe_group), "backbone": dict(safe_group),
            },
            **updates,
        }
        return _legacy_90pct_end_to_end_loss_reproduced(
            0.100001,
            report,
            active_v3_arm=True,
            minimum_attributable_fraction_p10=0.10,
            minimum_corrective_direction_retention_p10=0.10,
            minimum_optimizer_action_alignment_p10=0.05,
        )

    assert reproduced() is False
    assert reproduced(
        optimizer_action_attributable_update_fraction_p10=0.10,
    ) is False
    assert reproduced(
        corrective_direction_retention_after_risk_and_clip_p10=0.10,
    ) is False

    group_boundary = dict(safe_group)
    group_boundary["optimizer_action_attributable_update_fraction_p10"] = 0.10
    assert reproduced(
        parameter_group_action_signal={
            "head": group_boundary, "backbone": dict(safe_group),
        },
    ) is False
    corrective_group_boundary = dict(safe_group)
    corrective_group_boundary[
        "corrective_direction_retention_after_risk_and_clip_p10"
    ] = 0.10
    assert reproduced(
        parameter_group_action_signal={
            "head": corrective_group_boundary, "backbone": dict(safe_group),
        },
    ) is False

    assert _legacy_90pct_signal_boundary_observed(
        0.100001,
        {
            "optimizer_action_attributable_update_fraction_p10": 0.10,
            "corrective_direction_retention_after_risk_and_clip_p10": 0.100001,
            "parameter_group_action_signal": {
                "head": dict(safe_group), "backbone": dict(safe_group),
            },
        },
        active_v3_arm=True,
    ) is True

    assert _legacy_90pct_end_to_end_loss_reproduced(
        0.10,
        {},
        active_v3_arm=False,
        minimum_attributable_fraction_p10=0.10,
        minimum_corrective_direction_retention_p10=0.10,
        minimum_optimizer_action_alignment_p10=0.05,
    ) is True
    assert _legacy_90pct_end_to_end_loss_reproduced(
        0.100001,
        {},
        active_v3_arm=False,
        minimum_attributable_fraction_p10=0.10,
        minimum_corrective_direction_retention_p10=0.10,
        minimum_optimizer_action_alignment_p10=0.05,
    ) is False


def test_registered_formal_v3_configuration_fails_closed_on_drift() -> None:
    values = {
        **REGISTERED_V3_EXACT_CONFIGURATION,
        **REGISTERED_V3_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
    }
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    values["grad_clip"] = 2.0
    try:
        _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    except RuntimeError as error:
        assert "grad_clip" in str(error)
    else:
        raise AssertionError("formal v3 accepted a rejected clip-2 configuration")
    values["development"] = True
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    truncated = {
        **REGISTERED_V3_EXACT_CONFIGURATION,
        **REGISTERED_V3_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
        "maximum_clean_queries": 128,
    }
    try:
        _validate_registered_formal_v3_configuration(SimpleNamespace(**truncated))
    except RuntimeError as error:
        assert "maximum_clean_queries" in str(error)
    else:
        raise AssertionError("formal v3 accepted a development query truncation")


def test_registered_v6_report_configuration_is_complete() -> None:
    values = {
        **REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
        **REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    }
    reported = _registered_report_configuration(SimpleNamespace(**values))
    assert set(reported) == set(values)
    assert reported == values


def test_registered_v6_cap_safe_schedule_is_exact_and_pre_model() -> None:
    schedule = dict(REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY)
    _validate_registered_best_action_v6_schedule_geometry(
        schedule, context="test",
    )
    assert schedule["original_corrective_batches"] == 871
    assert schedule["cap_safe_corrective_batches"] == 877
    assert schedule["corrective_batches_added_by_cap_safe_repartition"] == 6
    assert schedule["required_optimizer_steps"] == 3508
    assert schedule["effective_corrective_recycle_factor"] == 4.0
    assert schedule["minimum_size_corrective_batches"] == 26
    assert schedule["maximum_size_corrective_batches"] == 851
    tampered = dict(schedule)
    tampered["cap_safe_corrective_batches"] = 876
    try:
        _validate_registered_best_action_v6_schedule_geometry(
            tampered, context="tamper-test",
        )
    except RuntimeError as error:
        assert "cap_safe_corrective_batches" in str(error)
    else:
        raise AssertionError("registered v6 schedule accepted a stale batch count")

    source = Path(__file__).with_name(
        "train_noise_corrected_routed_direct.py"
    ).read_text(encoding="utf-8")
    preflight = source.index("pre_model_schedule_geometry = v3_schedule_geometry(")
    model_load = source.index("store = SpectrumStore(args.data, reachable")
    assert preflight < model_load


def test_dense_auxiliary_scale_preserves_sparse_mass_and_caps_dense_mass() -> None:
    assert _bounded_dense_auxiliary_scale(3, 8) == 1.0
    assert np.isclose(_bounded_dense_auxiliary_scale(10, 4), 0.4)
    assert np.isclose(10 * _bounded_dense_auxiliary_scale(10, 4), 4.0)
    for batches, steps in ((0, 4), (4, 0)):
        try:
            _bounded_dense_auxiliary_scale(batches, steps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dense auxiliary scale was accepted")


def main() -> None:
    test_query_local_corrective_stops_only_reference_gradient()
    test_all_registered_sources_map_to_three_mechanisms()
    test_vectorized_pair_score_reference_selection_matches_direct_dot()
    test_identity_stratified_batches_cover_once_and_spread_repeated_views()
    test_formula_stratified_calibration_batches_are_formula_diverse()
    test_global_mechanism_weights_cancel_cross_query_p_coverage()
    test_calibration_prefix_prioritizes_rare_mechanisms_without_rebatching()
    test_actual_schedule_mass_preserves_global_mechanism_balance()
    test_sparse_auxiliary_schedule_is_single_exposure_without_duty_amplification()
    test_sparse_semantic_panels_are_interleaved_without_avoidable_overlap()
    test_dense_semantic_panels_keep_coverage_bounded_load_and_bounded_mass()
    test_partial_batch_mean_is_scaled_to_registered_query_mass()
    test_transfer_edge_breakdown_uses_pooled_counts_not_batch_means()
    test_calibration_caps_fail_closed_instead_of_silently_underdosing()
    test_reference_refresh_unions_molecules_without_duplicating_same_molecule()
    test_action_conditioned_reference_adds_exact_switch_with_shared_boundary()
    test_action_conditioned_reference_merges_new_row_for_existing_molecule()
    test_virtual_adamw_update_matches_real_optimizer_step()
    test_optimizer_parameter_groups_partition_action_gradient_ledger()
    test_inner_semantic_projection_cannot_erase_corrective_direction()
    test_gradient_cosine_is_bounded_and_marks_inactive_branches()
    test_legacy_90pct_boundary_is_reported_without_tightening_registered_gate()
    test_registered_formal_v3_configuration_fails_closed_on_drift()
    test_registered_v6_report_configuration_is_complete()
    test_registered_v6_cap_safe_schedule_is_exact_and_pre_model()
    test_dense_auxiliary_scale_preserves_sparse_mass_and_caps_dense_mass()
    print("[test_noise_corrected_direct_v3_trainer] PASS tests=26")


if __name__ == "__main__":
    main()
