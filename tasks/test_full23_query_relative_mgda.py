#!/usr/bin/env python
"""Small CPU contracts for query-relative Full23 MGDA training."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from train_enveda_unified_candidate_integrator import PanelData, WSE
from train_full23_query_relative_mgda import (
    EXPERT_FEATURE_PAIRS,
    RAW_EXPERT_FEATURES,
    formula_oof_partition,
    fit_mgda_model,
    query_block_zscore,
    query_relative_panel,
    two_task_mgda_coefficient,
)
from enveda_unified_candidate_integrator_core import formula_fold


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for raw, percentile in EXPERT_FEATURE_PAIRS:
        names.extend((raw, percentile))
    names.extend((
        "reliability_top1_vote_fraction",
        "reliability_mean_percentile",
        "reliability_percentile_std",
        "chemaware_v2_available",
        "chemaware_v2_correct_minus_zero",
        "chemaware_v2_correct_minus_worst_null",
        "chemaware_v2_null_spread",
        "chemaware_v2_delta_percentile",
        "chemaware_v2_delta_top_gap",
    ))
    return tuple(names)


def main() -> None:
    ptr = np.asarray([0, 3, 5, 6], dtype=np.int64)
    raw = np.asarray([1.0, 2.0, 3.0, 7.0, 7.0, 4.0], dtype=np.float32)
    transformed = query_block_zscore(raw, ptr)
    assert np.allclose(transformed[:3].mean(), 0.0, atol=1e-7)
    assert np.allclose(np.mean(np.square(transformed[:3])), 1.0, atol=1e-7)
    assert np.array_equal(transformed[3:], np.zeros(3, dtype=np.float32))

    names = _feature_names()
    assert len(names) == 23
    scalar = np.arange(6 * len(names), dtype=np.float32).reshape(6, len(names))
    # Make every expert informative in the first query and constant thereafter.
    for name in RAW_EXPERT_FEATURES:
        index = names.index(name)
        scalar[:, index] = raw + index
    panel = PanelData(
        name="identity_disjoint", graph=SimpleNamespace(), scalar=scalar,
        feature_names=names, labels=np.asarray([1, 0, 0, 1, 0, 1], dtype=np.int8),
        query_ptr=ptr, formulas=np.asarray(["C1", "C2", "C3"]),
        query_keys=np.asarray(["enveda::q1", "enveda::q2", "enveda::q3"]),
        near=np.zeros(3, dtype=bool), query_weights=np.ones(3), scores={},
    )
    relative = query_relative_panel(panel)
    assert relative.scalar.shape == scalar.shape
    assert WSE + "_query_zscore" in relative.feature_names
    for raw_name in RAW_EXPERT_FEATURES:
        index = names.index(raw_name)
        assert relative.feature_names[index] == raw_name + "_query_zscore"
        assert np.allclose(relative.scalar[:3, index].mean(), 0.0, atol=1e-7)
    untouched = [index for index, name in enumerate(names) if name not in RAW_EXPERT_FEATURES]
    assert np.array_equal(relative.scalar[:, untouched], scalar[:, untouched])

    first = (torch.tensor([1.0]),)
    second = (torch.tensor([-1.0]),)
    alpha, cosine, norm = two_task_mgda_coefficient(first, second)
    assert alpha == 0.5 and cosine == -1.0 and norm == 0.0
    alpha, _, _ = two_task_mgda_coefficient((torch.tensor([2.0]),), (torch.tensor([1.0]),))
    assert alpha == 0.0
    alpha, _, _ = two_task_mgda_coefficient(first, first)
    assert alpha == 0.5

    panels = {
        "identity_disjoint": relative,
        "formula_disjoint": PanelData(
            name="formula_disjoint", graph=SimpleNamespace(), scalar=relative.scalar.copy(),
            feature_names=relative.feature_names, labels=relative.labels.copy(), query_ptr=ptr,
            formulas=np.asarray(["C1", "C4", "C5"]),
            query_keys=np.asarray(["gnps::q1", "gnps::q4", "gnps::q5"]),
            near=np.zeros(3, dtype=bool), query_weights=np.ones(3), scores={},
        ),
    }
    occurrences = [(name, query) for name in panels for query in range(3)]
    observed_folds = {
        formula_fold(str(formula), 5, 20261026)
        for panel_value in panels.values() for formula in panel_value.formulas
    }
    for fold in observed_folds:
        train, valid = formula_oof_partition(panels, occurrences, fold, 5, 20261026)
        train_formulas = {str(panels[name].formulas[query]) for name, query in train}
        valid_formulas = {str(panels[name].formulas[query]) for name, query in valid}
        assert not (train_formulas & valid_formulas)

    # Exercise one real optimizer step with separate source objectives.  The
    # model begins at WSE but both source gradients must reach its parameters.
    train_ptr = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    train_scalar = np.zeros((8, len(relative.feature_names)), dtype=np.float32)
    wse_index = relative.feature_names.index(WSE + "_query_zscore")
    train_scalar[:, wse_index] = np.tile([1.0, -1.0], 4)
    training_panel = PanelData(
        name="identity_disjoint", graph=SimpleNamespace(), scalar=train_scalar,
        feature_names=relative.feature_names,
        labels=np.tile(np.asarray([1, 0], dtype=np.int8), 4), query_ptr=train_ptr,
        formulas=np.asarray(["C1", "C2", "C3", "C4"]),
        query_keys=np.asarray([
            "enveda::e1", "enveda::e2", "gnps::g1", "gnps::g2",
        ]),
        near=np.zeros(4, dtype=bool), query_weights=np.full(4, 0.25), scores={},
    )
    args = SimpleNamespace(
        device="cpu", learning_rate=1e-3, weight_decay=1e-4,
        anchor_strength=1e-2, epochs=1, batch_queries=2,
    )
    model, history = fit_mgda_model(
        {"identity_disjoint": training_panel},
        [("identity_disjoint", query) for query in range(4)],
        np.zeros(len(relative.feature_names), dtype=np.float32),
        np.ones(len(relative.feature_names), dtype=np.float32),
        args, 7,
    )
    assert len(history) == 1 and np.isfinite(history[0]["gradient_cosine_mean"])
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    print("PASS: Full23 query-relative MGDA contracts")


if __name__ == "__main__":
    main()
