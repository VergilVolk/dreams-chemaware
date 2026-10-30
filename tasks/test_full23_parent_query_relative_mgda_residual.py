#!/usr/bin/env python
"""Synthetic contracts for strict Full23-parent query-relative MGDA residual."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import formula_fold
from train_enveda_unified_candidate_integrator import PanelData
from train_full23_parent_query_relative_mgda_residual import (
    ParentPlusLinearDelta,
    fit_delta_mgda,
    outer_parent_training_occurrences,
    pack_parent,
)
from train_full23_query_relative_mgda import formula_oof_partition


def panel(name: str, source: str) -> PanelData:
    ptr = np.asarray([0, 2, 4], dtype=np.int64)
    first = "C6H12O6"
    first_fold = formula_fold(first, 5, 20261026)
    second = next(
        f"C{carbon}H{2 * carbon}O{carbon}"
        for carbon in range(7, 100)
        if formula_fold(f"C{carbon}H{2 * carbon}O{carbon}", 5, 20261026) != first_fold
    )
    return PanelData(
        name=name,
        graph=SimpleNamespace(),
        scalar=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        feature_names=("feature",),
        labels=np.asarray([1, 0, 0, 1], dtype=np.int8),
        query_ptr=ptr,
        formulas=np.asarray([first, second], dtype="U"),
        query_keys=np.asarray([f"{source}::q0", f"{source}::q1"], dtype="U"),
        near=np.asarray([False, True]),
        query_weights=np.asarray([0.25, 0.25], dtype=np.float64),
        scores={},
    )


def main() -> None:
    model = ParentPlusLinearDelta(1)
    features = torch.tensor([[0.0], [2.0], [-1.0]], dtype=torch.float32)
    parent = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float32)
    assert torch.equal(model(features, parent), parent)
    assert torch.equal(model.delta(features), torch.zeros(3))

    enveda = {"identity_disjoint": panel("identity_disjoint", "enveda")}
    held_out = formula_fold(str(enveda["identity_disjoint"].formulas[0]), 5, 20261026)
    selected = outer_parent_training_occurrences(enveda, held_out, 5, 20261026)
    assert selected
    # The helper is Enveda-only by construction and every selected formula is
    # outside the held-out formula fold (it raises otherwise).
    assert all(name == "identity_disjoint" for name, _ in selected)
    assert all(query != 0 for _, query in selected)

    combined = {
        "identity_disjoint": PanelData(
            **{
                **enveda["identity_disjoint"].__dict__,
                "query_keys": np.asarray(["enveda::q0", "gnps::q1"], dtype="U"),
            }
        )
    }
    all_occ = [("identity_disjoint", 0), ("identity_disjoint", 1)]
    train, valid = formula_oof_partition(combined, all_occ, held_out, 5, 20261026)
    train_formula = {combined[n].formulas[q] for n, q in train}
    valid_formula = {combined[n].formulas[q] for n, q in valid}
    assert not train_formula.intersection(valid_formula)

    parent_scores = {"identity_disjoint": np.asarray([2.0, 1.0, -1.0, 0.5], np.float32)}
    mean = np.asarray([0.0], np.float32)
    scale = np.asarray([1.0], np.float32)
    _, labels, ptr, weights, fixed = pack_parent(
        enveda, [("identity_disjoint", 1)], mean, scale, parent_scores,
    )
    assert np.array_equal(labels, np.asarray([0, 1], np.int8))
    assert np.array_equal(ptr, np.asarray([0, 2], np.int64))
    assert np.array_equal(fixed, np.asarray([-1.0, 0.5], np.float32))
    assert weights.shape == (1,)

    args = SimpleNamespace(
        device="cpu", learning_rate=1e-3, weight_decay=1e-4,
        batch_queries=1, epochs=1, anchor_strength=1e-2,
    )
    fitted, history = fit_delta_mgda(
        combined, all_occ, parent_scores, mean, scale, args, 17,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["enveda_loss"])
    assert np.isfinite(history[0]["gnps_loss"])
    assert all(torch.all(torch.isfinite(value)) for value in fitted.parameters())
    print("PASS: Full23 parent query-relative MGDA residual synthetic contracts")


if __name__ == "__main__":
    main()
