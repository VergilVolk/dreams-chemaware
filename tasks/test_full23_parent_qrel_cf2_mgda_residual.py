#!/usr/bin/env python
"""Synthetic contracts for the fixed Full23 parent-qrel CF2 candidate."""
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
)
from train_full23_parent_qrel_cf2_mgda_residual import (
    AVAILABILITY,
    CHEMICAL,
    EXPECTED_BASE_FEATURES,
    EXPECTED_FEATURES,
    EXPECTED_INTERACTIONS,
    QUERY_CONSTANT,
    REPRESENTATION,
    SPECTRAL,
    _product_name,
    cf2_panel,
)
from train_full23_query_relative_mgda import formula_oof_partition
from train_full23_multisource_additive import fit_normalization, pack


RAW_NAMES = (
    "noise_v1", "noise_v1_percentile",
    "weighted_spectral_entropy", "weighted_spectral_entropy_percentile",
    "p2b_noise_v1_frozen", "p2b_noise_v1_frozen_percentile",
    "noise_msg_fusion_stack_v1", "noise_msg_fusion_stack_v1_percentile",
    "chemaware_stage1_cosine", "chemaware_stage1_percentile",
    "chemaware_phasea_cosine", "chemaware_phasea_percentile",
    "noise_chemaware_fullmsg_cosine", "noise_chemaware_fullmsg_percentile",
    "reliability_top1_vote_fraction", "reliability_mean_percentile",
    "reliability_percentile_std", AVAILABILITY,
    "chemaware_v2_correct_minus_zero",
    "chemaware_v2_correct_minus_worst_null",
    "chemaware_v2_null_spread", "chemaware_v2_delta_percentile",
    "chemaware_v2_delta_top_gap",
)


def panel(name: str, source: str) -> PanelData:
    ptr = np.asarray([0, 2, 4], dtype=np.int64)
    first = "C6H12O6"
    first_fold = formula_fold(first, 5, 20261026)
    second = next(
        f"C{carbon}H{2 * carbon}O{carbon}"
        for carbon in range(7, 100)
        if formula_fold(f"C{carbon}H{2 * carbon}O{carbon}", 5, 20261026) != first_fold
    )
    scalar = np.arange(4 * len(RAW_NAMES), dtype=np.float32).reshape(4, len(RAW_NAMES))
    # Deliberately inject stale nonzero chemistry while unavailable.  CF2 must
    # still turn every chemical main effect and product into an exact zero.
    scalar[:, RAW_NAMES.index(AVAILABILITY)] = np.asarray([0, 0, 1, 1], np.float32)
    for name_ in CHEMICAL:
        scalar[:, RAW_NAMES.index(name_)] = np.asarray([3, -4, 2, -1], np.float32)
    return PanelData(
        name=name,
        graph=SimpleNamespace(),
        scalar=scalar,
        feature_names=RAW_NAMES,
        labels=np.asarray([1, 0, 0, 1], dtype=np.int8),
        query_ptr=ptr,
        formulas=np.asarray([first, second], dtype="U"),
        query_keys=np.asarray([f"{source}::q0", f"{source}::q1"], dtype="U"),
        near=np.asarray([False, True]),
        query_weights=np.asarray([0.25, 0.25], dtype=np.float64),
        scores={},
    )


def main() -> None:
    transformed = cf2_panel(panel("identity_disjoint", "enveda"))
    names = transformed.feature_names
    assert transformed.scalar.shape == (4, EXPECTED_FEATURES)
    assert len(names) == EXPECTED_FEATURES == 51
    assert sum("__x__" in name for name in names) == EXPECTED_INTERACTIONS == 32
    assert len(names) - EXPECTED_INTERACTIONS == EXPECTED_BASE_FEATURES == 19
    assert not set(QUERY_CONSTANT).intersection(names)

    expected_products = {
        _product_name(left, right)
        for left, right in (
            [(left, right) for left in REPRESENTATION for right in SPECTRAL]
            + [(left, right) for left in REPRESENTATION for right in CHEMICAL]
            + [(left, right) for left in SPECTRAL for right in CHEMICAL]
        )
    }
    actual_products = {name for name in names if "__x__" in name}
    assert actual_products == expected_products
    assert all(
        name.startswith("chemaware_v2_")
        for name in actual_products
        if any(chemical in name for chemical in CHEMICAL)
    )
    assert all(
        name.startswith("cf2__")
        for name in actual_products
        if not any(chemical in name for chemical in CHEMICAL)
    )
    assert not any("noise_msg_fusion_stack_v1" in name for name in actual_products)
    assert not any("percentile__x__" in name for name in actual_products)
    assert not any(AVAILABILITY in name for name in actual_products)

    unavailable = np.asarray([0, 1], dtype=np.int64)
    assert np.all(transformed.scalar[unavailable, names.index(AVAILABILITY)] == 0)
    for chemical in CHEMICAL:
        assert np.all(transformed.scalar[unavailable, names.index(chemical)] == 0)
        for product in actual_products:
            if chemical in product:
                assert np.all(transformed.scalar[unavailable, names.index(product)] == 0)

    model = ParentPlusLinearDelta(EXPECTED_FEATURES)
    features = torch.randn(3, EXPECTED_FEATURES)
    parent_scores = torch.tensor([2.0, -1.0, 0.5])
    assert torch.equal(model(features, parent_scores), parent_scores)
    assert torch.equal(model.delta(features), torch.zeros(3))

    enveda = {"identity_disjoint": panel("identity_disjoint", "enveda")}
    held_out = formula_fold(str(enveda["identity_disjoint"].formulas[0]), 5, 20261026)
    parent_train = outer_parent_training_occurrences(enveda, held_out, 5, 20261026)
    assert parent_train and all(query != 0 for _, query in parent_train)

    combined_panel = cf2_panel(panel("identity_disjoint", "enveda"))
    combined_panel.query_keys = np.asarray(["enveda::q0", "gnps::q1"], dtype="U")
    combined = {"identity_disjoint": combined_panel}
    all_occurrences = [("identity_disjoint", 0), ("identity_disjoint", 1)]
    train, valid = formula_oof_partition(
        combined, all_occurrences, held_out, 5, 20261026,
    )
    train_formula = {combined[name].formulas[query] for name, query in train}
    valid_formula = {combined[name].formulas[query] for name, query in valid}
    assert not train_formula.intersection(valid_formula)

    # Fit the real fold-local scaler on an unavailable-chemistry training
    # query.  After the exact production normalization path, every chemical
    # main effect and chemistry-bearing interaction must remain exactly zero.
    unavailable_fold = formula_fold(
        str(combined_panel.formulas[1]), 5, 20261026,
    )
    unavailable_train, _ = formula_oof_partition(
        combined, all_occurrences, unavailable_fold, 5, 20261026,
    )
    assert unavailable_train == [("identity_disjoint", 0)]
    fold_mean, fold_scale = fit_normalization(combined, unavailable_train)
    normalized, _, _, _ = pack(
        combined, unavailable_train, fold_mean, fold_scale,
    )
    chemistry_columns = [
        index for index, name in enumerate(names)
        if name == AVAILABILITY or name in CHEMICAL
        or ("__x__" in name and any(chemical in name for chemical in CHEMICAL))
    ]
    assert chemistry_columns
    assert np.all(normalized[:, chemistry_columns] == 0.0)

    # One optimizer step proves that the inherited parent+MGDA path accepts the
    # 51-dimensional fixed CF2 axis without changing its source objectives.
    fixed_parent = {"identity_disjoint": np.asarray([1.0, 0.0, -0.5, 0.5], np.float32)}
    args = SimpleNamespace(
        device="cpu", learning_rate=1e-3, weight_decay=1e-4,
        batch_queries=1, epochs=1, anchor_strength=1e-2,
    )
    fitted, history = fit_delta_mgda(
        combined, all_occurrences, fixed_parent,
        np.zeros(EXPECTED_FEATURES, np.float32),
        np.ones(EXPECTED_FEATURES, np.float32), args, 17,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["enveda_loss"])
    assert np.isfinite(history[0]["gnps_loss"])
    assert all(torch.all(torch.isfinite(value)) for value in fitted.parameters())
    print("PASS: Full23 parent-qrel CF2 MGDA residual synthetic contracts")


if __name__ == "__main__":
    main()
