#!/usr/bin/env python
"""Scientific contracts for the WSE-anchored two-arm residual experiment."""
from __future__ import annotations

import numpy as np
import torch

from conditional_null_energy_core import AnchoredEvidenceEnergy
from train_wse_conditional_residual import (
    CHEMICAL_FEATURE_NAMES,
    OLD_MODULES,
    SPECTRAL_FEATURE_NAMES,
    build_chemical_features,
    fit_scaler,
    query_score_features,
    reconstruct_old_scores,
    training_residual_bound,
    transform_features,
)
from train_unified_candidate_energy_crossfit import metric_summary, prepare


def test_raw_scores_and_competition_state_are_preserved() -> None:
    values = np.asarray([0.2, 0.8, 0.5, 0.5, 0.1], dtype=np.float32)
    ptr = np.asarray([0, 2, 5], dtype=np.int64)
    features = query_score_features(values, ptr)
    assert np.array_equal(features[:, 0], values)
    assert np.allclose(features[:2, 1], [-0.6, 0.0])
    assert np.allclose(features[:2, 2], 0.6)
    assert np.allclose(features[2:, 2], 0.0)  # exact top tie is retained


def test_chemical_features_cancel_shared_nuisance_and_mark_missing() -> None:
    nuisance = np.asarray([10.0, -4.0, 2.0], dtype=np.float32)
    residual = {
        "correct": np.asarray([0.5, 0.2, 9.0], dtype=np.float32),
        "zero": np.asarray([0.0, 0.1, 7.0], dtype=np.float32),
        "reverse": np.asarray([-0.2, 0.3, 8.0], dtype=np.float32),
        "rotated": np.asarray([0.1, -0.1, 6.0], dtype=np.float32),
    }
    chem = {
        "availability": np.asarray([True, True, False]),
        "utility_correct": nuisance + residual["correct"],
        "utility_zero": nuisance + residual["zero"],
        "utility_reversed": nuisance + residual["reverse"],
        "utility_candidate_mismatch": nuisance + residual["rotated"],
    }
    actual, nulls, available = build_chemical_features(chem)
    residual_only = dict(chem)
    residual_only.update({
        "utility_correct": residual["correct"],
        "utility_zero": residual["zero"],
        "utility_reversed": residual["reverse"],
        "utility_candidate_mismatch": residual["rotated"],
    })
    expected, _, _ = build_chemical_features(residual_only)
    assert np.allclose(actual, expected, atol=1e-6)
    assert np.array_equal(nulls[0], np.zeros_like(nulls[0]))
    assert np.array_equal(actual[~available], np.zeros((1, len(CHEMICAL_FEATURE_NAMES))))


def test_training_scaler_preserves_exact_chemical_zero() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(8, len(SPECTRAL_FEATURE_NAMES) + len(CHEMICAL_FEATURE_NAMES))).astype(np.float32)
    features[4:, len(SPECTRAL_FEATURE_NAMES):] = 0.0
    available = np.asarray([True, True, True, True, False, False, False, False])
    train = np.asarray([0, 1, 2, 4, 5], dtype=np.int64)
    mean, scale = fit_scaler(features, train, available)
    assert np.array_equal(mean[len(SPECTRAL_FEATURE_NAMES):], np.zeros(len(CHEMICAL_FEATURE_NAMES)))
    zero = np.zeros((3, features.shape[1]), dtype=np.float32)
    zero[:, :len(SPECTRAL_FEATURE_NAMES)] = mean[:len(SPECTRAL_FEATURE_NAMES)]
    transformed = transform_features(zero, mean, scale)
    assert np.array_equal(
        transformed[:, len(SPECTRAL_FEATURE_NAMES):],
        np.zeros((3, len(CHEMICAL_FEATURE_NAMES)), dtype=np.float32),
    )


def test_residual_bound_reads_training_queries_only() -> None:
    ptr = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    base = np.asarray([0.0, 0.1, 0.0, 0.2, 0.0, 10.0, 0.0, 20.0])
    queries = np.asarray([0, 1], dtype=np.int64)
    first, _ = training_residual_bound(base, ptr, queries)
    changed = base.copy()
    changed[4:] *= 1_000_000
    second, _ = training_residual_bound(changed, ptr, queries)
    assert first == second
    assert 0.1 <= first <= 0.2


def test_joint_chemical_zero_is_exact_noop() -> None:
    feature_count = len(SPECTRAL_FEATURE_NAMES) + len(CHEMICAL_FEATURE_NAMES)
    model = AnchoredEvidenceEnergy(
        feature_count, mode="joint",
        spectral_feature_count=len(SPECTRAL_FEATURE_NAMES),
        hidden=4, residual_bound=0.2,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.5, 0.5)
    features = torch.randn(6, feature_count)
    features[:, len(SPECTRAL_FEATURE_NAMES):] = 0
    spectral_only = 0.5 * model.residual_bound * torch.tanh(
        model.spectral_network(features[:, :len(SPECTRAL_FEATURE_NAMES)]).squeeze(-1)
    )
    assert torch.allclose(model.residual(features), spectral_only, atol=1e-7, rtol=0)


def test_old_four_module_scores_are_replayed_without_retraining() -> None:
    ptr = np.asarray([0, 3, 6, 9], dtype=np.int64)
    bundle = {
        "module_names": np.asarray(OLD_MODULES),
        "scores": np.asarray([
            [0.8, 0.3, 0.1, 0.1, 0.5, 0.2, 0.3, 0.2, 0.9],
            [0.7, 0.4, 0.2, 0.2, 0.4, 0.3, 0.4, 0.3, 0.8],
            [0.6, 0.2, 0.5, 0.4, 0.3, 0.2, 0.5, 0.1, 0.7],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32),
        "availability": np.ones((4, 9), dtype=np.float32),
        "query_ptr": ptr,
        "labels": np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8),
        "group_ids": np.asarray(["F1", "F2", "F3"]),
        "near_query": np.asarray([False, True, True]),
    }
    selected = prepare(bundle)
    coefficients = {
        OLD_MODULES[0]: 0.1, OLD_MODULES[1]: 0.0,
        OLD_MODULES[2]: 0.2, OLD_MODULES[3]: 0.3,
    }
    base = selected["evidence"][1]
    expected = base.astype(np.float64, copy=True)
    for index, name in enumerate(OLD_MODULES):
        expected += coefficients[name] * (selected["evidence"][index] - base)
    metrics, _ = metric_summary(expected, selected)
    report = {
        "optimization": {"seed": 17},
        "formula_crossfit_folds": 2,
        "arms": {
            "fixed_four": {
                "modules": list(OLD_MODULES),
                "folds": [
                    {"fold": 0, "residual_coefficients": coefficients},
                    {"fold": 1, "residual_coefficients": coefficients},
                ],
                "panels": {"identity": {"candidate": metrics}},
            }
        },
    }
    observed, arm = reconstruct_old_scores(bundle, report, "identity")
    assert arm == "fixed_four"
    assert np.allclose(observed, expected, atol=1e-6)


if __name__ == "__main__":
    test_raw_scores_and_competition_state_are_preserved()
    test_chemical_features_cancel_shared_nuisance_and_mark_missing()
    test_training_scaler_preserves_exact_chemical_zero()
    test_residual_bound_reads_training_queries_only()
    test_joint_chemical_zero_is_exact_noop()
    test_old_four_module_scores_are_replayed_without_retraining()
    print("PASS: WSE conditional-residual scientific contracts")
