#!/usr/bin/env python
"""CPU contract tests for the conditional-null candidate energy core."""
from __future__ import annotations

import numpy as np
import torch

from conditional_null_energy_core import (
    FEATURE_NAMES,
    AdditiveBoundedResidualEnergy,
    AnchoredEvidenceEnergy,
    build_candidate_features,
    conditional_null_loss,
    deterministic_keyed_derangements,
    percentile_by_query,
    strict_top1,
)


def fixture():
    query_ptr = np.asarray([0, 3, 6], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int8)
    v1 = np.asarray([0.2, 0.8, 0.5, 0.7, 0.6, 0.1], dtype=np.float64)
    p2b = np.asarray([0.3, 0.9, 0.4, 0.65, 0.8, 0.2], dtype=np.float64)
    neutral = np.asarray([0.1, 0.7, 0.4, 0.8, 0.6, 0.2], dtype=np.float64)
    chem = np.asarray([0.0, 0.8, 0.2, 0.0, 0.4, 0.1], dtype=np.float64)
    nulls = np.asarray([
        [0.0, 0.2, 0.2, 0.0, 0.2, 0.2],
        [0.0, -0.1, 0.1, 0.0, 0.5, 0.0],
        [0.2, 0.1, 0.8, 0.2, 0.7, 0.3],
    ], dtype=np.float64)
    candidate_keys = np.asarray(["A", "B", "C", "D", "E", "F"])
    query_keys = np.asarray(["q0", "q1"])
    return query_ptr, labels, v1, p2b, neutral, chem, nulls, candidate_keys, query_keys


def test_percentile_ties_and_keyed_derangement_invariance():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ptr = np.asarray([0, 3, 6])
    keys = np.asarray(["z", "a", "m", "u", "b", "r"])
    query_keys = np.asarray(["q0", "q1"])
    percentile = percentile_by_query(np.asarray([1, 1, 0, 4, 2, 3]), ptr)
    assert np.allclose(percentile[:3], [0.75, 0.75, 0.0])
    deranged = deterministic_keyed_derangements(values, keys, query_keys, ptr)
    assert deranged.shape == (3, 6)
    for arm in deranged:
        for left, right in zip(ptr[:-1], ptr[1:]):
            assert np.array_equal(np.sort(arm[left:right]), np.sort(values[left:right]))
            assert np.all(arm[left:right] != values[left:right])
    permutation = np.asarray([2, 0, 1, 5, 3, 4])
    reordered = deterministic_keyed_derangements(
        values[permutation], keys[permutation], query_keys, ptr,
    )
    restored = np.empty_like(reordered)
    restored[:, permutation] = reordered
    assert np.array_equal(restored, deranged)


def test_centered_chemical_features_and_input_immutability():
    ptr, _, v1, p2b, neutral, chem, nulls, _, _ = fixture()
    chem_before = chem.copy()
    actual = build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=chem, chem_references=nulls, query_ptr=ptr,
    )
    assert actual.shape == (6, len(FEATURE_NAMES))
    assert np.array_equal(chem, chem_before)
    equal = build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=chem, chem_references=np.stack([chem, chem, chem]), query_ptr=ptr,
    )
    assert np.all(equal[:, 3:] == 0)


def test_zero_initialization_and_listwise_null_gradients():
    torch.manual_seed(7)
    ptr, labels, v1, p2b, neutral, chem, nulls, _, _ = fixture()
    actual = build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=chem, chem_references=nulls, query_ptr=ptr,
    )
    null_rows = [build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=nulls[arm], chem_references=nulls, query_ptr=ptr,
    ) for arm in range(3)]
    model = AnchoredEvidenceEnergy(len(FEATURE_NAMES), mode="joint", hidden=8)
    base = torch.tensor(percentile_by_query(v1, ptr), dtype=torch.float32)
    actual_tensor = torch.tensor(actual)
    null_tensor = torch.tensor(np.stack(null_rows))
    with torch.no_grad():
        assert torch.allclose(model(base, actual_tensor), base)
    breakdown = conditional_null_loss(
        model=model, base_score=base, actual_features=actual_tensor,
        null_features=null_tensor, labels=torch.tensor(labels), query_ptr=ptr,
    )
    breakdown.total.backward()
    assert float(model.spectral_network[-1].weight.grad.abs().sum()) > 0
    assert float(model.chemical_network[-1].weight.grad.abs().sum()) > 0


def test_chemical_zero_is_exact_noop_after_training():
    torch.manual_seed(11)
    chem_only = AnchoredEvidenceEnergy(len(FEATURE_NAMES), mode="chem_only", hidden=6)
    optimizer = torch.optim.Adam(chem_only.parameters(), lr=0.1)
    arbitrary = torch.randn(8, len(FEATURE_NAMES))
    for _ in range(4):
        optimizer.zero_grad()
        loss = -chem_only.residual(arbitrary).mean()
        loss.backward()
        optimizer.step()
    zero_chem = arbitrary.clone()
    zero_chem[:, 3:] = 0
    assert torch.equal(chem_only.residual(zero_chem), torch.zeros(8))


def test_additive_comparator_has_no_cross_modal_marginal():
    torch.manual_seed(13)
    model = AdditiveBoundedResidualEnergy(len(FEATURE_NAMES), hidden=7)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    base = torch.zeros(1)
    first = torch.randn(1, len(FEATURE_NAMES))
    second = first.clone(); second[:, :3] += 0.5
    alternate_first = first.clone(); alternate_first[:, 3:] += 1.7
    alternate_second = second.clone(); alternate_second[:, 3:] += 1.7
    effect_a = model(base, second) - model(base, first)
    effect_b = model(base, alternate_second) - model(base, alternate_first)
    assert torch.allclose(effect_a, effect_b, atol=1e-7, rtol=1e-6)


def test_nonbinary_labels_rejected_and_strict_tie_fails():
    ptr, labels, v1, p2b, neutral, chem, nulls, _, _ = fixture()
    features = build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=chem, chem_references=nulls, query_ptr=ptr,
    )
    invalid = labels.copy(); invalid[0] = 2
    model = AnchoredEvidenceEnergy(len(FEATURE_NAMES), mode="joint", hidden=4)
    try:
        conditional_null_loss(
            model=model,
            base_score=torch.tensor(percentile_by_query(v1, ptr), dtype=torch.float32),
            actual_features=torch.tensor(features),
            null_features=torch.tensor(np.stack([features, features])),
            labels=torch.tensor(invalid), query_ptr=ptr,
        )
    except ValueError as error:
        assert "binary" in str(error)
    else:
        raise AssertionError("non-binary labels were accepted")
    tie_ptr = np.asarray([0, 2, 4])
    tie_labels = np.asarray([1, 0, 1, 0])
    assert np.array_equal(
        strict_top1(np.asarray([1.0, 1.0, 0.9, 0.1]), tie_labels, tie_ptr), [0, 1],
    )


def main():
    tests = [
        test_percentile_ties_and_keyed_derangement_invariance,
        test_centered_chemical_features_and_input_immutability,
        test_zero_initialization_and_listwise_null_gradients,
        test_chemical_zero_is_exact_noop_after_training,
        test_additive_comparator_has_no_cross_modal_marginal,
        test_nonbinary_labels_rejected_and_strict_tie_fails,
    ]
    for test in tests:
        test()
    print(f"[test_conditional_null_energy] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
