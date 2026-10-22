from __future__ import annotations

import numpy as np

from build_ibd200_plate3_algorithm_application_slice import (
    deterministic_stratified_folds,
    wrong_context_map,
)


def test_stratified_folds_and_wrong_context_preserve_fixed_margins() -> None:
    sample_ids = [f"sample_{index}" for index in range(10)]
    labels = ["IBD"] * 6 + ["healthy"] * 4
    folds = deterministic_stratified_folds(sample_ids, labels)
    assert np.array_equal(folds, deterministic_stratified_folds(sample_ids, labels))
    for label in {"IBD", "healthy"}:
        selected = folds[np.asarray(labels) == label]
        assert abs(int(np.sum(selected == 0)) - int(np.sum(selected == 1))) <= 1

    entity_ids = np.asarray(["a", "b", "c", "d"], dtype=str)
    detected = np.asarray([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    values = detected.astype(float) + np.arange(4, dtype=float)[None, :] / 10
    train = np.ones(len(values), dtype=bool)
    mapping = wrong_context_map(entity_ids, values, detected, train, fold=0).set_index(
        "recipient_entity_id"
    )
    lookup = {entity: index for index, entity in enumerate(entity_ids)}
    donor = np.asarray([lookup[value] for value in mapping.loc[entity_ids, "donor_entity_id"]])
    assert np.array_equal(detected.sum(axis=1), detected[:, donor].sum(axis=1))
    assert np.array_equal(detected.sum(axis=0), detected[:, donor].sum(axis=0))
    assert not mapping["fixed_point"].any()
