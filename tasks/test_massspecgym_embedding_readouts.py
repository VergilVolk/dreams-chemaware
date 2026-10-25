#!/usr/bin/env python
"""Dependency-light contracts for the frozen embedding readout mainline."""
from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from massspecgym_embedding_readout_core import (
    ARMS, ARM_PAIR, ARM_PROJECTED, ARM_SCALAR, FULL_INTEGRATION_ARMS,
    FrozenEmbeddingReadout, ReadoutSettings, build_independent_scalar_features,
    fit_scalar_normalization, hard_competitor_matrix, masked_percentile_by_query,
    masked_query_gap_column, padded_candidate_matrix, percentile_by_query,
    query_gap_column, select_full_integration_arm, strict_ranks,
)


def main() -> None:
    query_ptr = np.asarray([0, 3, 5], dtype=np.int64)
    values = np.asarray([0.8, 0.8, 0.2, 0.5, 0.4])
    percentile = percentile_by_query(values, query_ptr)
    assert np.allclose(percentile[:3], [0.75, 0.75, 0.0])
    assert np.allclose(query_gap_column(values, query_ptr), [0.0, 0.0, 0.0, 0.1, 0.1])
    available = np.asarray([0, 1, 1, 0, 1], dtype=bool)
    assert np.allclose(masked_percentile_by_query(values, available, query_ptr), [0, 1, 0, 0, 0])
    assert np.allclose(masked_query_gap_column(values, available, query_ptr), [0, .6, .6, 0, 0])
    full, mask = padded_candidate_matrix(query_ptr)
    assert full.tolist() == [[0, 1, 2], [3, 4, -1]]
    assert mask.sum() == 5
    hard, hard_mask = hard_competitor_matrix(
        query_ptr,
        [np.asarray([0.1, 0.9, 0.2, 0.3, 0.8]),
         np.asarray([0.2, 0.1, 0.95, 0.7, 0.2])],
        maximum_negatives=2,
    )
    assert hard[0, 0] == 0 and set(hard[0, 1:3]) == {1, 2}
    assert hard[1, 0] == 3 and hard_mask[1].sum() == 2
    assert strict_ranks(np.asarray([0.8, 0.8, 0.2, 0.5, 0.4]), query_ptr).tolist() == [2, 1]

    graph = SimpleNamespace(
        query_ptr=query_ptr,
        molecule_ptr=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        molecule_label=np.asarray([1, 0, 0, 1, 0], dtype=np.int8),
    )
    evidence = {
        "v1_cosine": values,
        "weighted_entropy": values[::-1],
        "p2b_fused": np.asarray([0.7, 0.1, 0.3, 0.4, 0.2]),
    }
    features, names = build_independent_scalar_features(
        evidence, graph,
        {"noise_v1": values, "chemaware": values * 0.9}, "noise_v1",
        {
            "chem_valid": np.asarray([0, 1, 1, 0, 1], dtype=bool),
            "candidate_utility_correct": np.asarray([0, .7, .4, 0, .5]),
            "candidate_utility_zero": np.asarray([0, .2, .3, 0, .2]),
            "candidate_utility_reversed": np.asarray([0, .1, .35, 0, .1]),
            "candidate_utility_rotated": np.asarray([0, .3, .2, 0, .3]),
        },
    )
    assert features.shape == (5, 24)
    assert "sqrt_cosine" not in names and "neutral_loss_sqrt_cosine" not in names
    assert names.count("p2b_noise_v1_frozen") == 1
    assert names.count("chemaware_v2_available") == 1
    unavailable = ~np.asarray([0, 1, 1, 0, 1], dtype=bool)
    chem_columns = [i for i, name in enumerate(names) if name.startswith("chemaware_v2_")]
    assert np.all(features[np.ix_(unavailable, chem_columns)] == 0)
    mean, scale = fit_scalar_normalization(features, names, np.arange(len(features)))
    normalized = (features - mean) / scale
    assert np.all(normalized[np.ix_(unavailable, chem_columns)] == 0)

    settings = ReadoutSettings(
        scalar_features=4, encoder_names=("noise_v1", "fullmsg"),
        embedding_dimension=8, projection_dimension=3,
        scalar_width=5, scalar_control_width=7,
    )
    scalar = torch.randn(5, 4)
    query_rows = torch.tensor([0, 0, 1, 1, 1])
    references = torch.tensor([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    banks = {name: torch.nn.functional.normalize(torch.randn(8, 8), dim=1)
             for name in settings.encoder_names}
    counts = []
    for arm in ARMS:
        model = FrozenEmbeddingReadout(arm, settings)
        score = model(scalar, query_rows, references, banks)
        assert score.shape == (5,) and torch.isfinite(score).all()
        counts.append(model.trainable_parameters)
    assert all(count > 0 for count in counts)
    # A stronger scalar control must not be mislabeled as the complete system.
    selected = select_full_integration_arm({
        ARM_SCALAR: {"recall1": 0.99, "mrr": 0.99},
        ARM_PROJECTED: {"recall1": 0.80, "mrr": 0.90},
        ARM_PAIR: {"recall1": 0.81, "mrr": 0.82},
    })
    assert selected == ARM_PAIR and selected in FULL_INTEGRATION_ARMS
    assert ARM_SCALAR not in FULL_INTEGRATION_ARMS
    for invalid in (
        {ARM_PROJECTED: {"recall1": .8, "mrr": .8}},
        {
            ARM_PROJECTED: {"recall1": np.nan, "mrr": .8},
            ARM_PAIR: {"recall1": .8, "mrr": .8},
        },
    ):
        try:
            select_full_integration_arm(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid full-integration metrics were accepted")
    print("PASS: MassSpecGym frozen embedding readout contracts")


if __name__ == "__main__":
    main()
