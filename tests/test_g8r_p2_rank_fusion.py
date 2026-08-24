import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

from g8r_p2_rank_fusion_core import (  # noqa: E402
    FusionConfiguration,
    fusion_configuration_from_mapping,
    fuse_one_query,
    grouped_max,
    normalize_pair_features,
    strict_rank,
)


def test_json_configuration_is_normalized_to_hashable_tuple():
    configuration = fusion_configuration_from_mapping({
        "normalization": "absolute",
        "weights": [0.1, 0.0, 0.1, 0.8],
        "min_support": 1,
        "min_advantage": 0.0,
    })
    assert configuration == FusionConfiguration("absolute", (0.1, 0.0, 0.1, 0.8), 1, 0.0)
    assert isinstance(configuration.weights, tuple)
    assert hash(configuration)


def test_json_configuration_rejects_schema_drift():
    try:
        fusion_configuration_from_mapping({
            "normalization": "absolute",
            "weights": [0.1, 0.0, 0.1, 0.8],
            "min_support": 1,
            "min_advantage": 0.0,
            "unexpected": True,
        })
    except ValueError:
        pass
    else:
        raise AssertionError("configuration schema drift was accepted")


def test_grouped_max_keeps_spectrum_pair_coherence():
    pairs = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.6, 0.6]])
    ptr = np.asarray([0, 2, 3])
    fused, used, _ = fuse_one_query(
        pairs, pairs[:, 0], ptr, np.asarray([0.5, 0.5]), (1,), 0, 0.0,
    )
    # Candidate zero must be max(mean([.9,.1]), mean([.1,.9]))=.5,
    # not mean(max(feature-wise))=.9.
    assert used
    assert np.allclose(fused, [0.5, 0.6])


def test_gate_falls_back_when_raw_support_is_missing():
    pairs = np.asarray([[0.9, 0.2], [0.8, 1.0]])
    scores, used, support = fuse_one_query(
        pairs, pairs[:, 0], np.asarray([0, 1, 2]), np.asarray([0.2, 0.8]),
        (1,), 2, 0.0,
    )
    assert not used and support == 1
    assert np.allclose(scores, [0.9, 0.8])


def test_strict_rank_counts_ties_against_positive():
    rank, mrr, margin = strict_rank(np.asarray([0.7, 0.7, 0.2]))
    assert rank == 2 and mrr == 0.5 and margin == 0.0


def test_query_minmax_is_independent_between_queries():
    values = np.asarray([[1.0, 0.2], [0.0, 0.8], [0.5, 0.4], [0.5, 0.9]])
    normalized = normalize_pair_features(values, np.asarray([0, 2, 4]), "query_minmax")
    assert np.allclose(normalized[:2], [[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(normalized[2:, 0], [0.0, 0.0])
    assert np.allclose(normalized[2:, 1], [0.0, 1.0])


def test_grouped_max_rejects_empty_groups():
    try:
        grouped_max(np.asarray([1.0, 2.0]), np.asarray([0, 0, 2]))
    except ValueError:
        pass
    else:
        raise AssertionError("empty group was accepted")
