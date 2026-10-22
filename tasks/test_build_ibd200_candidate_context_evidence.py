import numpy as np

from build_ibd200_candidate_context_evidence import (
    aggregate_anchor_evidence,
    pair_evidence,
    rank_average,
)


def test_rank_average_handles_ties():
    ranks = rank_average(np.asarray([4.0, 1.0, 1.0, 7.0]))
    np.testing.assert_allclose(ranks, [2.0, 0.5, 0.5, 3.0])


def test_pair_evidence_keeps_signed_context():
    detected = np.ones(6, dtype=bool)
    positive = pair_evidence(
        np.arange(6.0), detected, np.arange(6.0), detected, minimum_joint=5,
    )
    negative = pair_evidence(
        np.arange(6.0), detected, np.arange(5.0, -1.0, -1.0), detected,
        minimum_joint=5,
    )
    assert positive == (1.0, 1.0, 1.0)
    assert negative == (1.0, -1.0, -1.0)


def test_aggregate_uses_only_training_samples_and_median_anchor():
    values = np.asarray([
        [0.0, 0.0, 5.0],
        [1.0, 1.0, 4.0],
        [2.0, 2.0, 3.0],
        [3.0, 3.0, 2.0],
        [4.0, 4.0, 1.0],
        [5.0, 5.0, 0.0],
        [100.0, -100.0, 100.0],
    ])
    detected = np.ones_like(values, dtype=bool)
    training = np.asarray([True] * 6 + [False])
    result = aggregate_anchor_evidence(
        0, [1, 2], values, detected, training, minimum_joint=5,
    )
    assert result is not None
    jaccard, rho, support, anchors = result
    assert jaccard == 1.0
    assert abs(rho) < 1e-12
    assert abs(support) < 1e-12
    assert anchors == 2
