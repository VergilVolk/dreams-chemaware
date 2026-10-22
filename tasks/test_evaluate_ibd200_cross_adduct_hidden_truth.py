import numpy as np

from evaluate_ibd200_cross_adduct_hidden_truth import rank_metrics


def test_rank_metrics_does_not_credit_arbitrary_top_tie():
    result = rank_metrics(np.asarray([0.5, 0.5, 0.1]), np.asarray(["T", "X", "Y"]), "T")
    assert result["truth_in_top_tie"]
    assert not result["unique_correct_top1"]
    assert result["optimistic_rank"] == 1
    assert result["conservative_rank"] == 2


def test_rank_metrics_unique_truth_top1():
    result = rank_metrics(np.asarray([0.8, 0.5]), np.asarray(["T", "X"]), "T")
    assert result["unique_correct_top1"]
    assert result["top_tie_size"] == 1
