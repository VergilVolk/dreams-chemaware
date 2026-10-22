import numpy as np

from evaluate_ibd200_cross_adduct_rank_stability import ranking_stability


def test_rank_stability_credits_overlap_without_arbitrary_tie_break():
    result = ranking_stability(
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([0.5, 0.4, 0.0]),
        np.asarray(["A", "B", "C"]),
    )
    assert result["top_tie_overlap"]
    assert result["top_tie_jaccard"] == 0.5
    assert not result["unique_top1_comparable"]


def test_rank_stability_unique_concordance():
    result = ranking_stability(
        np.asarray([1.0, 0.5, 0.0]),
        np.asarray([0.8, 0.4, 0.1]),
        np.asarray(["A", "B", "C"]),
    )
    assert result["unique_top1_comparable"]
    assert result["unique_top1_concordant"]
    assert result["spearman_rank_correlation"] == 1.0
