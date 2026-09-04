from __future__ import annotations

import pandas as pd

from tasks.evaluate_kgmn_hidden_seed_recovery import score_arm


def _denominator() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "repeat": [0, 0, 0],
            "truth_inchikey1": ["A" * 14, "B" * 14, "C" * 14],
            "polarity_presence": ["positive", "negative", "negative+positive"],
        }
    )


def test_missing_predictions_stay_in_denominator_and_ties_fail_top1() -> None:
    predictions = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0],
            "truth_inchikey1": ["A" * 14, "A" * 14, "B" * 14, "B" * 14],
            "candidate_inchikey1": ["A" * 14, "X" * 14, "B" * 14, "Y" * 14],
            "candidate_score": [0.8, 0.8, 0.7, 0.6],
            "propagation_depth": [1, 1, 2, 1],
        }
    )
    scored = score_arm(_denominator(), predictions)
    assert scored["annotated"].tolist() == [True, True, False]
    assert scored["truth_recovered"].tolist() == [True, True, False]
    assert scored["truth_rank"].tolist()[:2] == [2.0, 1.0]
    assert scored["top1_correct"].tolist() == [False, True, False]
    assert scored["top3_correct"].tolist() == [True, True, False]


def test_duplicate_paths_use_strongest_identity_score_and_shallowest_tied_depth() -> None:
    predictions = pd.DataFrame(
        {
            "repeat": [0, 0, 0],
            "truth_inchikey1": ["A" * 14] * 3,
            "candidate_inchikey1": ["A" * 14, "A" * 14, "X" * 14],
            "candidate_score": [0.4, 0.9, 0.8],
            "propagation_depth": [1, 3, 1],
        }
    )
    scored = score_arm(_denominator().iloc[[0]].copy(), predictions)
    assert bool(scored.loc[0, "top1_correct"])
    assert scored.loc[0, "truth_rank"] == 1
    assert scored.loc[0, "truth_propagation_depth"] == 3
