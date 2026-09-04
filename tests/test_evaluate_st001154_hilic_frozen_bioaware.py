import pandas as pd

from tasks.evaluate_st001154_hilic_frozen_bioaware import unique_top


def test_unique_top_counts_ties_against_truth():
    frame = pd.DataFrame(
        {"candidate_id": ["B", "A", "C"], "score": [1.0, 1.0, 0.5]}
    )
    candidate, unique = unique_top(frame, "score")
    assert candidate == "A"
    assert not unique
