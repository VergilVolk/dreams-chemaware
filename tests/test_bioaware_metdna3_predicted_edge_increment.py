from __future__ import annotations

import pandas as pd

from tasks.summarize_bioaware_metdna3_predicted_edge_increment import decide


def test_predicted_edge_decision_requires_four_consistent_votes() -> None:
    evidence = []
    for fold in range(7):
        for candidate, score in (("truth", 0.8 if fold < 4 else 0.2), ("wrong", 0.5)):
            evidence.append({
                "maximum_depth": 3, "fold": fold, "query_id": "q",
                "candidate_id": candidate, "truth_candidate_id": "truth",
                "truth_formula": "C2H4O2", "best_bottleneck": score,
            })
    _, queries = decide(pd.DataFrame(evidence), pd.Series({"q": "wrong"}), 3)
    assert bool(queries.iloc[0].corrected)
    assert queries.iloc[0].winning_vote_count == 4
