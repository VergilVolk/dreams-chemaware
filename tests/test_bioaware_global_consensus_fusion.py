from __future__ import annotations

import pandas as pd

from tasks.develop_bioaware_global_consensus_fusion import (
    PEER_FEATURES,
    add_leave_query_out_peer_features,
)
from tasks.develop_bioaware_rank_consensus_fusion import add_family_features


def test_peer_features_exclude_current_query() -> None:
    rows = []
    query_scores = {"q1": {"a": 1.0, "b": 0.0}, "q2": {"a": 0.5, "b": 1.0}}
    for query, scores in query_scores.items():
        for candidate, score in scores.items():
            rows.append({
                "query_id": query,
                "truth_formula": "C5H10O5",
                "candidate_id": candidate,
                "decoder_score": score,
                "rule_jaccard_idf": score,
                "sparse_rule_overlap": score,
                "known_edge_best_bottleneck": score,
                "predicted_edge_best_bottleneck": score,
                "smn_best_bottleneck": score,
                "rt_score": score,
            })
    result = add_leave_query_out_peer_features(add_family_features(pd.DataFrame(rows)))
    q1a = result[(result.query_id == "q1") & (result.candidate_id == "a")].iloc[0]
    q2a = result[(result.query_id == "q2") & (result.candidate_id == "a")].iloc[0]
    assert q1a["peer_family_decoder"] == 0.0
    assert q2a["peer_family_decoder"] == 1.0
    assert not result[PEER_FEATURES].isna().any().any()
