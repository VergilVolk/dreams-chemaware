from __future__ import annotations

import pandas as pd

from tasks.develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES,
    add_family_features,
    apply_gate,
    fit_family_weights,
    score_queries,
)


def test_family_consensus_can_rescue_without_duplicate_rule_votes() -> None:
    rows = []
    for index in range(30):
        query = f"q{index}"
        formula = f"C{index + 5}H{index + 8}"
        for candidate, truth, spectral, evidence in [
            ("truth", "truth", 0.49, 1.0),
            ("wrong", "truth", 0.50, 0.0),
        ]:
            rows.append({
                "query_id": query,
                "candidate_id": candidate,
                "truth_candidate_id": truth,
                "truth_formula": formula,
                "spectral_score": spectral,
                "decoder_score": evidence,
                "rule_jaccard_idf": evidence,
                "sparse_rule_overlap": evidence,
                "known_edge_best_bottleneck": evidence,
                "predicted_edge_best_bottleneck": evidence,
                "smn_best_bottleneck": evidence,
                "rt_score": evidence,
            })
    frame = add_family_features(pd.DataFrame(rows))
    assert len(FAMILY_FEATURES) == 6
    weights = fit_family_weights(frame, temperature=0.1, l2=0.05, maximum_weight=1.0)
    predictions = score_queries(frame, weights)
    result = apply_gate(predictions, (0.025, 0.0, 3))
    assert int(result["corrected"].sum()) == 30
    assert int(result["introduced"].sum()) == 0
