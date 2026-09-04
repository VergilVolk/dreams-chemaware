from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.develop_bioaware_risk_controlled_fusion import (
    FEATURES,
    apply_gate,
    fit_residual_weights,
    score_queries,
)


def synthetic_ledger() -> pd.DataFrame:
    rows = []
    for query in range(8):
        truth = f"t{query}"
        for candidate, spectral, evidence in [
            (truth, 0.50, 1.0),
            (f"w{query}", 0.55, 0.0),
        ]:
            row = {
                "query_id": f"q{query}", "candidate_id": candidate,
                "truth_candidate_id": truth, "truth_formula": f"F{query}",
                "spectral_score": spectral,
            }
            row.update({feature: evidence for feature in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_bounded_residual_can_recover_consistent_evidence() -> None:
    ledger = synthetic_ledger()
    weights = fit_residual_weights(ledger, temperature=0.1, l2=0.1, maximum_weight=0.25)
    assert np.all(weights >= 0)
    assert np.all(weights <= 0.25 + 1e-12)
    predictions = score_queries(ledger, weights)
    gated = apply_gate(predictions, (0.1, 0.0, 2))
    assert gated["corrected"].sum() == len(gated)
    assert gated["introduced"].sum() == 0
