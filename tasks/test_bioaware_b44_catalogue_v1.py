#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd

from audit_bioaware_b16_pairwise_nonlinear_action import fit_model
from freeze_bioaware_b44_catalogue_v1 import FEATURES


def main() -> None:
    rows = []
    for query in range(40):
        truth = f"t{query}"
        wrong = f"w{query}"
        for candidate, positive, spectral, members in (
            (truth, True, 0.70, 2.0), (wrong, False, 0.72, 0.0)
        ):
            rows.append({
                "query_id": str(query), "candidate_id": candidate,
                "truth_candidate_id": truth, "truth_formula": f"F{query}",
                "is_positive": positive, "baseline_correct": False,
                "spectral_score": spectral,
                "independent_member_count": members,
                "independent_member_intersection": members / 2,
                "independent_log_degree_mean": members,
                "independent_log_degree_min": members / 2,
            })
    frame = pd.DataFrame(rows)
    model, report = fit_model(frame, FEATURES, 7)
    positive = frame.loc[frame["is_positive"], FEATURES].to_numpy(float)
    negative = frame.loc[~frame["is_positive"], FEATURES].to_numpy(float)
    assert np.all(model.predict_proba(positive - negative)[:, 1] > 0.5)
    assert report["training_queries"] == 40
    print("[test_bioaware_b44_catalogue_v1] PASS", flush=True)


if __name__ == "__main__":
    main()
