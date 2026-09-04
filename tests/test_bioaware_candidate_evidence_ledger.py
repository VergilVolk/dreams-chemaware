from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.build_bioaware_candidate_evidence_ledger import aggregate_paths


def test_path_aggregation_uses_heldout_fraction(tmp_path) -> None:
    frame = pd.DataFrame({
        "fold": [0, 1], "query_id": ["q", "q"], "candidate_id": ["c", "c"],
        "truth_candidate_id": ["t", "t"], "truth_formula": ["F", "F"],
        "maximum_depth": [3, 3], "complete_ms2_paths": [1, 0],
        "identity_paths": [2, 0], "best_bottleneck": [0.8, np.nan],
        "median_bottleneck": [0.6, np.nan],
    })
    path = tmp_path / "paths.csv"
    frame.to_csv(path, index=False)
    result = aggregate_paths(path, "x")
    assert result.iloc[0].x_path_fraction == 0.5
    assert result.iloc[0].x_identity_path_fraction == 0.5
    assert result.iloc[0].x_best_bottleneck == 0.8
