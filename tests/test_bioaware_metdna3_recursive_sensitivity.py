from __future__ import annotations

import pandas as pd

from tasks.summarize_bioaware_metdna3_recursive_sensitivity import robust_rows


def test_robust_rows_requires_majority_of_seven_rotations() -> None:
    frame = pd.DataFrame({
        "graph_step": [0] * 14,
        "maximum_depth": [2] * 14,
        "baseline_correct": [False] * 14,
        "query_id": ["q1"] * 7 + ["q2"] * 7,
        "truth_candidate_id": ["a"] * 7 + ["b"] * 7,
        "fold": list(range(7)) * 2,
        "strict_rescue_headroom": [True] * 4 + [False] * 3 + [True] * 3 + [False] * 4,
        "truth_supported": [True] * 14,
        "wrong_supported": [False] * 14,
    })
    result = robust_rows(frame, 0, 2, 4)
    assert result["query_id"].tolist() == ["q1"]
