"""Focused tests for the frozen-score A1 strong-baseline audit."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from audit_reference_anchored_multi_probe_a1_strong_baseline import bootstrap  # noqa: E402


def main() -> None:
    frame = pd.DataFrame({
        "query_ik14": ["A", "A", "B", "C"],
        "candidate__ndcg5": [0.8, 0.6, 0.7, 0.9],
        "baseline__ndcg5": [0.5, 0.5, 0.5, 0.5],
    })
    result = bootstrap(frame, "candidate", "baseline", "ndcg5", 2000, 11)
    assert result["identity_clusters"] == 3
    assert math.isclose(result["mean_delta"], np.mean([0.2, 0.2, 0.4]))
    assert result["ci_low"] >= 0
    print("[test_reference_anchored_multi_probe_a1_strong_baseline] PASS")


if __name__ == "__main__":
    main()
