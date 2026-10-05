#!/usr/bin/env python
"""Unit checks for the B16 symmetric pairwise construction."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b16_pairwise_nonlinear_action import pairwise_training


def main() -> None:
    frame = pd.DataFrame({
        "query_id": ["q", "q", "q"],
        "truth_candidate_id": ["t", "t", "t"],
        "baseline_correct": [False, False, False],
        "is_positive": [True, False, False],
        "x": [3.0, 2.0, 0.0],
    })
    x, y, weights = pairwise_training(frame, ["x"])
    assert np.array_equal(x[:, 0], [1.0, 3.0, -1.0, -3.0])
    assert np.array_equal(y, [1, 1, 0, 0])
    assert np.isclose(weights.sum(), 2.0)
    print("[test_bioaware_b16_pairwise_nonlinear_action] PASS")


if __name__ == "__main__":
    main()
