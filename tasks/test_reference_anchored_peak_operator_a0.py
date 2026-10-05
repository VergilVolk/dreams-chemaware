from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from build_reference_anchored_peak_operator_a0 import (  # noqa: E402
    exact_mcnemar_p,
    fit_pairwise_logistic,
    parse_feature_ids,
)


def main() -> None:
    assert parse_feature_ids("1;2.0; 3") == {1, 2, 3}
    assert parse_feature_ids("") == set()
    assert exact_mcnemar_p(0, 0) == 1.0
    assert 0.0 <= exact_mcnemar_p(8, 1) <= 1.0

    rows = []
    for identity, shift in (("AAAAAAAAAAAAAA", 0.0), ("BBBBBBBBBBBBBB", 0.1)):
        for sample in ("S1", "S2"):
            rows.append({
                "panel": "pos_rp", "ik14": identity, "sample_id": sample,
                "is_positive": True, "x": 0.9 + shift, "z": 0.8,
            })
            rows.append({
                "panel": "pos_rp", "ik14": identity, "sample_id": sample,
                "is_positive": False, "x": 0.2 + shift, "z": 0.1,
            })
    frame = pd.DataFrame(rows)
    weight, scale, audit = fit_pairwise_logistic(
        frame, ("x", "z"), {"AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"}, l2=0.1
    )
    assert audit["positive_negative_pairs"] == 4
    assert np.all(np.isfinite(weight))
    assert np.all(scale > 0)
    assert float(np.dot(np.array([0.7, 0.7]) / scale, weight)) > 0
    print("[test_reference_anchored_peak_operator_a0] PASS")


if __name__ == "__main__":
    main()
