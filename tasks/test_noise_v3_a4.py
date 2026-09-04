"""Dependency-light unit checks for A4 matching and strict ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from audit_noise_v3_a4_exact_peak_scan import select_matched_controls, strict_detail


def main() -> None:
    detail = strict_detail(
        np.asarray([0.8, 0.7, 0.8, 0.2]), np.asarray([10, 11, 12, 13]),
        np.asarray([0, 2, 4]),
    )
    assert detail["rank"] == 2, detail
    assert detail["adversarial_molecule_local"] == 1, detail
    assert detail["adversarial_pair_row"] == 12, detail

    rows = []
    for query in range(12):
        rows.append({
            "query_index": query,
            "query_row": query,
            "query_ik14": f"IK{query:012d}",
            "query_formula": "F1" if query < 7 else "F2",
            "has_near": query % 2 == 0,
            "baseline_rank": 2 if query in (0, 1) else 1,
            "baseline_margin": -0.1 if query in (0, 1) else 0.1 + query / 100,
            "candidate_molecules": 3 + query % 3,
            "peak_count": 20 + query,
        })
    frame = pd.DataFrame(rows)
    matches = select_matched_controls(
        frame, frame.loc[frame["baseline_rank"] > 1], per_error=2, reuse_cap=2,
    )
    assert matches.groupby("error_query_index").size().eq(2).all()
    assert set(matches["error_query_index"]) == {0, 1}
    assert not set(matches["control_query_index"]) & {0, 1}
    print("[A4 unit tests] PASS", flush=True)


if __name__ == "__main__":
    main()
