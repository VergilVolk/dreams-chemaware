"""Static and semantic tests for E11 reference selection."""
from __future__ import annotations

import numpy as np

from audit_noise_final_e11_reference_diversity_matrix import (
    REFERENCE_POLICIES, maxmin_indices, select_rows,
)


def main() -> None:
    vectors = np.eye(4, dtype=np.float32)
    scores = np.asarray([0.9, 0.6, 0.2, 0.1])
    chosen = maxmin_indices(vectors, scores, 3)
    if len(chosen) != 3 or chosen[0] != 0 or len(set(chosen.tolist())) != 3:
        raise AssertionError("max-min selection contract failed")
    rows = np.asarray([10, 11, 12, 13])
    instruments = {10: "Orbitrap", 11: "QTOF", 12: "Orbitrap", 13: "unknown"}
    ce = {10: 20.0, 11: 20.0, 12: 50.0, 13: float("nan")}
    far = select_rows(rows, scores, vectors, "farthest3", "Orbitrap", 20.0, instruments, ce)
    if far.tolist() != [13, 12, 11]:
        raise AssertionError("farthest selection is not deterministic")
    for policy in REFERENCE_POLICIES:
        selected = select_rows(rows, scores, vectors, policy, "Orbitrap", 20.0, instruments, ce)
        if not len(selected) or len(np.unique(selected)) != len(selected):
            raise AssertionError(f"invalid E11 selection for {policy}")
    print("[test_noise_final_e11_reference_diversity] PASS", flush=True)


if __name__ == "__main__":
    main()
