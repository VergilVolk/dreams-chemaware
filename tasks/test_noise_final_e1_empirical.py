#!/usr/bin/env python
"""Small deterministic unit tests for E1 peak clustering and calibration helpers."""

from __future__ import annotations

import numpy as np

from calibrate_noise_final_e1_empirical import (
    cluster_group_peaks,
    condition_relation,
    e2_dropout_screening_grid,
    identity_equal_relation_quantiles,
    weighted_quantile,
)


def main() -> None:
    spectra = {
        1: (np.asarray([50.000, 100.000]), np.asarray([1.0, 0.2])),
        2: (np.asarray([50.008, 120.000]), np.asarray([0.8, 0.3])),
        3: (np.asarray([49.995, 100.010]), np.asarray([0.9, 0.1])),
    }
    clusters, duplicates = cluster_group_peaks(spectra, 0.02)
    supports = sorted(len(cluster) for cluster in clusters)
    assert supports == [1, 2, 3], supports
    assert duplicates == 0
    assert condition_relation("Orbitrap", 20.0, "Orbitrap", 22.0) == "same_instrument_same_ce"
    assert condition_relation("Orbitrap", 20.0, "Orbitrap", 40.0) == "same_instrument_cross_ce"
    assert condition_relation("Orbitrap", 20.0, "QTOF", 20.0) == "cross_instrument"
    quantiles = weighted_quantile(np.asarray([0.1, 0.5, 0.9]), np.asarray([1, 2, 1]), [0.25, 0.5, 0.75])
    assert quantiles[0] <= quantiles[1] <= quantiles[2]
    frame = __import__("pandas").DataFrame(
        {
            "relation": ["cross"] * 4,
            "ik14": ["a", "a", "a", "b"],
            "formula": ["x", "x", "x", "y"],
            "drop": [0.1, 0.2, 0.3, 0.9],
        }
    )
    summary = identity_equal_relation_quantiles(frame, "drop")
    assert summary["cross"]["q50"] > 0.3, summary
    screening = e2_dropout_screening_grid(
        {"cross": {**summary["cross"], "n": 100, "identities": 25}}
    )
    assert max(screening["cross"]["doses"]) <= 0.30
    print("[test_noise_final_e1_empirical] PASS", flush=True)


if __name__ == "__main__":
    main()
