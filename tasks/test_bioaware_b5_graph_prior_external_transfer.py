#!/usr/bin/env python
"""Unit checks for the B5 frozen graph-prior transfer audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b5_graph_prior_external_transfer import (  # noqa: E402
    FEATURES,
    frozen_candidate_scores,
)
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    BASELINE_MARGIN_MAX,
    PRIMARY_C,
    PROPOSAL_PROBABILITY_MIN,
)


def main() -> None:
    assert FEATURES == [
        "spectral_score", "network_member", "known_log_degree",
        "known_mass_candidate_fraction",
    ]
    assert PRIMARY_C == 0.1
    assert BASELINE_MARGIN_MAX == 0.05
    assert PROPOSAL_PROBABILITY_MIN == 0.75
    import pandas as pd
    frame = pd.DataFrame({
        "spectral_score": [1.0, 2.0],
        "network_member": [0.0, 1.0],
        "known_log_degree": [1.0, 3.0],
        "known_mass_candidate_fraction": [0.0, 1.0],
    })
    artifact = {
        "features": FEATURES,
        "scaler_mean": [1.0, 0.0, 1.0, 0.0],
        "scaler_scale": [1.0, 1.0, 2.0, 1.0],
        "model_coef": [1.0, 2.0, 4.0, 8.0],
        "model_intercept": 0.0,
    }
    observed = frozen_candidate_scores(frame, artifact)
    expected = np.asarray([0.0, 15.0])
    if not np.allclose(observed, expected, rtol=0, atol=1e-12):
        raise AssertionError((observed, expected))
    print("[test_bioaware_b5_graph_prior_external_transfer] PASS", flush=True)


if __name__ == "__main__":
    main()
