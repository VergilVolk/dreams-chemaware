"""Fast structural tests for the A4 nonlinear action teacher."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from train_noise_v3_a4_nonlinear_action_teacher import cluster_bootstrap, safe_auc


def main() -> None:
    metric = safe_auc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert metric["roc_auc"] == 1.0
    assert metric["average_precision"] == 1.0
    frame = pd.DataFrame({"query_formula": ["A", "A", "B", "C"]})
    result = cluster_bootstrap(frame, np.asarray([1.0, -1.0, 1.0, 1.0]), 200, 7)
    assert np.isfinite(result["mean"])
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    with tempfile.TemporaryDirectory() as directory:
        assert Path(directory).is_dir()
    print("[test_noise_v3_a4_action_teacher] PASS", flush=True)


if __name__ == "__main__":
    main()
