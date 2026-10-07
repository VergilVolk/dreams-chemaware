"""Synthetic tests for the shared factorial step selector.

S1: the selector picks the step with the best formula-cluster lower bound,
    not the noisiest point-estimate winner (a step that wins by one lucky
    cluster must lose to a stable step).
S2: ties resolve to the lowest index (determinism).
S3: cluster bootstrap bounds are sane (between min/max cluster means).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_select_shared_factorial_step import (  # noqa: E402
    cluster_bootstrap_lower_bound, select_step,
)


def main() -> None:
    # deterministic construction: 40 clusters x 20 queries
    cids = np.repeat(np.arange(40), 20)
    n = len(cids)
    correct_block = np.zeros(20); correct_block[:14] = 1.0  # 0.70 exact

    # step A: every cluster exactly 0.70
    a = np.tile(correct_block, 40)
    # step B: HIGHER point estimate 0.716 but high between-cluster variance
    # (22 clusters perfect, 18 clusters at 0.40) -> bootstrap LB ~0.62
    b = a.copy()
    spike = np.zeros(20); spike[:8] = 1.0          # 0.40 block
    for c in range(40):
        if c < 22:
            b[cids == c] = 1.0
        else:
            b[cids == c] = spike
    assert abs(b.mean() - (22 * 1.0 + 18 * 0.40) / 40) < 1e-9
    assert b.mean() > a.mean(), "deterministic setup: B wins point estimate"
    res = select_step([a, b], cids, n_boot=500)
    assert res["step_index"] == 0, "stable step must win on lower bound"
    assert res["lower_bounds"][0] > res["lower_bounds"][1] + 0.02, \
        "the win must come from a STRICTLY lower bound, not a tie"
    print(f"S1 PASS: chose stable step (lb {res['lower_bounds'][0]:.4f}) "
          f"over high-variance step (lb {res['lower_bounds'][1]:.4f}) "
          f"despite point estimates {res['point_estimates']}")

    # S2 ties
    res2 = select_step([a.copy(), a.copy()], cids, n_boot=200)
    assert res2["step_index"] == 0
    print("S2 PASS: tie resolves to lowest index")

    # S3 bound sanity
    cl_means = np.array([a[cids == c].mean() for c in np.unique(cids)])
    lb = cluster_bootstrap_lower_bound(a, cids, n_boot=500)
    assert cl_means.min() - 1e-9 <= lb <= a.mean() + 1e-9
    print(f"S3 PASS: bound {lb:.4f} within cluster-mean range "
          f"[{cl_means.min():.4f}, {a.mean():.4f}]")
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
