"""Shared factorial checkpoint-step selector (audit section 11, file 4).

Contract: ONE checkpoint step is chosen using ONLY the R arm's role-2
validation curve, then mechanically applied to all six arms. No arm may
pick its own best point.

Selection rule (mechanical, formula-cluster guarded):
  1. For each candidate step, compute the formula-cluster bootstrap LOWER
     BOUND (2.5th percentile, B resamples resampling formula clusters) of
     the R arm's primary metric (default: Recall@1).
  2. Choose the step with the highest lower bound (not the point estimate)
     - this is the noise-robust choice and cannot ride cluster noise.
  3. Record the full curve, bounds, and the chosen step into a JSON that
     the trainer consumes; the trainer MUST refuse to run without it.

Ties: lowest step index wins (deterministic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cluster_bootstrap_lower_bound(per_query: np.ndarray,
                                  cluster_ids: np.ndarray,
                                  n_boot: int = 2000,
                                  seed: int = 20261007,
                                  percentile: float = 2.5) -> float:
    """Bootstrap the metric by resampling FORMULA CLUSTERS (not queries)."""
    uniq = np.unique(cluster_ids)
    index = {c: np.flatnonzero(cluster_ids == c) for c in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[c] for c in picked])
        stats[b] = per_query[rows].mean()
    return float(np.percentile(stats, percentile))


def select_step(metrics_per_step: list[np.ndarray],
                cluster_ids: np.ndarray,
                n_boot: int = 2000) -> dict:
    """metrics_per_step[k]: per-query 0/1 correctness of step k on R arm."""
    bounds, points = [], []
    for m in metrics_per_step:
        points.append(float(np.mean(m)))
        bounds.append(cluster_bootstrap_lower_bound(m, cluster_ids, n_boot))
    best = int(np.argmax(bounds))  # ties -> lowest index
    return {"step_index": best, "lower_bounds": bounds, "point_estimates":
            points, "rule": "max formula-cluster bootstrap lower bound (2.5th "
            "pct), R arm only, ties to lowest step"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-arm-metrics", type=Path, required=True,
                    help="npz: metrics_per_step (S, N) 0/1 arrays + "
                         "cluster_ids (N,)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.r_arm_metrics) as z:
        mps = np.asarray(z["metrics_per_step"], dtype=np.float32)
        cids = np.asarray(z["cluster_ids"])
    metrics = [mps[k] for k in range(mps.shape[0])]
    result = select_step(metrics, cids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"chosen step {result['step_index']} "
          f"(lb {max(result['lower_bounds']):.4f}); "
          f"point-estimate-best step {int(np.argmax(result['point_estimates']))}"
          f" ({max(result['point_estimates']):.4f})")
    print("written:", args.output)


if __name__ == "__main__":
    main()
