"""Deployable confidence gate for library-search annotations.

Given any method's per-query evaluation table (query_index, rank,
top1_top2_gap) this tool emits the truth-blind risk-coverage operating
curve and gate thresholds for target accuracies. Deployment story: accept
the top-1 annotation when the gap percentile clears the gate; otherwise
escalate the feature to orthogonal evidence (MSn, chemical experts,
manual). The gate needs only the score distribution of the current batch
-- no labels, no training.

Usage:
  python -X utf8 tasks/GLM_confidence_gate.py \
      --table data/validation/.../queries_identity_disjoint_noise_v1.csv.gz \
      --targets 99.0 99.5 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def gate_table(rank: np.ndarray, gap: np.ndarray,
               targets=(99.0, 99.5)) -> dict:
    correct = rank == 1
    n = len(rank)
    order = np.argsort(-gap, kind="stable")  # most confident first
    c_sorted = correct[order]
    g_sorted = gap[order]
    covs = np.arange(1, n + 1) / n
    accs = np.cumsum(c_sorted) / np.arange(1, n + 1)
    out = {
        "n_queries": int(n),
        "full_recall1": round(float(correct.mean() * 100), 2),
        "curve": [
            {"coverage": round(float(covs[i]), 4),
             "accuracy": round(float(accs[i]) * 100),  # per-query integer %
             "gap_threshold": round(float(g_sorted[i]), 6)}
            for i in range(0, n, max(1, n // 200))
        ] + [{"coverage": 1.0, "accuracy": round(float(accs[-1]) * 100),
              "gap_threshold": round(float(g_sorted[-1]), 6)}],
        "gates": {},
    }
    for t in targets:
        ok = np.flatnonzero(accs * 100 >= t)
        if len(ok) == 0:
            out["gates"][f"acc>={t}"] = None
            continue
        k = ok[-1]  # largest coverage still meeting the target
        out["gates"][f"acc>={t}"] = {
            "coverage": round(float(covs[k]) * 100, 2),
            "accuracy": round(float(accs[k] * 100), 2),
            "gap_threshold_at_least": round(float(g_sorted[k]), 6),
            "note": "accept queries with gap >= threshold; escalates the rest",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True,
                    help="per-query csv.gz with rank and top1_top2_gap")
    ap.add_argument("--targets", type=float, nargs="+", default=[99.0, 99.5])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.table, low_memory=False).sort_values(
        "query_index", kind="stable")
    res = gate_table(df["rank"].to_numpy(), df["top1_top2_gap"].to_numpy(),
                     tuple(args.targets))
    res["table"] = str(args.table)
    text = json.dumps(res, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
        print(f"written: {args.json}")
    print(f"full R@1 = {res['full_recall1']}%")
    for k, v in res["gates"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
