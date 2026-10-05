"""GLM Track-A: Gate-1 upper-bound migration reliability card.

Library-spectrum upper bound for cross-study entity migration: recall@k of
the true reference among candidates, per method, on both frozen GNPS panels.
Run 2349091 per-query tables; no model fitting, read-only.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

BASE = os.path.join("data", "validation", "noise_gnps_article_benchmark_run_2349091",
                    "evaluation")
KS = (1, 3, 5, 10, 20)


def main() -> None:
    card = {}
    for panel in ("identity_disjoint", "formula_disjoint"):
        rows = {}
        for path in sorted(glob.glob(f"{BASE}/queries_{panel}_*.csv.gz")):
            method = os.path.basename(path)[len(f"queries_{panel}_"):-len(".csv.gz")]
            frame = pd.read_csv(path, usecols=["rank", "near", "macro_query_auc"])
            rank = frame["rank"]
            rows[method] = {
                **{f"recall@{k}": round(float((rank <= k).mean()), 4) for k in KS},
                "mrr": round(float((1.0 / rank.astype(float)).mean()), 4),
                "macro_query_auc": round(float(frame["macro_query_auc"].mean()), 4),
                "near_recall@5": round(float((frame.loc[frame["near"], "rank"] <= 5)
                                             .mean()), 4),
                "queries": int(len(frame)),
            }
        card[panel] = dict(sorted(rows.items(),
                                  key=lambda kv: -kv[1]["recall@1"]))
    out = {"status": "GLM_MIGRATION_RELIABILITY_UPPER_BOUND_CARD",
           "scope": "library-spectrum cross-file matching; upper bound only, "
                    "cohort-DDA features will be worse (Gate 1 proper)",
           "run": "noise_gnps_article_benchmark_run_2349091",
           "panels": card}
    payload = json.dumps(out, indent=2)
    print(payload)
    os.makedirs("data/validation/GLM_track2_census", exist_ok=True)
    with open(os.path.join("data", "validation", "GLM_track2_census",
                           "migration_reliability_upper_bound.json"), "w",
              encoding="utf-8") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
