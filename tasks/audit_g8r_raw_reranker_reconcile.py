"""Reconcile the 45/18 (ungated) vs 44/17 (gated) transition counts.

The gated final model (44 corrected / 17 introduced) drops 1 corrected + 1
introduced relative to the ungated full reranker (45/18).  Both are net +27.
This script outputs the exact query IDs of those two differences and confirms
they are the gate-off queries the ungated reranker would have flipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_g8r_raw_reranker import (  # noqa: E402
    fit_ranker, score_frame, retrieval_query, gated, RAW_FEATURES,
)

DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_reconcile.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--gate-threshold", type=float, default=0.24098341166973114)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    va_cache = np.load(a.val_cache, allow_pickle=True)
    va = pd.DataFrame({k: va_cache[k] for k in va_cache.files})

    features = ["dreams_similarity"] + RAW_FEATURES
    scaler, rk = fit_ranker(tr, features, a.hard_k, a.C)
    base = retrieval_query(va, "dreams_similarity")
    reranked = retrieval_query(score_frame(va, features, scaler, rk), "score")
    g = gated(base, reranked, a.gate_threshold, False)

    def transitions(query):
        out = []
        for _, row in query.iterrows():
            b = bool(row["top1_base"]) if "top1_base" in row else bool(row["top1"])
            # handled below via merge
        return out

    # ungated transition
    ung = base[["query_index", "ik14", "top1"]].merge(
        reranked[["query_index", "top1"]], on="query_index", suffixes=("_b", "_r"))
    ung["ungated"] = np.where((~ung["top1_b"].astype(bool)) & ung["top1_r"].astype(bool), "corrected",
                    np.where(ung["top1_b"].astype(bool) & (~ung["top1_r"].astype(bool)), "introduced", "unchanged"))

    # gated transition
    gat = base[["query_index", "ik14", "top1"]].merge(
        g[["query_index", "top1"]], on="query_index", suffixes=("_b", "_r"))
    gat["gated"] = np.where((~gat["top1_b"].astype(bool)) & gat["top1_r"].astype(bool), "corrected",
                  np.where(gat["top1_b"].astype(bool) & (~gat["top1_r"].astype(bool)), "introduced", "unchanged"))

    merged = ung[["query_index", "ik14", "ungated"]].merge(
        gat[["query_index", "gated"]], on="query_index")
    diff = merged[merged["ungated"] != merged["gated"]]

    report = {
        "status": "g8r_raw_reranker_reconcile",
        "gate_threshold": a.gate_threshold,
        "ungated_counts": ung["ungated"].value_counts().to_dict(),
        "gated_counts": gat["gated"].value_counts().to_dict(),
        "diff_queries": [
            {"query_index": int(r["query_index"]), "ik14": r["ik14"],
             "ungated": r["ungated"], "gated": r["gated"]}
            for _, r in diff.iterrows()
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
