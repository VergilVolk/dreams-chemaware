"""Decompose the RAW reranker's overall 44/17 Top-1 corrections by MCES stratum.

Answers (per the 2026-08-22 audit): of the 44 baseline errors the reranker fixed
and the 17 errors it introduced, what is the MCES of the WRONG Top-1 candidate?
Reports conditional correction rate (corrected/N per stratum), introduced type
distribution, net gain per stratum, and gate-on vs gate-off (counterfactual).

Reuses the val feature cache; only MCES of wrong candidates is computed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from build_utils import compute_mces  # noqa: E402
from train_g8r_raw_reranker import (  # noqa: E402
    fit_ranker, score_frame, retrieval_query, gated, RAW_FEATURES,
)

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_44_17.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--gate-threshold", type=float, default=0.24098341166973114)
    p.add_argument("--gate-require-disagreement", action="store_true", default=False)
    return p.parse_args()


def mces_bin(v):
    if v is None or not np.isfinite(v):
        return "missing"
    if 0 <= v <= 2:
        return "0-2"
    if 3 <= v <= 5:
        return "3-5"
    if 6 <= v <= 10:
        return "6-10"
    return ">10"


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


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
    gated_ = gated(base, reranked, a.gate_threshold, a.gate_require_disagreement)

    # ik14 -> smiles map from the val anchors (via the val.json anchor rows).
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    with h5py.File(a.data, "r") as h:
        smiles_all = read_str(h, "smiles")
    ik2smi = {}
    for e in val:
        ik2smi.setdefault(e["ik14"], smiles_all[int(e["anchor_row"])])

    def wrong_mces(query_ik, wrong_ik):
        if query_ik == wrong_ik or wrong_ik not in ik2smi or query_ik not in ik2smi:
            return None
        return compute_mces(ik2smi[query_ik], ik2smi[wrong_ik])

    # full (ungated) transition
    merged = base[["query_index", "ik14", "chosen_ik14", "top1"]].merge(
        reranked[["query_index", "chosen_ik14", "top1"]], on="query_index",
        suffixes=("_base", "_rer"))
    corrected_mask = (~merged["top1_base"].astype(bool)) & merged["top1_rer"].astype(bool)
    introduced_mask = merged["top1_base"].astype(bool) & (~merged["top1_rer"].astype(bool))

    corrected = merged[corrected_mask].copy()
    introduced = merged[introduced_mask].copy()
    corrected["wrong_mces"] = [wrong_mces(q, w) for q, w in
                               zip(corrected["ik14"], corrected["chosen_ik14_base"])]
    introduced["wrong_mces"] = [wrong_mces(q, w) for q, w in
                                zip(introduced["ik14"], introduced["chosen_ik14_rer"])]

    # baseline error counts per MCES stratum (denominator for conditional rate)
    base_err = merged[~merged["top1_base"].astype(bool)].copy()
    base_err["wrong_mces"] = [wrong_mces(q, w) for q, w in
                              zip(base_err["ik14"], base_err["chosen_ik14_base"])]

    bins = ["0-2", "3-5", "6-10", ">10", "missing"]
    table1 = {}
    for b in bins:
        n = int((base_err["wrong_mces"].map(mces_bin) == b).sum())
        c = int((corrected["wrong_mces"].map(mces_bin) == b).sum())
        table1[b] = {"baseline_errors": n, "corrected": c,
                     "correction_rate": (c / n) if n else None}
    table2 = {}
    for b in bins:
        table2[b] = {"introduced": int((introduced["wrong_mces"].map(mces_bin) == b).sum())}

    # net gain per stratum
    net = {}
    for b in bins:
        c = table1[b]["corrected"]
        i = table2[b]["introduced"]
        net[b] = {"corrected": c, "introduced": i, "net": c - i}

    # gate analysis: gate-on actual vs gate-off counterfactual (full reranker)
    gated_merged = base[["query_index", "ik14", "top1"]].merge(
        gated_[["query_index", "top1", "gate_used"]], on="query_index", suffixes=("_b", "_g"))
    gate_on = gated_merged[gated_merged["gate_used"]]
    gate_off = gated_merged[~gated_merged["gate_used"]]
    # counterfactual: if forced reranker on gate-off, what changes
    off_forced = merged[merged["query_index"].isin(set(gate_off["query_index"]))]
    gate_report = {
        "gate_on": {
            "n": int(len(gate_on)),
            "corrected": int(((~gate_on["top1_b"].astype(bool)) & gate_on["top1_g"].astype(bool)).sum()),
            "introduced": int((gate_on["top1_b"].astype(bool) & (~gate_on["top1_g"].astype(bool))).sum()),
        },
        "gate_off_counterfactual": {
            "n": int(len(gate_off)),
            "corrected": int(((~off_forced["top1_base"].astype(bool)) & off_forced["top1_rer"].astype(bool)).sum()),
            "introduced": int((off_forced["top1_base"].astype(bool) & (~off_forced["top1_rer"].astype(bool))).sum()),
        },
    }

    report = {
        "status": "g8r_raw_reranker_44_17",
        "n_queries": int(len(merged)),
        "total_corrected": int(corrected_mask.sum()),
        "total_introduced": int(introduced_mask.sum()),
        "table1_baseline_error_conditional_correction": table1,
        "table2_introduced_error_types": table2,
        "net_gain_per_stratum": net,
        "gate_analysis": gate_report,
        "mces_missing": {
            "corrected": int(corrected["wrong_mces"].isna().sum()),
            "introduced": int(introduced["wrong_mces"].isna().sum()),
            "baseline_errors": int(base_err["wrong_mces"].isna().sum()),
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
