"""Bootstrap CI + McNemar for the RAW reranker's per-stratum net gain.

Companion to audit_g8r_raw_reranker_44_17.py: re-runs the retrieval, computes the
per-query transition (+1 corrected / -1 introduced / 0 unchanged) and the MCES
stratum of the wrong candidate, then for each stratum reports:
  - net gain (sum of +1/-1);
  - formula-cluster paired bootstrap 95% CI of the mean per-query delta;
  - McNemar (exact binomial) p-value that corrected != introduced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from build_utils import compute_mces  # noqa: E402
from train_g8r_raw_reranker import (  # noqa: E402
    fit_ranker, score_frame, retrieval_query, RAW_FEATURES,
)

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_bootstrap.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260822)
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


def mcnemar(corrected, introduced):
    total = corrected + introduced
    if total == 0:
        return None
    from scipy.stats import binomtest
    return float(binomtest(corrected, total, 0.5).pvalue)


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

    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    with h5py.File(a.data, "r") as h:
        smiles_all = read_str(h, "smiles")
    ik2smi = {}
    for e in val:
        ik2smi.setdefault(e["ik14"], smiles_all[int(e["anchor_row"])])

    merged = base[["query_index", "ik14", "chosen_ik14", "top1"]].merge(
        reranked[["query_index", "chosen_ik14", "top1"]], on="query_index",
        suffixes=("_base", "_rer"))
    # formula for cluster bootstrap
    merged = merged.merge(base[["query_index", "formula"]], on="query_index", how="left")

    corrected = (~merged["top1_base"].astype(bool)) & merged["top1_rer"].astype(bool)
    introduced = merged["top1_base"].astype(bool) & (~merged["top1_rer"].astype(bool))

    def wrong_stratum(row):
        q = row["ik14"]
        w = row["chosen_ik14_base"] if not row["top1_base"] else row["chosen_ik14_rer"]
        return mces_bin(compute_mces(ik2smi.get(q, ""), ik2smi.get(w, "")))

    merged["wrong_stratum"] = merged.apply(wrong_stratum, axis=1)
    # per-query delta (overall transition) + stratum-g delta
    delta = np.where(corrected, 1.0, np.where(introduced, -1.0, 0.0))
    merged["delta"] = delta

    bins = ["0-2", "3-5", "6-10", ">10", "missing"]
    report = {"status": "g8r_raw_reranker_bootstrap",
              "n_queries": int(len(merged)),
              "total_corrected": int(corrected.sum()),
              "total_introduced": int(introduced.sum()),
              "overall": {}}
    # overall
    report["overall"] = {
        "net_gain": int(delta.sum()),
        "mean_delta": float(delta.mean()),
        "bootstrap": bootstrap_stratum(delta, merged["formula"].to_numpy(), a.bootstrap, a.seed),
        "mcnemar_p": mcnemar(int(corrected.sum()), int(introduced.sum())),
    }
    for b in bins:
        mask = (merged["wrong_stratum"] == b).to_numpy()
        d = delta.copy()
        d[~mask] = 0.0
        c = int((corrected & (merged["wrong_stratum"] == b)).sum())
        i = int((introduced & (merged["wrong_stratum"] == b)).sum())
        report[b] = {
            "corrected": c, "introduced": i, "net_gain": c - i,
            "mean_delta": float(d.mean()),
            "bootstrap": bootstrap_stratum(d, merged["formula"].to_numpy(), a.bootstrap, a.seed),
            "mcnemar_p": mcnemar(c, i),
        }

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


def bootstrap_stratum(delta, formulas, n_boot, seed):
    d = pd.DataFrame({"delta": delta, "formula": formulas})
    by_f = d.groupby("formula")["delta"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(by_f, len(by_f), replace=True).mean() for _ in range(n_boot)])
    return {"mean": float(d["delta"].mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


if __name__ == "__main__":
    main()
