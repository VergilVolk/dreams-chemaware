"""Diagnose the P3 candidate pool: exclusion breakdown + train/val fold sizes.

Answers three questions for the P3 lock decision:
  1. Which exclusion source eats the most IK14 (marginal counts)?
  2. train fold (minus g8r/cache/large-audit) -> how many valid / isomer IK14?
  3. val fold (minus Test-A/B/C) -> how many valid / isomer IK14 (untapped)?
No manifest is written; this is read-only diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_TEST_DIR = ROOT / "data/validation/g8r_final_test"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_LARGE_DISCO = ROOT / "data/validation/large_observability_residual_audit/discovery_query_audit.csv"
DEFAULT_LARGE_CONFIRM = ROOT / "data/validation/large_observability_residual_audit/confirmation_query_audit.csv"


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def collect_sources(args):
    src = {}
    g8r_anchor = set(); g8r_neg = set()
    for p in (args.train, args.val):
        data = json.loads(p.read_text(encoding="utf-8"))
        for e in data["entries"]:
            g8r_anchor.add(e["ik14"])
            for n in e.get("neg", []):
                g8r_neg.add(n["ik14"])
    src["g8r_anchor"] = g8r_anchor
    src["g8r_neg"] = g8r_neg - g8r_anchor
    cache_ik = set()
    for cp, tag in ((args.cache, "cache"), (args.val_cache, "cache_val")):
        if cp.exists():
            d = np.load(cp, allow_pickle=True)
            for k in ("query_ik14", "candidate_ik14"):
                if k in d.files:
                    cache_ik.update(str(x) for x in d[k])
    src["raw_cache"] = cache_ik - g8r_anchor
    test_ik = set()
    for tp in ("test_a_manifest.json", "test_b_manifest.json", "test_c_manifest.json"):
        tp = args.test_dir / tp
        if tp.exists():
            m = json.loads(tp.read_text(encoding="utf-8"))
            for q in m["queries"]:
                test_ik.add(q["ik14"])
                for c in q.get("candidates", []):
                    test_ik.add(c["ik14"])
    src["test_abc"] = test_ik
    large_ik = set()
    for cp in (args.large_discovery, args.large_confirmation):
        if cp.exists():
            with open(cp, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    large_ik.add(row["ik14"])
    src["large_audit"] = large_ik
    return src


def pool_stats(args, fold_sel, excluded, pairs_near, pairs_mid):
    with h5py.File(args.data, "r") as h:
        fold = read_str(h, "fold")
        ikf = read_str(h, "INCHIKEY")
        formula = read_str(h, "FORMULA")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)
    ik14 = np.asarray([k[:14] for k in ikf], dtype=object)
    mask = np.isin(fold, fold_sel) & np.array([ik not in excluded for ik in ik14], dtype=bool)
    idx = np.where(mask)[0]
    iso_ik = set(); valid_ik = set(); near_ik = set(); nearmid_ik = set()
    for ad in np.unique(adduct[idx]):
        g = idx[adduct[idx] == ad]
        m = pmz[g]; order = np.argsort(m)
        gs = g[order]; ms = m[order]; ik_s = ik14[gs]; f_s = formula[gs]
        for pos in range(len(gs)):
            qik = ik_s[pos]; qf = f_s[pos]; qm = ms[pos]
            lo = np.searchsorted(ms, qm - 10e-6 * qm, side="left")
            hi = np.searchsorted(ms, qm + 10e-6 * qm, side="right")
            cand = np.arange(lo, hi); cand = cand[cand != pos]
            if len(cand) == 0:
                continue
            cik = ik_s[cand]; cf = f_s[cand]
            if not (cik == qik).any() or not (cik != qik).any():
                continue
            valid_ik.add(str(qik))
            iso_mask = (cik != qik) & (cf == qf)
            if iso_mask.any():
                iso_ik.add(str(qik))
                ns = set(pairs_near.get(str(qik), ()))
                ms_ = set(pairs_mid.get(str(qik), ()))
                if any(str(c) in ns for c in cik[iso_mask]):
                    near_ik.add(str(qik))
                if any(str(c) in (ns | ms_) for c in cik[iso_mask]):
                    nearmid_ik.add(str(qik))
    return {
        "n_spectra": int(mask.sum()),
        "n_ik14": int(len(np.unique(ik14[idx]))),
        "valid_ik14": len(valid_ik),
        "isomer_ik14": len(iso_ik),
        "near_ik14": len(near_ik),
        "nearmid_ik14": len(nearmid_ik),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    ap.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--large-discovery", type=Path, default=DEFAULT_LARGE_DISCO)
    ap.add_argument("--large-confirmation", type=Path, default=DEFAULT_LARGE_CONFIRM)
    args = ap.parse_args()

    src = collect_sources(args)
    # marginal breakdown (order = anchor -> neg -> cache -> test -> large)
    order = ["g8r_anchor", "g8r_neg", "raw_cache", "test_abc", "large_audit"]
    seen = set()
    print("=== exclusion marginal IK14 ===")
    for k in order:
        marg = src[k] - seen
        print(f"  {k:14s}: +{len(marg):6d}  (cumulative {len(seen | src[k])})")
        seen |= src[k]
    full_excl = seen
    print(f"  total excluded: {len(full_excl)}")

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    near_map = {}
    mid_map = {}
    for e in pairs.get("near", []):
        near_map.setdefault(e["ik_a"], set()).add(e["ik_b"])
        near_map.setdefault(e["ik_b"], set()).add(e["ik_a"])
    for e in pairs.get("mid", []):
        mid_map.setdefault(e["ik_a"], set()).add(e["ik_b"])
        mid_map.setdefault(e["ik_b"], set()).add(e["ik_a"])

    # train fold minus g8r/cache/large (test_abc is val-fold so irrelevant to train)
    train_excl = (src["g8r_anchor"] | src["g8r_neg"] | src["raw_cache"] | src["large_audit"])
    print("\n=== train fold, minus g8r(anchor+neg)+raw_cache+large_audit ===")
    st_tr = pool_stats(args, ["train"], train_excl, near_map, mid_map)
    print(f"  spectra={st_tr['n_spectra']} ik14={st_tr['n_ik14']} valid={st_tr['valid_ik14']} "
          f"isomer={st_tr['isomer_ik14']} near(0-2)={st_tr['near_ik14']} nearmid(0-5)={st_tr['nearmid_ik14']}")

    # val fold minus Test-A/B/C (fresh, untapped by g8r)
    val_excl = src["test_abc"]
    print("\n=== val fold, minus Test-A/B/C ===")
    st_va = pool_stats(args, ["val"], val_excl, near_map, mid_map)
    print(f"  spectra={st_va['n_spectra']} ik14={st_va['n_ik14']} valid={st_va['valid_ik14']} "
          f"isomer={st_va['isomer_ik14']} near(0-2)={st_va['near_ik14']} nearmid(0-5)={st_va['nearmid_ik14']}")

    # combined train+val
    print("\n=== combined train+val (train minus g8r/cache/large; val minus TestABC) ===")
    st_co = pool_stats(args, ["train", "val"], train_excl | val_excl, near_map, mid_map)
    print(f"  spectra={st_co['n_spectra']} ik14={st_co['n_ik14']} valid={st_co['valid_ik14']} "
          f"isomer={st_co['isomer_ik14']} near(0-2)={st_co['near_ik14']} nearmid(0-5)={st_co['nearmid_ik14']}")


if __name__ == "__main__":
    main()
