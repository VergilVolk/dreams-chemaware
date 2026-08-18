"""Explore rule-hit structure on Met/neg query spectra to decide the rule-injection
mechanism (do NOT invent a threshold; measure first).

Caches the 13,770 x 335 rule-hit matrix to
data/msv100574/embeddings/met_neg/rule_hits.npy + rule_meta.json, then reports:
  1. per-category hit density (NL is ~18.6/spectrum -> no discrimination; CF is
     ~1.3/spectrum -> sparse, structurally specific),
  2. CF-hit count vs confident-annotation rate (does CF evidence track confident
     library matches?).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M_H = 1.00782503223


def target_hit(sv: np.ndarray, t: float, tol: float) -> bool:
    if sv.size == 0:
        return False
    p = int(np.searchsorted(sv, t))
    if p < sv.size and abs(float(sv[p]) - t) < tol:
        return True
    return p > 0 and abs(float(sv[p - 1]) - t) < tol


def range_hit(sv: np.ndarray, lo: float, hi: float) -> bool:
    if sv.size == 0:
        return False
    p = int(np.searchsorted(sv, lo, side="left"))
    return p < sv.size and float(sv[p]) <= hi


def spectrum_rule_vector(mz_padded: np.ndarray, precursor: float, rules: list[dict]) -> np.ndarray:
    # Canonical matcher, bit-for-bit with tasks/build_spectrum_rule_label_cache.py
    # and pilot_rule_noise_stress.FastRuleMatcher (P1 code path).
    mz = np.sort(mz_padded[np.isfinite(mz_padded) & (mz_padded > 0)].astype(np.float64))
    diffs = np.sort(np.abs(mz[:, None] - mz[None, :]).reshape(-1)) if mz.size else np.empty(0)
    labels = np.zeros(len(rules), dtype=np.uint8)
    for i, r in enumerate(rules):
        k = r["match_type"]
        v = r["value"]
        if k == "mass_diff":
            labels[i] = target_hit(diffs, float(v), 0.02)
        elif k == "peak_mz":
            labels[i] = target_hit(mz, float(v), 0.02)
        elif k == "mass_range":
            labels[i] = range_hit(diffs, float(v[0]), float(v[1]))
        elif k == "hr_shift":
            nh = float(v)
            if nh == 0:
                e = diffs[diffs >= 12.0]
                labels[i] = bool(e.size and np.any(np.abs(e - np.round(e)) < 0.02))
            else:
                labels[i] = target_hit(diffs, abs(nh) * M_H, 0.02)
        elif k == "parity":
            labels[i] = bool(diffs.size and np.any((np.round(diffs).astype(np.int64) % 2) == (round(precursor) % 2)))
        elif k == "mass_diff_range":
            lo, hi = map(float, v)
            labels[i] = bool(diffs.size and np.any((diffs > hi) | (diffs < lo)))
    return labels


def main() -> None:
    rules = json.loads((ROOT / "dreams/models/chem_aware/chem_rules_data.json").read_text(encoding="utf-8"))["rules"]
    manifest = pd.read_csv(ROOT / "data/msv100574/embeddings/met_neg/manifest.csv")

    vecs = []
    for fname, grp in manifest.groupby("file_name"):
        hdf = ROOT / f"data/msv100574/Metabolomics/neg/{fname}.hdf5"
        with h5py.File(hdf, "r") as h:
            spec = np.asarray(h["spectrum"][:], dtype=np.float32)
            prec = np.asarray(h["precursor_mz"][:], dtype=np.float32)
        for r in grp["row_in_file"].to_numpy():
            vecs.append(spectrum_rule_vector(spec[r][0], float(prec[r]), rules))
    V = np.stack(vecs).astype(np.uint8)
    print("rule-hit matrix:", V.shape, flush=True)

    out_dir = ROOT / "data/msv100574/embeddings/met_neg"
    np.save(out_dir / "rule_hits.npy", V)
    (out_dir / "rule_meta.json").write_text(json.dumps({
        "rule_name": [r["name"] for r in rules],
        "rule_category": [r["category"] for r in rules],
        "rule_match_type": [r["match_type"] for r in rules],
        "rule_source": [r.get("source", "") for r in rules],
        "n_rules": len(rules),
        "tolerance_Da": 0.02,
        "matcher": "build_spectrum_rule_label_cache.spectrum_rule_vector (canonical)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("cached -> rule_hits.npy + rule_meta.json", flush=True)

    cats = [r["category"] for r in rules]
    cat_idx = {c: [i for i, x in enumerate(cats) if x == c] for c in dict.fromkeys(cats)}
    print("\nper-category hit density (hits/spectrum):", flush=True)
    for c, idx in cat_idx.items():
        print(f"  {c:4s} n_rules={len(idx):3d}  mean={V[:, idx].sum(axis=1).mean():6.2f}", flush=True)

    # CF sparse-signal discrimination check
    cf = V[:, cat_idx["CF"]].sum(axis=1)
    print("\nCF hit count distribution:", flush=True)
    from collections import Counter
    print("  ", dict(sorted(Counter(cf.tolist()).items())[:10]), flush=True)
    print(f"  fraction spectra with >=1 CF hit: {(cf >= 1).mean():.4f}", flush=True)

    # CF hit count vs confident-annotation rate (top1 cosine>=0.7 & mz_pass)
    ann = pd.read_csv(ROOT / "data/msv100574/annotation/met_neg/annotations.csv")
    top1 = ann[ann["rank"] == 1].sort_values("query_idx")
    conf = ((top1["cosine"] >= 0.7) & top1["mz_pass"]).to_numpy()
    assert len(conf) == len(cf), (len(conf), len(cf))
    print("\nCF hit count vs confident-annotation rate:", flush=True)
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 999)]:
        mask = (cf >= lo) & (cf <= hi)
        if mask.sum():
            print(f"  CF in [{lo},{hi if hi<999 else 'inf'}]: n={mask.sum():5d}  confident_rate={conf[mask].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
