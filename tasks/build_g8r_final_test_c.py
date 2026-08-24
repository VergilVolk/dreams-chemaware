"""Lock Test-C: the next-generation blind test, disjoint from Test-A/Test-B.

Samples valid queries at the IK14 level from the HDF5 val fold, EXCLUDING all
IK14 already used by Test-A and Test-B.  Seeded random, target 1500-3000.
Saves the full candidate list + hashes, so the next-generation model
(peak token + listwise + peak-level RAW) has a fresh blind test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_A = ROOT / "data/validation/g8r_final_test/test_a_manifest.json"
DEFAULT_B = ROOT / "data/validation/g8r_final_test/test_b_manifest.json"
DEFAULT_OUT = ROOT / "data/validation/g8r_final_test/test_c_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--test-a", type=Path, default=DEFAULT_A)
    p.add_argument("--test-b", type=Path, default=DEFAULT_B)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=1500)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260823)
    return p.parse_args()


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def main() -> None:
    a = parse_args()
    used_ik = set()
    for p in (a.test_a, a.test_b):
        m = json.loads(p.read_text(encoding="utf-8"))
        used_ik.update(q["ik14"] for q in m["queries"])
    print(f"used IK14 (A+B): {len(used_ik)}")

    with h5py.File(a.data, "r") as h:
        fold = read_str(h, "fold")
        ikf = read_str(h, "INCHIKEY")
        formula = read_str(h, "FORMULA")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)

    val_idx = np.where(fold == "val")[0]
    ik14 = np.asarray([k[:14] for k in ikf], dtype=object)
    va_ik14 = ik14[val_idx]
    va_formula = formula[val_idx]
    va_adduct = adduct[val_idx]
    va_pmz = pmz[val_idx]

    valid = []
    for ad in np.unique(va_adduct):
        g = np.where(va_adduct == ad)[0]
        m = va_pmz[g]
        order = np.argsort(m)
        gs = g[order]; ms = m[order]; ik_s = va_ik14[gs]
        for pos in range(len(gs)):
            qrow = int(val_idx[gs[pos]]); qm = ms[pos]; qik = ik_s[pos]
            if qik in used_ik:
                continue
            lo = np.searchsorted(ms, qm - a.ppm_tol * 1e-6 * qm, side="left")
            hi = np.searchsorted(ms, qm + a.ppm_tol * 1e-6 * qm, side="right")
            cand = np.arange(lo, hi); cand = cand[cand != pos]
            if len(cand) == 0:
                continue
            cik = ik_s[cand]
            if not (cik == qik).any() or not (cik != qik).any():
                continue
            cand_rows = [int(val_idx[gs[j]]) for j in cand]
            valid.append({"row": qrow, "ik14": str(qik),
                          "formula": str(formula[qrow]),
                          "adduct": str(ad), "precursor_mz": float(qm),
                          "candidates": [{"row": r, "ik14": str(ik14[r]),
                                          "formula": str(formula[r]),
                                          "adduct": str(adduct[r]),
                                          "precursor_mz": float(pmz[r])}
                                         for r in cand_rows]})

    # IK14-level seeded sample
    by_ik = {}
    for q in valid:
        by_ik.setdefault(q["ik14"], []).append(q)
    iks = list(by_ik)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(iks)
    picked = []
    for ik in iks:
        if len(picked) >= a.target:
            break
        c = by_ik[ik]
        picked.append(c[int(rng.integers(0, len(c)))])

    picked = sorted(picked, key=lambda q: q["row"])
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    graph_obj = [{"row": q["row"], "cand_rows": [c["row"] for c in q["candidates"]]}
                 for q in picked]
    manifest = {
        "status": "g8r_final_test_manifest",
        "panel": "Test-C",
        "description": "next-generation blind test (peak token + listwise + peak RAW), disjoint from Test-A/B",
        "source": "HDF5 val fold",
        "n_queries": len(picked),
        "n_unique_ik14": len({q["ik14"] for q in picked}),
        "n_unique_formula": len({q["formula"] for q in picked}),
        "build_script_sha256": script_sha,
        "candidate_graph_sha256": hashlib.sha256(json.dumps(graph_obj, sort_keys=True).encode()).hexdigest(),
        "queries": picked,
    }
    manifest["query_manifest_sha256"] = hashlib.sha256(
        json.dumps({"queries": picked}, sort_keys=True).encode()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Test-C: n_queries={manifest['n_queries']} n_ik14={manifest['n_unique_ik14']} "
          f"n_formula={manifest['n_unique_formula']}")
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
