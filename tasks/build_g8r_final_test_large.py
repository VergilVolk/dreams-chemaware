"""Build a LARGE, balanced final test (Test-A/B/C) from the FULL HDF5.

Uses the entire HDF5 (231,104 spectra) as the candidate library, EXCLUDING every
IK14 used by g8r_train/g8r_val (zero IK14 overlap).  A query is VALID iff it has
>=1 same-IK14 (positive) and >=1 different-IK14 (negative) candidate within
strict-10ppm same-adduct.

Panels are assigned by IK14 (all spectra of a molecule go to one panel, so no
molecule leaks across panels), shuffled, split 1/3-1/3-1/3 into Test-A
(representative), Test-B (isomer-heavy), Test-C (next-generation reserve).
Manifests are compact CSV (query rows + metadata) plus a JSON with hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_OUT = ROOT / "data/validation/g8r_final_test_large"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260823)
    return p.parse_args()


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def main() -> None:
    a = parse_args()
    # g8r IK14 (exclude)
    g8r_ik = set()
    for p in (a.train, a.val):
        for e in json.loads(p.read_text(encoding="utf-8"))["entries"]:
            g8r_ik.add(e["ik14"])

    with h5py.File(a.data, "r") as h:
        ikf = read_str(h, "INCHIKEY")
        formula = read_str(h, "FORMULA")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)

    n = len(pmz)
    ik14 = np.asarray([k[:14] for k in ikf], dtype=object)
    keep = np.array([ik not in g8r_ik for ik in ik14], dtype=bool)
    idx = np.where(keep)[0]
    print(f"total spectra={n} excluded_g8r={n - len(idx)} usable={len(idx)}")

    # build candidate graph per adduct (valid queries)
    valid_rows = []
    for ad in np.unique(adduct[idx]):
        g = idx[adduct[idx] == ad]
        m = pmz[g]
        order = np.argsort(m)
        gs = g[order]; ms = m[order]; ik_s = ik14[gs]
        for pos in range(len(gs)):
            qrow = int(gs[pos]); qm = ms[pos]; qik = ik_s[pos]
            lo = np.searchsorted(ms, qm - a.ppm_tol * 1e-6 * qm, side="left")
            hi = np.searchsorted(ms, qm + a.ppm_tol * 1e-6 * qm, side="right")
            cand = np.arange(lo, hi); cand = cand[cand != pos]
            if len(cand) == 0:
                continue
            cik = ik_s[cand]
            if not (cik == qik).any() or not (cik != qik).any():
                continue
            valid_rows.append(qrow)

    valid_rows = np.asarray(valid_rows, dtype=np.int64)
    print(f"valid queries (spectra) = {len(valid_rows)}")

    # assign by IK14 -> panel, shuffled, 1/3 each
    iks = np.unique(ik14[valid_rows])
    rng = np.random.default_rng(a.seed)
    rng.shuffle(iks)
    third = len(iks) // 3
    panel_of_ik = {}
    for i, ik in enumerate(iks):
        panel_of_ik[ik] = "A" if i < third else ("B" if i < 2 * third else "C")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = {}
    for panel in ("A", "B", "C"):
        rows = valid_rows[np.array([panel_of_ik[ik14[r]] == panel for r in valid_rows], dtype=bool)]
        rows = np.sort(rows)
        df = pd.DataFrame({
            "row": rows,
            "ik14": [str(ik14[r]) for r in rows],
            "formula": [str(formula[r]) for r in rows],
            "adduct": [str(adduct[r]) for r in rows],
            "precursor_mz": [float(pmz[r]) for r in rows],
        })
        csv = df.to_csv(index=False)
        m = {
            "panel": f"Test-{panel}",
            "n_queries": int(len(rows)),
            "n_unique_ik14": int(df["ik14"].nunique()),
            "n_unique_formula": int(df["formula"].nunique()),
            "build_script_sha256": script_sha,
            "query_manifest_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        }
        out[panel] = m
        (a.output_dir / f"test_{panel.lower()}_manifest.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        (a.output_dir / f"test_{panel.lower()}_queries.csv").write_text(csv, encoding="utf-8")
        print(f"Test-{panel}: n_queries={m['n_queries']} n_ik14={m['n_unique_ik14']} "
              f"n_formula={m['n_unique_formula']} sha={m['query_manifest_sha256'][:16]}")

    print(f"Saved to {a.output_dir}")


if __name__ == "__main__":
    main()
