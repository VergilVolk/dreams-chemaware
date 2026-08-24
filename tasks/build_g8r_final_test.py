"""Build and seal TWO final test panels from the HDF5 original val fold.

Test-A (representative): sample valid queries at the IK14 level (one query per
  molecule, so multi-spectrum molecules do not dominate), seeded random, target
  1500-3000.  Selection uses NO model score and NO isomer difficulty.
Test-B (isomer challenge): sample the subset of valid queries that have >=1
  same-formula isomer negative, also at the IK14 level, target 1500-3000, for
  near/mid and isomer Top-1 evaluation.

Both panels share the full 45,185-spectrum val-fold candidate library.  For each
selected query the manifest stores the FULL candidate list (row + IK14 + formula
+ adduct + precursor_mz) plus three hashes (candidate-graph, build-script,
query-manifest).  Only count / missing-field / candidate-coverage checks are
allowed afterwards; never model quality.
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
DEFAULT_OUT = ROOT / "data/validation/g8r_final_test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260822)
    return p.parse_args()


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def sha256_of(obj) -> str:
    raw = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    a = parse_args()
    with h5py.File(a.data, "r") as h:
        fold = read_str(h, "fold")
        ikf = read_str(h, "INCHIKEY")
        formula = read_str(h, "FORMULA")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)
        inst = read_str(h, "INSTRUMENT_TYPE")
        ce = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)

    val_idx = np.where(fold == "val")[0]
    ik14 = np.asarray([k[:14] for k in ikf], dtype=object)
    va_ik14 = ik14[val_idx]
    va_formula = formula[val_idx]
    va_adduct = adduct[val_idx]
    va_pmz = pmz[val_idx]
    va_inst = inst[val_idx]
    va_ce = ce[val_idx]

    # Build candidate graph per adduct via sorted m/z.
    valid = []   # list of dict: {row, ik14, formula, adduct, pmz, inst, ce, iso, candidates}
    adducts = np.unique(va_adduct)
    for ad in adducts:
        g = np.where(va_adduct == ad)[0]
        m = va_pmz[g]
        order = np.argsort(m)
        gs = g[order]; ms = m[order]
        ik_s = va_ik14[gs]; f_s = va_formula[gs]
        for pos in range(len(gs)):
            qrow = int(val_idx[gs[pos]])
            qm = ms[pos]; qik = ik_s[pos]; qf = f_s[pos]
            lo = np.searchsorted(ms, qm - a.ppm_tol * 1e-6 * qm, side="left")
            hi = np.searchsorted(ms, qm + a.ppm_tol * 1e-6 * qm, side="right")
            cand = np.arange(lo, hi)
            cand = cand[cand != pos]
            if len(cand) == 0:
                continue
            cik = ik_s[cand]; cf = f_s[cand]
            if not (cik == qik).any() or not (cik != qik).any():
                continue
            iso = int((cf[cik != qik] == qf).sum())
            # candidate list: row -> ik14/formula (dedup not applied; eval does max-score)
            cand_rows = [int(val_idx[gs[j]]) for j in cand]
            valid.append({
                "row": qrow, "ik14": str(qik), "formula": str(qf),
                "adduct": str(ad), "precursor_mz": float(qm),
                "instrument": str(va_inst[gs[pos]]), "ce_finite": bool(np.isfinite(va_ce[gs[pos]])),
                "n_isomer_neg": iso,
                "candidates": [
                    {"row": r, "ik14": str(ik14[r]), "formula": str(formula[r]),
                     "adduct": str(adduct[r]), "precursor_mz": float(pmz[r])}
                    for r in cand_rows
                ],
            })

    iso_queries = [q for q in valid if q["n_isomer_neg"] >= 1]
    print(f"valid queries total={len(valid)} isomer={len(iso_queries)}")

    def sample_ik14_level(queries, target, rng):
        """One query per IK14, seeded random, up to target."""
        by_ik = {}
        for q in queries:
            by_ik.setdefault(q["ik14"], []).append(q)
        iks = list(by_ik)
        rng.shuffle(iks)
        picked = []
        for ik in iks:
            if len(picked) >= target:
                break
            cand = by_ik[ik]
            picked.append(cand[int(rng.integers(0, len(cand)))])
        return picked

    rng = np.random.default_rng(a.seed)
    test_a = sample_ik14_level(valid, a.target, rng)
    rng_b = np.random.default_rng(a.seed + 1)
    test_b = sample_ik14_level(iso_queries, a.target, rng_b)

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def manifest(queries, panel, desc):
        queries = sorted(queries, key=lambda q: q["row"])
        graph_obj = [{"row": q["row"], "cand_rows": [c["row"] for c in q["candidates"]]}
                     for q in queries]
        body = {
            "panel": panel, "description": desc,
            "n_queries": len(queries),
            "n_unique_ik14": len({q["ik14"] for q in queries}),
            "n_unique_formula": len({q["formula"] for q in queries}),
            "n_with_isomer_neg": sum(1 for q in queries if q["n_isomer_neg"] >= 1),
            "queries": queries,
        }
        m = {
            "status": "g8r_final_test_manifest",
            "source": "HDF5 val fold", "protocol": "strict-10ppm same-adduct",
            "build_script_sha256": script_sha,
            "candidate_graph_sha256": sha256_of(graph_obj),
            "query_manifest_sha256": sha256_of(body),
            **{k: body[k] for k in ("panel", "description", "n_queries",
                                    "n_unique_ik14", "n_unique_formula", "n_with_isomer_neg")},
            "queries": queries,
        }
        return m

    a.output_dir.mkdir(parents=True, exist_ok=True)
    ma = manifest(test_a, "Test-A", "representative overall retrieval (IK14-level seeded sample)")
    mb = manifest(test_b, "Test-B", "isomer challenge (same-formula negative present)")
    (a.output_dir / "test_a_manifest.json").write_text(json.dumps(ma, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output_dir / "test_b_manifest.json").write_text(json.dumps(mb, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, m in (("Test-A", ma), ("Test-B", mb)):
        print(f"{name}: n_queries={m['n_queries']} n_ik14={m['n_unique_ik14']} "
              f"n_formula={m['n_unique_formula']} n_isomer={m['n_with_isomer_neg']}")
        print(f"  candidate_graph_sha256={m['candidate_graph_sha256'][:16]}... "
              f"query_sha256={m['query_manifest_sha256'][:16]}...")
    print(f"Saved to {a.output_dir}")


if __name__ == "__main__":
    main()
