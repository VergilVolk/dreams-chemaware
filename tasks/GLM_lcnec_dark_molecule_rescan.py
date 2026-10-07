"""Molecule-level reverse-search rescan of LCNEC priority dark modules.

v1 (GLM_lcnec_dark_full_rescan) stored only the top-3 SPECTRUM neighbors,
which cannot support the molecule-level top1-top2 gap that the confidence
gate is calibrated on. This v2 keeps, per module, the best spectrum score
per MOLECULE (ik14), so the gate quantity gap = best_molecule - second_
best_molecule is exact. Scoring core is imported unchanged from v1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_lcnec_dark_full_rescan import (  # noqa: E402
    DARK_MGF, HIT_THRESHOLD, LIB_MGF, MANIFEST, MZ_GATE, TOL, cosine,
    parse_mgf, spec_sorted,
)
from noise_gnps_article_spectral_scores import weighted_entropy_similarity  # noqa: E402

BASE = ROOT / "data/validation"
OUT_JSON = BASE / "GLM_track2_census" / "lcnec_dark_molecule_rescan.json"
OUT_CSV = BASE / "GLM_track2_census" / "lcnec_dark_molecule_rescan.csv"
TOP_MOLECULES = 5


def main() -> None:
    dark = parse_mgf(DARK_MGF)
    lib = parse_mgf(LIB_MGF)
    manifest = pd.read_csv(MANIFEST)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    order = np.argsort(lib_mz)
    sorted_mz = lib_mz[order]
    print(f"modules: {len(dark)}; library: {len(lib)}", flush=True)

    cache: dict[int, tuple] = {}

    def lib_spec(idx):
        if idx not in cache:
            cache[idx] = spec_sorted(lib[idx]["peaks"])
        return cache[idx]

    modules = []
    rows = []
    for query in dark:
        qmz = query["precursor"]
        q_mz, q_int = spec_sorted(query["peaks"])
        lo = np.searchsorted(sorted_mz, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(sorted_mz, qmz * (1 + MZ_GATE))
        best_by_mol: dict[str, dict] = {}
        for idx in order[lo:hi]:
            cos = cosine(q_mz, q_int, *lib_spec(idx))
            meta = manifest.iloc[idx]
            ik = str(meta.get("ik14"))
            if cos > best_by_mol.get(ik, {}).get("cos", -1.0):
                best_by_mol[ik] = {"cos": cos, "idx": int(idx),
                                   "formula": meta.get("formula"),
                                   "usi": str(meta.get("usi"))[-55:]}
        mols = sorted(best_by_mol.values(), key=lambda d: -d["cos"])
        top1 = mols[0]["cos"] if mols else 0.0
        top2 = mols[1]["cos"] if len(mols) > 1 else 0.0
        gap = top1 - top2
        # entropy cross-metric on the best molecule's best spectrum
        entropy = None
        if mols and top1 >= HIT_THRESHOLD:
            l_mz, l_int = lib_spec(mols[0]["idx"])
            entropy = round(float(weighted_entropy_similarity(
                np.vstack([q_mz, q_int]), np.vstack([l_mz, l_int]), TOL)), 4)
        entry = {
            "precursor": qmz,
            "n_candidates": int(hi - lo),
            "n_molecules": len(mols),
            "best_cosine": round(float(top1), 4),
            "second_cosine": round(float(top2), 4),
            "molecule_gap": round(float(gap), 4),
            "entropy_best": entropy,
            "top_molecules": [
                {"ik14": ik, "cosine": round(d["cos"], 4),
                 "formula": d["formula"], "usi_tail": d["usi"]}
                for ik, d in sorted(best_by_mol.items(),
                                    key=lambda kv: -kv[1]["cos"])[:TOP_MOLECULES]],
        }
        modules.append(entry)
        for ik, d in sorted(best_by_mol.items(),
                            key=lambda kv: -kv[1]["cos"])[:TOP_MOLECULES]:
            rows.append({"precursor": qmz, "ik14": ik,
                         "cosine": round(float(d["cos"]), 4),
                         "formula": d["formula"], "usi_tail": d["usi"]})
        print(f"  m/z {qmz:.4f}: top1 {top1:.3f} gap {gap:.3f} "
              f"n_mol {len(mols)}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"status": "GLM_LCNEC_DARK_MOLECULE_RESCAN",
         "note": ("molecule-level (ik14) collapse of the corrected v1 "
                  "reverse scan; molecule_gap = best - second molecule"),
         "modules": modules}, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"written: {OUT_JSON} / {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
