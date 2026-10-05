"""GLM Track-A2: resolve identities of the strong LCNEC dark-module neighbors.

Re-runs the gated cosine search for the modules with best-cosine >= 0.85 in
the pilot, tracks library row indices, and joins the benchmark manifest to
produce an actionable candidate/standard-purchase table.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

BASE = os.path.join("data", "validation")
DARK_MGF = os.path.join(BASE, "lcnec_hsst3n_priority_ms2", "priority_dark_modules.mgf")
LIB_MGF = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1", "spectra.mgf")
MANIFEST = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1", "manifest.csv.gz")
PILOT = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_reverse_search_pilot.json")
OUT = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_neighbor_identities.csv")
MZ_GATE, THRESHOLD = 0.007, 0.85


def parse_mgf_indexed(path):
    spectra, cur = [], None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": [], "row": len(spectra)}
            elif line == "END IONS":
                if cur and len(cur["peaks"]) >= 5:
                    spectra.append(cur)
                cur = None
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2 and float(parts[1]) > 0:
                        cur["peaks"].append((float(parts[0]), float(parts[1])))
    return spectra


def norm(peaks, max_peaks=100):
    peaks = sorted(peaks, key=lambda p: -p[1])[:max_peaks]
    mzs = np.array([p[0] for p in peaks])
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    return mzs, ints / (ints.max() if ints.size else 1.0)


def greedy(a_mz, a_int, b_mz, b_int, tol=0.02):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, inten in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - tol, mz + tol))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(inten * b_int[best])
    return matched / math.sqrt(float((a_int ** 2).sum()) * float((b_int ** 2).sum()) + 1e-12)


def main() -> None:
    pilot = json.load(open(PILOT, encoding="utf-8"))
    targets = {e["precursor"] for e in pilot["modules"]
               if max((n["score"] for n in e["neighbors"]), default=0) >= THRESHOLD}
    dark = [s for s in parse_mgf_indexed(DARK_MGF) if s.get("precursor") in targets]
    lib = parse_mgf_indexed(LIB_MGF)
    manifest = pd.read_csv(MANIFEST)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    order = np.argsort(lib_mz)
    sorted_mz = lib_mz[order]
    print(f"targets: {len(dark)} modules; library: {len(lib)}; "
          f"manifest rows: {len(manifest)}", flush=True)

    rows = []
    for query in dark:
        qmz = query["precursor"]
        q_mz, q_int = norm(query["peaks"])
        lo = np.searchsorted(sorted_mz, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(sorted_mz, qmz * (1 + MZ_GATE))
        scored = []
        for idx in order[lo:hi]:
            l_mz, l_int = norm(lib[idx]["peaks"])
            scored.append((greedy(q_mz, q_int, l_mz, l_int), idx))
        scored.sort(reverse=True)
        for score, idx in scored[:3]:
            meta = manifest.iloc[idx] if idx < len(manifest) else {}
            rows.append({
                "dark_precursor": qmz,
                "cosine": round(float(score), 4),
                "neighbor_formula": meta.get("formula"),
                "neighbor_ik14": meta.get("ik14"),
                "neighbor_name_field": str(meta.get("usi", ""))[-60:],
                "neighbor_usi": meta.get("usi"),
                "quality": meta.get("quality_label"),
                "instrument": meta.get("instrument"),
                "delta_mz_da": round(qmz - float(meta.get("precursor_mz", qmz)), 4),
            })
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["dark_precursor", "neighbor_usi"]).sort_values(
        ["dark_precursor", "cosine"], ascending=[True, False])
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False), flush=True)
    print(f"written: {OUT}", flush=True)


if __name__ == "__main__":
    main()
