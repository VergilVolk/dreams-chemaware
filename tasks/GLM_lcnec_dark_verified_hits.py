"""GLM Track-A2 v2: correct re-verification of the five strong dark-module hits.

Fixes the v1 crosscheck: (a) spectra are m/z-sorted before any searchsorted
call (the v1 cosine pilot used intensity-ordered mz arrays — this re-check
validates or refutes it), and (b) entropy similarity now uses the pinned
reference backend imported from noise_gnps_article_spectral_scores.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join("tasks"))
from noise_gnps_article_spectral_scores import weighted_entropy_similarity  # noqa: E402

BASE = os.path.join("data", "validation")
DARK_MGF = os.path.join(BASE, "lcnec_hsst3n_priority_ms2", "priority_dark_modules.mgf")
LIB_MGF = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1", "spectra.mgf")
MANIFEST = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1", "manifest.csv.gz")
PILOT = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_reverse_search_pilot.json")
OUT = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_verified_hits.csv")
MZ_GATE, THRESHOLD, TOPK, TOL = 0.007, 0.85, 3, 0.02


def parse_mgf(path):
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


def spec_mz_sorted(peaks, max_peaks=100):
    """Top-intensity slice, then m/z-ascending order for searchsorted."""
    peaks = sorted(peaks, key=lambda p: -p[1])[:max_peaks]
    peaks = sorted(peaks, key=lambda p: p[0])
    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    return mzs, ints / (ints.max() if ints.size else 1.0)


def greedy_cosine_sorted(a_mz, a_int, b_mz, b_int):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - TOL, mz + TOL))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(val * b_int[best])
    return matched / math.sqrt(float((a_int ** 2).sum()) * float((b_int ** 2).sum()) + 1e-12)


def as_backend_array(mzs, ints):
    return np.vstack([mzs, ints])


def main() -> None:
    pilot = json.load(open(PILOT, encoding="utf-8"))
    targets = {e["precursor"] for e in pilot["modules"]
               if max((n["score"] for n in e["neighbors"]), default=0) >= THRESHOLD}
    dark = [s for s in parse_mgf(DARK_MGF) if s.get("precursor") in targets]
    lib = parse_mgf(LIB_MGF)
    manifest = pd.read_csv(MANIFEST)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    order = np.argsort(lib_mz)
    sorted_mz = lib_mz[order]
    print(f"targets: {len(dark)}; library: {len(lib)}", flush=True)

    cache = {}

    def lib_spec(idx):
        if idx not in cache:
            cache[idx] = spec_mz_sorted(lib[idx]["peaks"])
        return cache[idx]

    rows = []
    for query in dark:
        qmz = query["precursor"]
        q_mz, q_int = spec_mz_sorted(query["peaks"])
        lo = np.searchsorted(sorted_mz, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(sorted_mz, qmz * (1 + MZ_GATE))
        scored = []
        for idx in order[lo:hi]:
            l_mz, l_int = lib_spec(idx)
            cos = greedy_cosine_sorted(q_mz, q_int, l_mz, l_int)
            wse = weighted_entropy_similarity(as_backend_array(q_mz, q_int),
                                              as_backend_array(l_mz, l_int), TOL)
            scored.append((cos, wse, idx))
        scored.sort(key=lambda t: -(t[0] + t[1]))
        for cos, wse, idx in scored[:TOPK]:
            meta = manifest.iloc[idx]
            rows.append({"dark_precursor": qmz,
                         "cosine_sorted": round(cos, 4),
                         "entropy_pinned": round(float(wse), 4),
                         "pilot_cosine_reported": max(
                             (n["score"] for n in next(
                                 e["neighbors"] for e in pilot["modules"]
                                 if e["precursor"] == qmz)), default=0),
                         "neighbor_formula": meta.get("formula"),
                         "neighbor_ik14": meta.get("ik14"),
                         "delta_mz_da": round(qmz - float(meta.get("precursor_mz", qmz)), 4),
                         "neighbor_usi": str(meta.get("usi"))[-55:]})
    frame = pd.DataFrame(rows).sort_values(["dark_precursor", "cosine_sorted"],
                                           ascending=[True, False])
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False), flush=True)
    print(f"written: {OUT}", flush=True)


if __name__ == "__main__":
    main()
