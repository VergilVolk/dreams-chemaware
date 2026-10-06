"""GLM Track-A2: spectral-entropy cross-check of the five strong dark-module hits.

Independent second metric (entropy similarity, Li et al. 2021) scored on the
same gated candidate sets, to confirm or qualify the greedy-cosine pilot.
Read-only; outputs a cross-check table beside the cosine identity table.
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
OUT = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_wse_crosscheck.csv")
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


def top_peaks(peaks, max_peaks=100):
    peaks = sorted(peaks, key=lambda p: -p[1])[:max_peaks]
    mzs = np.array([p[0] for p in peaks])
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    return mzs, ints / (ints.max() if ints.size else 1.0)


def entropy(a_mz, a_int, b_mz, b_int):
    """Entropy similarity with tolerance merging (plain intensity weights)."""
    merged_mz, merged_int = [], []
    for mz, val in zip(a_mz, a_int):
        merged_mz.append(float(mz)); merged_int.append(float(val))
    for mz, val in zip(b_mz, b_int):
        placed = False
        for k in range(len(merged_mz)):
            if abs(merged_mz[k] - float(mz)) <= TOL:
                merged_int[k] += float(val); placed = True; break
        if not placed:
            merged_mz.append(float(mz)); merged_int.append(float(val))
    weights = np.array(merged_int)
    weights = weights / weights.sum()
    h_mix = -float(np.sum(weights * np.log(weights + 1e-12)))

    def ent(ints):
        w = ints / ints.sum()
        return -float(np.sum(w * np.log(w + 1e-12)))

    denom = max(ent(a_int), ent(b_int))
    return 1.0 - h_mix / denom if denom > 0 else 0.0


def greedy(a_mz, a_int, b_mz, b_int):
    used = np.zeros(b_mz.size, dtype=bool); matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - TOL, mz + TOL))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]; used[best] = True
                matched += float(val * b_int[best])
    return matched / math.sqrt(float((a_int**2).sum()) * float((b_int**2).sum()) + 1e-12)


def main() -> None:
    pilot = json.load(open(PILOT, encoding="utf-8"))
    targets = {e["precursor"] for e in pilot["modules"]
               if max((n["score"] for n in e["neighbors"]), default=0) >= THRESHOLD}
    dark = [s for s in parse_mgf(DARK_MGF) if s.get("precursor") in targets]
    lib = parse_mgf(LIB_MGF)
    manifest = pd.read_csv(MANIFEST)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    order = np.argsort(lib_mz); sorted_mz = lib_mz[order]
    print(f"targets: {len(dark)}; library: {len(lib)}", flush=True)

    rows = []
    for query in dark:
        qmz = query["precursor"]
        q_mz, q_int = top_peaks(query["peaks"])
        lo = np.searchsorted(sorted_mz, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(sorted_mz, qmz * (1 + MZ_GATE))
        scored = []
        for idx in order[lo:hi]:
            l_mz, l_int = top_peaks(lib[idx]["peaks"])
            scored.append((greedy(q_mz, q_int, l_mz, l_int),
                           entropy(q_mz, q_int, l_mz, l_int), idx))
        scored.sort(key=lambda t: -t[1])  # rank by entropy similarity
        for cos, wse, idx in scored[:TOPK]:
            meta = manifest.iloc[idx]
            rows.append({"dark_precursor": qmz, "cosine": round(cos, 4),
                         "entropy_similarity": round(wse, 4),
                         "neighbor_formula": meta.get("formula"),
                         "neighbor_usi": str(meta.get("usi"))[-55:],
                         "delta_mz_da": round(qmz - float(meta.get("precursor_mz", qmz)), 4)})
    frame = pd.DataFrame(rows).sort_values(["dark_precursor", "entropy_similarity"],
                                           ascending=[True, False])
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False), flush=True)
    print(f"written: {OUT}", flush=True)


if __name__ == "__main__":
    main()
