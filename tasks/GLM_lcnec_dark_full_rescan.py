"""GLM Track-A2 v3: full corrected re-scan of all LCNEC priority dark modules.

Replaces the corrupted v1 pilot (unsorted-mz searchsorted bug). Corrected
greedy cosine for every module x gated library candidates; pinned-backend
entropy similarity added only for modules whose best cosine >= 0.85.
This is the authoritative reverse-search table.
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
OUT_JSON = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_full_rescan.json")
OUT_CSV = os.path.join(BASE, "GLM_track2_census", "lcnec_dark_full_rescan.csv")
MZ_GATE, HIT_THRESHOLD, TOPK, TOL = 0.007, 0.85, 3, 0.02


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


def spec_sorted(peaks, max_peaks=100):
    peaks = sorted(sorted(peaks, key=lambda p: -p[1])[:max_peaks], key=lambda p: p[0])
    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    return mzs, ints / (ints.max() if ints.size else 1.0)


def cosine(a_mz, a_int, b_mz, b_int):
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


def main() -> None:
    dark = parse_mgf(DARK_MGF)
    lib = parse_mgf(LIB_MGF)
    manifest = pd.read_csv(MANIFEST)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    order = np.argsort(lib_mz)
    sorted_mz = lib_mz[order]
    print(f"modules: {len(dark)}; library: {len(lib)}", flush=True)

    cache = {}

    def lib_spec(idx):
        if idx not in cache:
            cache[idx] = spec_sorted(lib[idx]["peaks"])
        return cache[idx]

    report, rows = [], []
    for query in dark:
        qmz = query["precursor"]
        q_mz, q_int = spec_sorted(query["peaks"])
        lo = np.searchsorted(sorted_mz, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(sorted_mz, qmz * (1 + MZ_GATE))
        scored = []
        for idx in order[lo:hi]:
            l_mz, l_int = lib_spec(idx)
            scored.append((cosine(q_mz, q_int, l_mz, l_int), idx))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        entry = {"precursor": qmz, "n_candidates": int(hi - lo),
                 "best_cosine": round(best, 4), "neighbors": []}
        for cos, idx in scored[:TOPK]:
            meta = manifest.iloc[idx]
            wse = None
            if best >= HIT_THRESHOLD:
                l_mz, l_int = lib_spec(idx)
                wse = round(float(weighted_entropy_similarity(
                    np.vstack([q_mz, q_int]), np.vstack([l_mz, l_int]), TOL)), 4)
            entry["neighbors"].append({"cosine": round(float(cos), 4), "entropy": wse,
                                       "formula": meta.get("formula"),
                                       "ik14": meta.get("ik14"),
                                       "delta_mz": round(qmz - float(meta.get("precursor_mz", qmz)), 4),
                                       "usi_tail": str(meta.get("usi"))[-55:]})
            rows.append({"precursor": qmz, "cosine": round(float(cos), 4),
                         "entropy": wse, "formula": meta.get("formula"),
                         "ik14": meta.get("ik14"),
                         "delta_mz": round(qmz - float(meta.get("precursor_mz", qmz)), 4)})
        report.append(entry)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump({"status": "GLM_LCNEC_DARK_FULL_RESCAN_CORRECTED",
                   "note": "authoritative; supersedes lcnec_dark_reverse_search_pilot",
                   "modules": report}, handle, indent=2)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    bests = [e["best_cosine"] for e in report]
    hits = sum(1 for b in bests if b >= HIT_THRESHOLD)
    print(f"modules: {len(report)}; median best-cosine: {float(np.median(bests)):.3f}; "
          f"hits >= {HIT_THRESHOLD}: {hits}", flush=True)
    print(f"written: {OUT_JSON} / {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
