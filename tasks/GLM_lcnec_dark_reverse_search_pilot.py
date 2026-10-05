"""GLM Track-A2 pilot: LCNEC dark-module reverse search against the local
GNPS gold/silver library spectra (52,871 panel spectra, sealed run artifact).

Read-only pilot: greedy-cosine neighborhood scan with a precursor gate.
Output: per-module top neighbors + a ranked hit table for the PI's review.
Analog-tolerant gate (0.7%) so class-level neighbors are not excluded.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

DARK_MGF = os.path.join("data", "validation", "lcnec_hsst3n_priority_ms2",
                        "priority_dark_modules.mgf")
LIB_MGF = os.path.join("data", "validation", "gnps_gold_silver_10ppm_benchmark_v1",
                       "spectra.mgf")
OUT_DIR = os.path.join("data", "validation", "GLM_track2_census")
OUT_JSON = os.path.join(OUT_DIR, "lcnec_dark_reverse_search_pilot.json")
OUT_CSV = os.path.join(OUT_DIR, "lcnec_dark_reverse_search_pilot.csv")
MZ_GATE = 0.007          # relative precursor gate (analog-tolerant)
TOPK = 5
MIN_PEAKS = 5


def parse_mgf(path: str, limit: int | None = None) -> list[dict]:
    spectra, cur, mode = [], None, None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur, mode = {"peaks": []}, None
            elif line == "END IONS":
                if cur and cur["peaks"] and len(cur["peaks"]) >= MIN_PEAKS:
                    spectra.append(cur)
                cur = None
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line.startswith("TITLE="):
                    cur["title"] = line[6:120]
                elif line.startswith("SCANS="):
                    cur["scan"] = line.split("=")[1][:40]
                elif line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        mz, inten = float(parts[0]), float(parts[1])
                        if inten > 0:
                            cur["peaks"].append((mz, inten))
                if limit and len(spectra) >= limit:
                    break
    return spectra


def norm_spectrum(peaks: list[tuple[float, float]], max_peaks: int = 100):
    peaks = sorted(peaks, key=lambda p: -p[1])[:max_peaks]
    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    ints = ints / ints.max() if ints.size else ints
    return mzs, ints


def greedy_cosine(a_mz, a_int, b_mz, b_int, tol: float = 0.02):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, inten in zip(a_mz, a_int):
        pos = np.searchsorted(b_mz, (mz - tol, mz + tol))
        lo, hi = pos[0], pos[1]
        if hi > lo:
            candidates = np.flatnonzero(~used[lo:hi]) + lo
            if candidates.size:
                best = candidates[np.argmax(b_int[candidates])]
                used[best] = True
                matched += float(inten * b_int[best])
    return matched / math.sqrt(float((a_int ** 2).sum()) * float((b_int ** 2).sum()) + 1e-12)


def main() -> None:
    dark = parse_mgf(DARK_MGF)
    lib = parse_mgf(LIB_MGF)
    lib_mz = np.array([s.get("precursor", 0.0) for s in lib])
    lib_order = np.argsort(lib_mz)
    lib_mz_sorted = lib_mz[lib_order]
    print(f"dark modules: {len(dark)}; library spectra: {len(lib)}", flush=True)

    rows, report = [], []
    for query in dark:
        qmz = query.get("precursor", 0.0)
        q_mz, q_int = norm_spectrum(query["peaks"])
        lo = np.searchsorted(lib_mz_sorted, qmz * (1 - MZ_GATE))
        hi = np.searchsorted(lib_mz_sorted, qmz * (1 + MZ_GATE))
        candidates = lib_order[lo:hi]
        scored = []
        for idx in candidates:
            l_mz, l_int = norm_spectrum(lib[idx]["peaks"])
            score = greedy_cosine(q_mz, q_int, l_mz, l_int)
            scored.append((score, idx))
        scored.sort(reverse=True)
        entry = {"precursor": qmz, "title": query.get("title", ""),
                 "n_candidates_in_gate": int(candidates.size), "neighbors": []}
        for score, idx in scored[:TOPK]:
            neighbor = {"score": round(float(score), 4),
                        "precursor": lib[idx].get("precursor"),
                        "title": lib[idx].get("title", "")[:90]}
            entry["neighbors"].append(neighbor)
            rows.append({"query_precursor": qmz,
                         "query_title": query.get("title", "")[:80],
                         "neighbor_precursor": neighbor["precursor"],
                         "cosine": neighbor["score"],
                         "neighbor_title": neighbor["title"]})
        report.append(entry)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump({"status": "GLM_LCNEC_DARK_REVERSE_SEARCH_PILOT",
                   "library": "gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf",
                   "gate": MZ_GATE, "modules": report}, handle, indent=2)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    best = [max((n["score"] for n in e["neighbors"]), default=0.0) for e in report]
    print(f"modules searched: {len(report)}; median best-cosine: "
          f"{float(np.median(best)):.3f}; top-decile best: "
          f"{float(np.quantile(best, 0.9)):.3f}", flush=True)
    print(f"written: {OUT_JSON} and {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
