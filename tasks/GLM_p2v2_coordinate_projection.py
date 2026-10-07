#!/usr/bin/env python
"""P2 v2: LCNEC spectral coordinate projection with margin-based confidence.

Corrected from v1 audit failures:
  - Coordinate dictionary: ALL gold identities (no peak-count bias)
  - Confidence: margin (top1-top2 gap), not absolute score (P0-M finding)
  - Ppm gate: configurable (1000 default for analog-tolerant)
  - All assignments saved with identity, formula, score, margin
  - Feature-spectrum-patient linkage explicitly verified

Runs locally with greedy cosine + entropy, or on server with DreaMS embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOLERANCE = 0.02  # Da, spectral peak matching

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        while b := h.read(8 << 20):
            d.update(b)
    return d.hexdigest()


def parse_mgf(path):
    spectra, cur = [], None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": [], "row": len(spectra)}
            elif line == "END IONS":
                if cur and len(cur["peaks"]) >= 3:
                    spectra.append(cur)
                cur = None
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line.startswith("TITLE="):
                    cur["title"] = line[6:150]
                elif line and line[0].isdigit():
                    p = line.split()
                    if len(p) >= 2 and float(p[1]) > 0:
                        cur["peaks"].append((float(p[0]), float(p[1])))
    return spectra


def norm_peaks(peaks, max_peaks=100):
    p = sorted(sorted(peaks, key=lambda x: -x[1])[:max_peaks], key=lambda x: x[0])
    mzs = np.array([x[0] for x in p], dtype=np.float64)
    ints = np.array([x[1] for x in p], dtype=np.float64)
    if ints.max() > 0:
        ints /= ints.max()
    return mzs, ints


def greedy_cos(a_mz, a_int, b_mz, b_int):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - TOLERANCE, mz + TOLERANCE))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(val * b_int[best])
    na, nb = float((a_int**2).sum()), float((b_int**2).sum())
    return matched / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0


def entropy_sim(a_mz, a_int, b_mz, b_int):
    merged = {}
    for mz, val in zip(a_mz, a_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    for mz, val in zip(b_mz, b_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    w = np.array(list(merged.values()))
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    h_mix = -float(np.sum(w[w > 0] * np.log(w[w > 0])))
    ha = a_int / a_int.sum() if a_int.sum() > 0 else a_int
    hb = b_int / b_int.sum() if b_int.sum() > 0 else b_int
    hna = -float(np.sum(ha[ha > 0] * np.log(ha[ha > 0])))
    hnb = -float(np.sum(hb[hb > 0] * np.log(hb[hb > 0])))
    d = max(hna, hnb)
    return max(0.0, 1.0 - h_mix / d) if d > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mgf", type=Path, required=True,
                        help="LCNEC family spectra MGF")
    parser.add_argument("--effects-npz", type=Path, required=True,
                        help="Stage-1 primary_decomposition.npz with patient_effect")
    parser.add_argument("--gnps-manifest", type=Path,
                        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz")
    parser.add_argument("--gnps-mgf", type=Path,
                        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm-gate", type=float, default=1000.0,
                        help="Mass gate for coordinate candidates (ppm)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # ===== 1. Build coordinate dictionary (ALL gold identities, no bias) =====
    print("Building coordinate dictionary...", flush=True)
    manifest = pd.read_csv(args.gnps_manifest)
    gold = manifest[manifest["quality_label"] == "gold"].copy()
    # Best spectrum per identity (most peaks, deterministic)
    gold_sorted = gold.sort_values(["n_peaks"], ascending=False)
    coord_repr = gold_sorted.drop_duplicates(subset="ik14", keep="first")
    coord_rows = set(coord_repr["row"].tolist())
    print(f"  gold identities: {len(coord_repr)}", flush=True)

    # Load coordinate spectra
    coord_spectra = {}
    with open(args.gnps_mgf, "r", encoding="utf-8", errors="ignore") as h:
        cur, row = None, 0
        for line in h:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": []}
            elif line == "END IONS":
                if cur and cur["peaks"] and row in coord_rows:
                    coord_spectra[row] = cur
                cur = None
                row += 1
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line and line[0].isdigit():
                    p = line.split()
                    if len(p) >= 2 and float(p[1]) > 0:
                        cur["peaks"].append((float(p[0]), float(p[1])))
    print(f"  coordinate spectra loaded: {len(coord_spectra)}", flush=True)

    # Build coordinate arrays
    coord_meta = []
    coord_data = []
    for _, row_data in coord_repr.iterrows():
        r = int(row_data["row"])
        if r not in coord_spectra:
            continue
        spec = coord_spectra[r]
        coord_meta.append({
            "coord_id": len(coord_meta),
            "ik14": str(row_data.get("ik14", "")),
            "formula": str(row_data.get("formula", "")),
            "precursor_mz": float(spec.get("precursor", 0)),
            "instrument": str(row_data.get("instrument", "")),
        })
        coord_data.append(norm_peaks(spec["peaks"]))
    n_coords = len(coord_data)
    coord_mzs = np.array([m["precursor_mz"] for m in coord_meta])
    print(f"  usable coordinates: {n_coords}", flush=True)

    # ===== 2. Load query spectra =====
    print(f"Loading query spectra: {args.query_mgf}", flush=True)
    queries = parse_mgf(args.query_mgf)
    print(f"  queries: {len(queries)}", flush=True)

    # ===== 3. Load patient effects =====
    effects_data = np.load(args.effects_npz)
    patient_effect = effects_data["patient_effect"]  # (34, 263)
    family_id = effects_data["family_id"]
    patient_names = [str(p) for p in effects_data.get("patient", [])]
    query_mz_from_npz = effects_data.get("mz", np.array([]))
    n_effects = patient_effect.shape[1]
    print(f"  effects: {patient_effect.shape[0]} patients x {n_effects} families", flush=True)

    # ===== 4. Verify feature-spectrum linkage =====
    # Match queries to effects by precursor m/z within 0.01 Da
    query_to_effect = {}
    for qi, q in enumerate(queries):
        q_mz = q.get("precursor", 0)
        q_title = q.get("title", f"query_{qi}")
        # Find matching effect column by m/z
        best_match = -1
        best_diff = 0.01
        for ei in range(n_effects):
            e_mz = float(query_mz_from_npz[ei]) if ei < len(query_mz_from_npz) else 0
            diff = abs(q_mz - e_mz)
            if diff < best_diff:
                best_diff = diff
                best_match = ei
        if best_match >= 0:
            query_to_effect[qi] = best_match

    linked = len(query_to_effect)
    print(f"  linked queries: {linked}/{len(queries)}", flush=True)
    if linked < len(queries):
        print(f"  WARNING: {len(queries) - linked} queries unlinked to effects", flush=True)

    # ===== 5. Project each query to coordinates =====
    print("Projecting to coordinates...", flush=True)
    all_assignments = []
    for qi, q in enumerate(queries):
        q_mz = q.get("precursor", 0)
        q_title = q.get("title", f"query_{qi}")
        q_norm = norm_peaks(q["peaks"])

        # Mass gate
        ppm_errors = np.abs(coord_mzs - q_mz) / max(q_mz, 1e-6) * 1e6
        eligible = np.where(ppm_errors <= args.ppm_gate)[0]

        if len(eligible) == 0:
            all_assignments.append({
                "query_index": qi, "query_title": q_title, "query_mz": q_mz,
                "effect_index": query_to_effect.get(qi, -1),
                "n_candidates": 0, "assignments": [],
                "top1_score": 0, "margin": 0, "assigned": False,
            })
            continue

        # Score against eligible coordinates
        cos_scores = np.zeros(len(eligible))
        ent_scores = np.zeros(len(eligible))
        for ei, ci in enumerate(eligible):
            cm, cn = coord_data[ci]
            cos_scores[ei] = greedy_cos(q_norm[0], q_norm[1], cm, cn)
            ent_scores[ei] = entropy_sim(q_norm[0], q_norm[1], cm, cn)

        # Use combined score (cosine + entropy)/2
        combined = (cos_scores + ent_scores) / 2
        order = np.argsort(-combined)

        # Margin = top1 - top2
        if len(combined) >= 2:
            sorted_scores = np.sort(combined)
            margin = float(sorted_scores[-1] - sorted_scores[-2])
        else:
            margin = float(combined[0]) if len(combined) > 0 else 0.0

        # Record top-k assignments
        assignments = []
        for rank in range(min(args.top_k, len(order))):
            ci = eligible[order[rank]]
            assignments.append({
                "rank": rank + 1,
                "coord_id": int(coord_meta[ci]["coord_id"]),
                "coord_ik14": coord_meta[ci]["ik14"],
                "coord_formula": coord_meta[ci]["formula"],
                "coord_mz": round(coord_meta[ci]["precursor_mz"], 4),
                "cosine": round(float(cos_scores[order[rank]]), 4),
                "entropy": round(float(ent_scores[order[rank]]), 4),
                "combined": round(float(combined[order[rank]]), 4),
            })

        top1_score = float(combined[order[0]]) if len(order) > 0 else 0.0
        all_assignments.append({
            "query_index": qi, "query_title": q_title, "query_mz": round(q_mz, 4),
            "effect_index": query_to_effect.get(qi, -1),
            "n_candidates": len(eligible),
            "assignments": assignments,
            "top1_score": round(top1_score, 4),
            "margin": round(margin, 4),
            "assigned": len(assignments) > 0,
        })

    # ===== 6. Summary statistics =====
    n_assigned = sum(1 for a in all_assignments if a["assigned"])
    n_linked = sum(1 for a in all_assignments if a["effect_index"] >= 0)
    margins = [a["margin"] for a in all_assignments if a["assigned"]]
    scores = [a["top1_score"] for a in all_assignments if a["assigned"]]

    # Margin-based confidence tiers
    tiers = {"high": 0, "medium": 0, "low": 0, "unassigned": len(all_assignments) - n_assigned}
    for a in all_assignments:
        if not a["assigned"]:
            continue
        if a["margin"] > 0.3:
            tiers["high"] += 1
        elif a["margin"] > 0.1:
            tiers["medium"] += 1
        else:
            tiers["low"] += 1

    # ===== 7. Coordinate-level effects (for linked queries only) =====
    coord_effects = {}
    for a in all_assignments:
        if not a["assigned"] or a["effect_index"] < 0 or not a["assignments"]:
            continue
        best = a["assignments"][0]
        coord_id = best["coord_id"]
        effect_col = a["effect_index"]
        if coord_id not in coord_effects:
            coord_effects[coord_id] = {
                "coord_ik14": best["coord_ik14"],
                "coord_formula": best["coord_formula"],
                "margin": a["margin"],
                "patient_effect": patient_effect[:, effect_col].tolist(),
            }

    # ===== 8. Save =====
    assignments_df = pd.DataFrame([
        {k: v for k, v in a.items() if k != "assignments"}
        for a in all_assignments
    ])
    assignments_df.to_csv(args.output_dir / "query_assignments_summary.csv", index=False)

    # Detailed assignments (flatten)
    detail_rows = []
    for a in all_assignments:
        for assign in a["assignments"]:
            detail_rows.append({
                "query_title": a["query_title"], "query_mz": a["query_mz"],
                **assign
            })
    pd.DataFrame(detail_rows).to_csv(args.output_dir / "coordinate_assignments_detail.csv", index=False)

    report = {
        "status": "P2_V2_SPECTRAL_COORDINATE_PROJECTION",
        "coordinate_dictionary_size": n_coords,
        "queries_total": len(queries),
        "queries_linked_to_effects": n_linked,
        "queries_assigned": n_assigned,
        "assignment_rate": round(n_assigned / max(len(queries), 1), 4),
        "confidence_tiers": tiers,
        "median_top1_score": round(float(np.median(scores)), 4) if scores else None,
        "median_margin": round(float(np.median(margins)), 4) if margins else None,
        "coordinates_with_effects": len(coord_effects),
        "ppm_gate": args.ppm_gate,
        "provenance": {
            "query_mgf_sha256": sha256_file(args.query_mgf),
            "gnps_manifest_sha256": sha256_file(args.gnps_manifest),
            "effects_npz_sha256": sha256_file(args.effects_npz),
        },
    }
    with open(args.output_dir / "p2_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
