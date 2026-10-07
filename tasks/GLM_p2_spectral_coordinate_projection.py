"""RETIRED pilot: LCNEC spectral-coordinate projection.

Run 2353437 showed that this implementation is not a valid P2 experiment. It
is retained only to reproduce that pilot and now fails closed unless the
explicit legacy flag is supplied. See
docs/LCNEC_P2_SPECTRAL_COORDINATE_RUN_2353437_AUDIT.md.

Paradigm shift from family-graph decomposition to fixed-coordinate mapping:
  1. Build a frozen coordinate dictionary from GNPS gold spectra.
  2. Project HSST3n's 263 QC families onto coordinates (spectral similarity).
  3. Project LIPn's features onto the SAME coordinates.
  4. Compute coordinate-level patient effects in each platform.
  5. Measure cross-platform effect consistency at the coordinate level.

This eliminates the graph instability problem entirely: both platforms
share the same fixed coordinate system, so cross-platform comparison
is direct projection comparison, not graph alignment.

Usage:
  Server: needs Stage-1 query MGF + HSST3n EIC matrix + LIPn EIC matrix
  Local pilot: works with 30 dark module MGF + effects from CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TOLERANCE = 0.02
BASE = Path("data/validation")


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
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
                if cur and len(cur["peaks"]) >= 5:
                    spectra.append(cur)
                cur = None
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line.startswith("TITLE="):
                    cur["title"] = line[6:120]
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


# ============================================================================
# Coordinate dictionary builder
# ============================================================================

def build_coordinate_dictionary(
    gnps_manifest: pd.DataFrame,
    gnps_spectra: list[dict],
    max_coordinates: int = 2000,
    min_peaks: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Build frozen coordinate dictionary from gold-quality GNPS spectra.

    Returns (coordinates, coord_meta) where each coordinate is a dict with
    precursor, peaks, ik14, formula, etc.
    """
    gold = gnps_manifest[gnps_manifest["quality_label"] == "gold"].copy()
    gold_sorted = gold.sort_values("n_peaks", ascending=False)
    # One per identity (ik14), prefer spectra with more peaks
    deduped = gold_sorted.drop_duplicates(subset="ik14", keep="first")

    # Build coordinate spectra
    needed_rows = set(deduped["row"].tolist())
    coord_specs = {s["row"]: s for s in gnps_spectra if s["row"] in needed_rows}

    coordinates, coord_meta = [], []
    for _, row in deduped.iterrows():
        r = int(row["row"])
        if r not in coord_specs:
            continue
        spec = coord_specs[r]
        if len(spec["peaks"]) < min_peaks:
            continue
        coordinates.append(spec)
        coord_meta.append({
            "ik14": str(row.get("ik14", "")),
            "formula": str(row.get("formula", "")),
            "precursor_mz": float(spec.get("precursor", 0)),
            "n_peaks": len(spec["peaks"]),
            "instrument": str(row.get("instrument", "")),
        })
        if len(coordinates) >= max_coordinates:
            break

    return coordinates, coord_meta


# ============================================================================
# Query-to-coordinate projection
# ============================================================================

def project_queries_to_coordinates(
    query_spectra: list[dict],
    coordinates: list[dict],
    method: str = "cosine",
    ppm_gate: float = 500.0,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> dict[str, list[dict]]:
    """Project each query spectrum to its top-k coordinate assignments.

    Returns dict: query_title -> list of {coord_index, score}
    Only assignments above score_threshold and within ppm_gate are returned.
    """
    # Pre-normalize coordinate peaks
    coord_norm = [norm_peaks(c["peaks"]) for c in coordinates]
    coord_mzs = np.array([c.get("precursor", 0) for c in coordinates])

    assignments = {}
    for qi, qspec in enumerate(query_spectra):
        q_title = qspec.get("title", f"query_{qi}")
        q_mz = qspec.get("precursor", 0)
        q_norm = norm_peaks(qspec["peaks"])

        # Mass gate
        ppm_errors = np.abs(coord_mzs - q_mz) / max(q_mz, 1e-6) * 1e6
        eligible = np.where(ppm_errors <= ppm_gate)[0]

        if len(eligible) == 0:
            assignments[q_title] = []
            continue

        scores = np.zeros(len(eligible))
        for ei, ci in enumerate(eligible):
            cm, ci_norm = coord_mzs[ci], coord_norm[ci]
            if method == "cosine":
                scores[ei] = greedy_cos(q_norm[0], q_norm[1], ci_norm[0], ci_norm[1])
            elif method == "entropy":
                scores[ei] = entropy_sim(q_norm[0], q_norm[1], ci_norm[0], ci_norm[1])
            else:  # both (average)
                c = greedy_cos(q_norm[0], q_norm[1], ci_norm[0], ci_norm[1])
                e = entropy_sim(q_norm[0], q_norm[1], ci_norm[0], ci_norm[1])
                scores[ei] = (c + e) / 2

        # Top-k above threshold
        order = np.argsort(-scores)
        result = []
        for rank_i in order[:top_k]:
            if scores[rank_i] >= score_threshold:
                result.append({
                    "coord_index": int(eligible[rank_i]),
                    "score": round(float(scores[rank_i]), 4),
                })
        assignments[q_title] = result

    return assignments


# ============================================================================
# Coordinate-level patient effects
# ============================================================================

def coordinate_effects(
    effect_matrix: np.ndarray,  # (n_patients, n_features)
    feature_titles: list[str],
    assignments: dict[str, list[dict]],
    n_coordinates: int,
    min_features_per_coord: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate feature-level effects to coordinate level.

    For each coordinate, take the max-score assigned feature's effect.
    Returns (coord_effects, coord_coverage) arrays of shape (n_patients, n_coords).
    """
    n_patients = effect_matrix.shape[0]
    coord_effects = np.full((n_patients, n_coordinates), np.nan)
    coord_coverage = np.zeros(n_coordinates)

    # For each coordinate, find the best-assigned feature
    coord_best_feature = {}
    for fi, title in enumerate(feature_titles):
        for assign in assignments.get(title, []):
            ci = assign["coord_index"]
            score = assign["score"]
            if ci not in coord_best_feature or score > coord_best_feature[ci][1]:
                coord_best_feature[ci] = (fi, score)

    for ci, (fi, score) in coord_best_feature.items():
        coord_effects[:, ci] = effect_matrix[:, fi]
        coord_coverage[ci] = 1

    # Coordinates with no assignment remain NaN
    return coord_effects, coord_coverage


# ============================================================================
# Cross-platform consistency
# ============================================================================

def cross_platform_consistency(
    effects_a: np.ndarray,  # (n_patients, n_coords) platform A
    effects_b: np.ndarray,  # (n_patients, n_coords) platform B
    patient_ids_a: list[str],
    patient_ids_b: list[str],
) -> dict:
    """Measure coordinate-level effect consistency across two platforms."""
    # Match patients
    common = sorted(set(patient_ids_a) & set(patient_ids_b))
    if len(common) < 10:
        return {"error": f"too few common patients: {len(common)}"}

    idx_a = [patient_ids_a.index(p) for p in common]
    idx_b = [patient_ids_b.index(p) for p in common]
    ea = effects_a[idx_a]
    eb = effects_b[idx_b]

    # For each coordinate with data in both platforms
    valid = ~np.isnan(ea).any(axis=0) & ~np.isnan(eb).any(axis=0)
    n_valid = int(valid.sum())

    if n_valid < 5:
        return {"error": f"too few shared coordinates: {n_valid}"}

    ea_valid = ea[:, valid]
    eb_valid = eb[:, valid]

    # Per-coordinate Pearson correlation across patients
    correlations = []
    for ci in range(n_valid):
        r = np.corrcoef(ea_valid[:, ci], eb_valid[:, ci])[0, 1]
        if np.isfinite(r):
            correlations.append(r)
    correlations = np.array(correlations)

    # Per-coordinate direction agreement (sign of median effect)
    sign_a = np.sign(np.median(ea_valid, axis=0))
    sign_b = np.sign(np.median(eb_valid, axis=0))
    direction_agree = np.mean(sign_a == sign_b)

    # Overall correlation of median effects
    median_a = np.median(ea_valid, axis=0)
    median_b = np.median(eb_valid, axis=0)
    overall_r = np.corrcoef(median_a, median_b)[0, 1]

    return {
        "common_patients": len(common),
        "shared_coordinates": n_valid,
        "median_per_coordinate_correlation": round(float(np.median(correlations)), 4),
        "mean_per_coordinate_correlation": round(float(np.mean(correlations)), 4),
        "direction_agreement": round(float(direction_agree), 4),
        "overall_median_effect_correlation": round(float(overall_r), 4) if np.isfinite(overall_r) else None,
        "n_coordinates_positive_correlation": int(np.sum(correlations > 0)),
        "n_coordinates_tested": len(correlations),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mgf", type=Path, required=True,
                        help="Query spectra MGF (e.g., 263 LCNEC family spectra)")
    parser.add_argument("--query-effects", type=Path, required=True,
                        help="Patient effect matrix NPZ (n_patients × n_features)")
    parser.add_argument("--query-mgf-2", type=Path, default=None,
                        help="Second platform query spectra MGF (e.g., LIPn)")
    parser.add_argument("--query-effects-2", type=Path, default=None,
                        help="Second platform effects NPZ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("cosine", "entropy", "both"),
                        default="both")
    parser.add_argument("--max-coordinates", type=int, default=2000)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-invalid-legacy-pilot",
        action="store_true",
        help="Reproduce the invalid 2353437 pilot only; never use for a formal P2 claim.",
    )
    args = parser.parse_args()

    if not args.allow_invalid_legacy_pilot:
        raise RuntimeError(
            "LCNEC_P2_RETIRED_INVALID_PILOT: P0-G no-match calibration and real "
            "LIPn MS/MS/effects are required before a formal P2 run"
        )

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Build coordinate dictionary from local GNPS
    print("Building coordinate dictionary...", flush=True)
    gnps_manifest = pd.read_csv(BASE / "gnps_gold_silver_10ppm_benchmark_v1" / "manifest.csv.gz")
    gnps_spectra = parse_mgf(BASE / "gnps_gold_silver_10ppm_benchmark_v1" / "spectra.mgf")
    coordinates, coord_meta = build_coordinate_dictionary(
        gnps_manifest, gnps_spectra, max_coordinates=args.max_coordinates)
    print(f"  coordinates: {len(coordinates)}", flush=True)

    # Load and project query spectra (platform 1)
    print(f"Projecting platform 1: {args.query_mgf}", flush=True)
    query_spectra = parse_mgf(args.query_mgf)
    print(f"  queries: {len(query_spectra)}", flush=True)

    assignments = project_queries_to_coordinates(
        query_spectra, coordinates, method=args.method,
        score_threshold=args.score_threshold)

    n_assigned = sum(1 for v in assignments.values() if v)
    n_total = len(assignments)
    print(f"  assigned: {n_assigned}/{n_total} ({n_assigned/max(n_total,1):.1%})", flush=True)

    # Load effects
    effects_data = np.load(args.query_effects)
    effects = effects_data["patient_effect"]
    patients = [str(p) for p in effects_data.get("patient", [])]
    feature_ids = effects_data.get("family_id", [])

    # Compute coordinate-level effects
    feature_titles = [q.get("title", f"query_{i}") for i, q in enumerate(query_spectra)]
    coord_effects, coord_coverage = coordinate_effects(
        effects, feature_titles, assignments, len(coordinates))

    result = {
        "status": "P2_SPECTRAL_COORDINATE_PROJECTION",
        "coordinate_dictionary_size": len(coordinates),
        "query_count": len(query_spectra),
        "assignment_rate": round(n_assigned / max(n_total, 1), 4),
        "coordinates_covered": int(coord_coverage.sum()),
        "method": args.method,
        "score_threshold": args.score_threshold,
    }

    # Cross-platform if second platform provided
    if args.query_mgf_2 and args.query_effects_2:
        print(f"Projecting platform 2: {args.query_mgf_2}", flush=True)
        query2 = parse_mgf(args.query_mgf_2)
        assignments2 = project_queries_to_coordinates(
            query2, coordinates, method=args.method,
            score_threshold=args.score_threshold)
        n2 = sum(1 for v in assignments2.values() if v)
        print(f"  assigned: {n2}/{len(query2)}", flush=True)

        effects2_data = np.load(args.query_effects_2)
        effects2 = effects2_data["patient_effect"]
        patients2 = [str(p) for p in effects2_data.get("patient", [])]
        titles2 = [q.get("title", f"q2_{i}") for i, q in enumerate(query2)]
        coord_effects2, coverage2 = coordinate_effects(
            effects2, titles2, assignments2, len(coordinates))

        consistency = cross_platform_consistency(
            coord_effects, coord_effects2, patients, patients2)
        result["cross_platform"] = consistency

        # Also compute known-only vs all-features if we have identity info
        known_coords = set()
        for i, cm in enumerate(coord_meta):
            if cm["ik14"] and cm["ik14"] != "nan":
                known_coords.add(i)
        result["known_coordinates"] = len(known_coords)

    report_path = args.output_dir / "p2_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
