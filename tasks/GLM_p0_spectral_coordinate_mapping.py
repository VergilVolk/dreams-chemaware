"""P0: Hidden-name cross-instrument spectral coordinate mapping.

Core question: can a unified MS/MS model map the same molecule to the same
coordinate across instruments, better than existing alternatives?

Method:
1. Build a FIXED coordinate dictionary from gold-quality GNPS spectra
   (each coordinate = one molecule's consensus spectrum).
2. Hide all names. For each spectrum in the test set, project to coordinates.
3. Measure: does the same molecule (from a different file/instrument) land
   on the correct coordinate?

Test universe: GNPS gold/silver benchmark (329,607 spectra, 5,534 identities).
Cross-file constraint: query and its correct coordinate must come from
different FILENAMEs (already enforced by benchmark construction).

Comparisons:
  A. m/z-only matching (nearest precursor mass)
  B. Cosine similarity (classical)
  C. Weighted spectral entropy
  D. Official DreaMS embedding
  E. Noise V1 embedding (if available locally)

Metrics:
  - Top-1/5/10/20 accuracy (same-molecule retrieval)
  - Near-structure false-merge rate (same-formula different-identity in top-k)
  - Coverage-error curve (abstention vs error)
  - Calibration (predicted confidence vs actual accuracy)

This is entirely CPU-compatible: uses frozen embeddings + numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

TOLERANCE = 0.02
BASE = os.path.join("data", "validation")


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=2000,
                        help="Number of test queries (subsampled for speed)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1",
                                 "manifest.csv.gz")
    mgf_path = os.path.join(BASE, "gnps_gold_silver_10ppm_benchmark_v1",
                            "spectra.mgf")
    manifest = pd.read_csv(manifest_path)
    print(f"manifest: {len(manifest)} spectra", flush=True)

    # Build coordinate dictionary: one representative spectrum per identity (ik14)
    # Use gold-quality spectra only for coordinates
    gold_mask = manifest["quality_label"] == "gold"
    gold = manifest[gold_mask].copy()
    # For each ik14, pick the spectrum with most peaks as coordinate representative
    gold_sorted = gold.sort_values("n_peaks", ascending=False)
    coord_rows = gold_sorted.drop_duplicates(subset="ik14", keep="first")
    print(f"coordinate dictionary: {len(coord_rows)} unique identities from {len(gold)} gold spectra", flush=True)

    # Query set: ALL spectra (including silver) whose identity has a gold coordinate
    # and come from a DIFFERENT file than the coordinate
    coord_row_set = set(coord_rows["row"].tolist())
    coord_file_by_ik = dict(zip(coord_rows["ik14"], coord_rows["filename"]))

    eligible = manifest[
        manifest["ik14"].isin(coord_file_by_ik.keys()) &
        ~manifest["row"].isin(coord_row_set)
    ].copy()
    eligible["coord_file"] = eligible["ik14"].map(coord_file_by_ik)
    # Must come from different file
    eligible = eligible[eligible["filename"] != eligible["coord_file"]]
    print(f"eligible queries (cross-file, has gold coordinate): {len(eligible)}", flush=True)

    if len(eligible) > args.max_queries:
        eligible = eligible.sample(n=args.max_queries, random_state=42)
    print(f"sampled queries: {len(eligible)}", flush=True)

    # Load coordinate spectra
    coord_row_list = coord_rows["row"].tolist()
    coord_row_set_sorted = sorted(coord_row_list)
    coord_spectra = {}
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as handle:
        cur, row = None, 0
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": []}
            elif line == "END IONS":
                if cur and cur["peaks"] and row in coord_row_set_sorted:
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
    print(f"coordinate spectra loaded: {len(coord_spectra)}", flush=True)

    # Load query spectra
    query_rows = set(eligible["row"].tolist())
    query_spectra = {}
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as handle:
        cur, row = None, 0
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": []}
            elif line == "END IONS":
                if cur and cur["peaks"] and row in query_rows:
                    query_spectra[row] = cur
                cur = None
                row += 1
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line and line[0].isdigit():
                    p = line.split()
                    if len(p) >= 2 and float(p[1]) > 0:
                        cur["peaks"].append((float(p[0]), float(p[1])))
    print(f"query spectra loaded: {len(query_spectra)}", flush=True)

    # Build coordinate arrays for fast scoring
    coord_ik14 = coord_rows["ik14"].tolist()
    coord_formula = coord_rows["formula"].tolist()
    coord_mz = []
    coord_peaks_norm = []
    coord_row_order = []
    for _, row_data in coord_rows.iterrows():
        r = int(row_data["row"])
        if r in coord_spectra:
            coord_row_order.append(r)
            coord_mz.append(coord_spectra[r]["precursor"])
            coord_peaks_norm.append(norm_peaks(coord_spectra[r]["peaks"]))
    coord_mz = np.array(coord_mz)
    n_coords = len(coord_row_order)
    coord_ik_arr = [coord_ik14[i] for i in range(len(coord_ik14))
                    if int(coord_rows.iloc[i]["row"]) in set(coord_row_order)]
    # Simpler: rebuild identity list matching coord_row_order
    row_to_ik = dict(zip(coord_rows["row"], coord_rows["ik14"]))
    row_to_formula = dict(zip(coord_rows["row"], coord_rows["formula"]))
    coord_ik_final = [row_to_ik[r] for r in coord_row_order]
    coord_formula_final = [row_to_formula[r] for r in coord_row_order]
    print(f"final coordinates: {n_coords}", flush=True)

    # Score all queries
    results_by_method = defaultdict(lambda: {"correct": 0, "near_miss": 0,
                                              "total": 0, "top1_correct": 0,
                                              "top5_correct": 0, "top10_correct": 0})
    for q_idx, q_row in eligible.iterrows():
        q_row_int = int(q_row["row"])
        if q_row_int not in query_spectra:
            continue
        spec = query_spectra[q_row_int]
        true_ik = str(q_row["ik14"])
        true_formula = str(q_row["formula"])
        q_mz_val = spec["precursor"]
        q_mz_arr, q_int_arr = norm_peaks(spec["peaks"])

        # Method A: m/z only
        mz_diffs = np.abs(coord_mz - q_mz_val)
        mz_rank = np.argsort(mz_diffs)

        # Method B: cosine
        cos_scores = np.array([
            greedy_cos(q_mz_arr, q_int_arr, cm[0], cm[1])
            for cm in coord_peaks_norm
        ])
        cos_rank = np.argsort(-cos_scores)

        # Method C: entropy
        ent_scores = np.array([
            entropy_sim(q_mz_arr, q_int_arr, cm[0], cm[1])
            for cm in coord_peaks_norm
        ])
        ent_rank = np.argsort(-ent_scores)

        for method, rank in [("mz_only", mz_rank), ("cosine", cos_rank),
                              ("entropy", ent_rank)]:
            r = results_by_method[method]
            r["total"] += 1
            top1_ik = coord_ik_final[rank[0]] if len(rank) > 0 else ""
            if top1_ik == true_ik:
                r["top1_correct"] += 1
            if true_ik in [coord_ik_final[i] for i in rank[:5]]:
                r["top5_correct"] += 1
            if true_ik in [coord_ik_final[i] for i in rank[:10]]:
                r["top10_correct"] += 1
            # Near-miss: same formula, wrong identity
            if top1_ik != true_ik and coord_formula_final[rank[0]] == true_formula:
                r["near_miss"] += 1

    # Summary
    summary = {}
    for method, r in results_by_method.items():
        n = r["total"]
        summary[method] = {
            "n_queries": n,
            "top1_accuracy": round(r["top1_correct"] / n, 4),
            "top5_accuracy": round(r["top5_correct"] / n, 4),
            "top10_accuracy": round(r["top10_correct"] / n, 4),
            "near_miss_rate": round(r["near_miss"] / n, 4),
        }

    report = {
        "status": "P0_SPECTRAL_COORDINATE_MAPPING",
        "coordinate_dict_size": n_coords,
        "queries_tested": len(eligible),
        "cross_file_constraint": True,
        "results": summary,
        "gate": {
            "cosine_top1_gt_50pct": summary.get("cosine", {}).get("top1_accuracy", 0) > 0.50,
            "entropy_beats_cosine": (
                summary.get("entropy", {}).get("top1_accuracy", 0) >
                summary.get("cosine", {}).get("top1_accuracy", 0)),
        },
        "claim_limit": (
            "P0: hidden-name coordinate mapping on GNPS gold/silver. "
            "Tests whether the same molecule from a different file maps to "
            "the correct coordinate. Not a biological result."
        ),
    }
    out_path = args.output_dir / "p0_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
