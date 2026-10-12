#!/usr/bin/env python
"""Truth-blind cross-design feature overlap audit for ChemProp/ChemProp2.

Candidate generation uses precursor mass only. Raw centroided MS/MS cosine ranks
candidates without drug labels. Library InChIKey-Planar matches are read only
after ranking and serve as an incomplete evaluation subset, never as candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from noise_gnps_article_spectral_scores import (
    weighted_entropy_backend,
    weighted_entropy_similarity,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mgf(
    path: Path, minimum_peaks: int = 3,
) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    spectra: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
    scan: int | None = None
    precursor = np.nan
    mz: list[float] = []
    intensity: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                scan, precursor, mz, intensity = None, np.nan, [], []
            elif line.startswith("SCANS="):
                scan = int(line.split("=", 1)[1])
            elif line.startswith("PEPMASS="):
                precursor = float(line.split("=", 1)[1].split()[0])
            elif line == "END IONS":
                if scan is not None and len(mz) >= minimum_peaks:
                    order = np.argsort(mz)
                    mz_array = np.asarray(mz, dtype=np.float32)[order]
                    raw_array = np.asarray(intensity, dtype=np.float32)[order]
                    int_array = np.sqrt(raw_array.astype(np.float64))
                    norm = np.linalg.norm(int_array)
                    if norm > 0:
                        spectra[scan] = (
                            precursor, mz_array, (int_array / norm).astype(np.float32), raw_array,
                        )
            elif line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        mz.append(float(fields[0])); intensity.append(float(fields[1]))
                    except ValueError:
                        continue
    return spectra


def cosine(a_mz: np.ndarray, a_i: np.ndarray, b_mz: np.ndarray, b_i: np.ndarray, tolerance: float) -> tuple[float, int]:
    i = j = matched = 0
    dot = 0.0
    while i < len(a_mz) and j < len(b_mz):
        delta = float(a_mz[i] - b_mz[j])
        if abs(delta) <= tolerance:
            dot += float(a_i[i] * b_i[j])
            matched += 1
            i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return dot, matched


def library_map(path: Path, score: float) -> dict[int, str]:
    table = pd.read_csv(path, sep="\t")
    table["#Scan#"] = pd.to_numeric(table["#Scan#"], errors="coerce")
    table["MQScore"] = pd.to_numeric(table["MQScore"], errors="coerce")
    table["InChIKey-Planar"] = table["InChIKey-Planar"].fillna("").astype(str).str.strip()
    table = table.loc[
        table["#Scan#"].notna() & table["MQScore"].ge(score)
        & table["InChIKey-Planar"].ne("") & table["InChIKey-Planar"].ne("nan")
    ].sort_values("MQScore", ascending=False).drop_duplicates("#Scan#")
    return dict(zip(table["#Scan#"].astype(int), table["InChIKey-Planar"]))


def discover_asset(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def audit_pair(
    name: str,
    query_quant_path: Path,
    query_mgf_path: Path,
    query_library_path: Path,
    reference_root: Path,
    eligible_query_ids: set[int],
    ppm: float,
    fragment_tolerance: float,
    library_score: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    ref_quant_path = discover_asset(reference_root, "quantification_table_reformatted/*.csv")
    ref_mgf_path = reference_root / "spectra/specs_ms.mgf"
    ref_library_path = discover_asset(reference_root, "DB_result/*.tsv")
    query_quant = pd.read_csv(query_quant_path, usecols=["row ID", "row m/z", "row retention time"])
    ref_quant = pd.read_csv(ref_quant_path, usecols=["row ID", "row m/z", "row retention time"])
    query_quant["row ID"] = query_quant["row ID"].astype(int)
    ref_quant["row ID"] = ref_quant["row ID"].astype(int)
    query_spectra = parse_mgf(query_mgf_path)
    ref_spectra = parse_mgf(ref_mgf_path)
    query_quant = query_quant.loc[query_quant["row ID"].isin(query_spectra)].copy()
    ref_quant = ref_quant.loc[ref_quant["row ID"].isin(ref_spectra)].copy()
    query_library = library_map(query_library_path, library_score)
    ref_library = library_map(ref_library_path, library_score)

    ref_sorted = ref_quant.sort_values("row m/z").reset_index(drop=True)
    ref_masses = ref_sorted["row m/z"].to_numpy(float)
    rows: list[dict[str, object]] = []
    cosine_edges: list[tuple[int, int, float, float]] = []
    wse_edges: list[tuple[int, int, float, float]] = []
    edge_rows: list[dict[str, object]] = []
    for query in query_quant.itertuples(index=False):
        query_id = int(getattr(query, "_0"))
        query_mz = float(getattr(query, "_1"))
        query_rt = float(getattr(query, "_2"))
        window = query_mz * ppm * 1e-6
        lo = bisect_left(ref_masses, query_mz - window)
        hi = bisect_right(ref_masses, query_mz + window)
        candidates: list[tuple[float, float, int, float, float, int]] = []
        _, q_mz, q_i, q_raw = query_spectra[query_id]
        q_spectrum = np.stack([q_mz, q_raw])
        for ref in ref_sorted.iloc[lo:hi].itertuples(index=False):
            ref_id = int(getattr(ref, "_0"))
            ref_mass = float(getattr(ref, "_1"))
            ref_rt = float(getattr(ref, "_2"))
            _, r_mz, r_i, r_raw = ref_spectra[ref_id]
            cosine_score, shared = cosine(q_mz, q_i, r_mz, r_i, fragment_tolerance)
            wse_score = weighted_entropy_similarity(
                q_spectrum, np.stack([r_mz, r_raw]), fragment_tolerance,
            )
            candidates.append((cosine_score, wse_score, ref_id, ref_mass, ref_rt, shared))
            mass_error = abs(ref_mass - query_mz)
            cosine_edges.append((query_id, ref_id, cosine_score, mass_error))
            wse_edges.append((query_id, ref_id, wse_score, mass_error))
            edge_rows.append({
                "pair": name,
                "query_id": query_id,
                "reference_id": ref_id,
                "query_mz": query_mz,
                "query_rt_min": query_rt,
                "reference_mz": ref_mass,
                "reference_rt_min": ref_rt,
                "absolute_mass_error_da": mass_error,
                "cosine": cosine_score,
                "wse": wse_score,
                "shared_peaks": shared,
            })
        cosine_candidates = sorted(candidates, key=lambda item: (-item[0], abs(item[3] - query_mz), item[2]))
        wse_candidates = sorted(candidates, key=lambda item: (-item[1], abs(item[3] - query_mz), item[2]))
        top = cosine_candidates[0] if cosine_candidates else (np.nan, np.nan, -1, np.nan, np.nan, 0)
        top_wse = wse_candidates[0] if wse_candidates else (np.nan, np.nan, -1, np.nan, np.nan, 0)
        q_key = query_library.get(query_id, "")
        top_key = ref_library.get(int(top[2]), "") if candidates else ""
        top_wse_key = ref_library.get(int(top_wse[2]), "") if candidates else ""
        truth_available = bool(q_key and q_key in set(ref_library.values()))
        rows.append({
            "pair": name,
            "query_id": query_id,
            "query_mz": query_mz,
            "query_rt_min": query_rt,
            "candidate_count_10ppm": len(candidates),
            "top_reference_id": int(top[2]),
            "top_reference_mz": float(top[3]),
            "top_reference_rt_min": float(top[4]),
            "top_ppm_error": float(1e6 * (float(top[3]) - query_mz) / query_mz) if candidates else np.nan,
            "top_rt_delta_min": float(top[4] - query_rt) if candidates else np.nan,
            "top_cosine": float(top[0]),
            "top_shared_peaks": int(top[5]),
            "top_wse_reference_id": int(top_wse[2]),
            "top_wse_reference_mz": float(top_wse[3]),
            "top_wse_reference_rt_min": float(top_wse[4]),
            "top_wse": float(top_wse[1]),
            "query_library_key": q_key,
            "top_reference_library_key": top_key,
            "top_wse_reference_library_key": top_wse_key,
            "library_truth_available": truth_available,
            "library_top1_correct": bool(truth_available and q_key == top_key),
            "library_wse_top1_correct": bool(truth_available and q_key == top_wse_key),
        })
    ranked = pd.DataFrame(rows)
    # A top match is mutual only if the query is also the best-scoring query
    # across every mass-constrained edge incident on that reference.
    edges = pd.DataFrame(cosine_edges, columns=["query_id", "reference_id", "cosine", "mass_error"])
    reverse_best = edges.sort_values(
        ["reference_id", "cosine", "mass_error", "query_id"],
        ascending=[True, False, True, True],
    ).drop_duplicates("reference_id")
    reverse_lookup = dict(zip(reverse_best["reference_id"], reverse_best["query_id"]))
    ranked["mutual_top1"] = [
        reference_id >= 0 and reverse_lookup.get(reference_id) == query_id
        for query_id, reference_id in zip(ranked["query_id"], ranked["top_reference_id"])
    ]
    wse_edge_table = pd.DataFrame(wse_edges, columns=["query_id", "reference_id", "wse", "mass_error"])
    wse_reverse_best = wse_edge_table.sort_values(
        ["reference_id", "wse", "mass_error", "query_id"],
        ascending=[True, False, True, True],
    ).drop_duplicates("reference_id")
    wse_reverse_lookup = dict(zip(wse_reverse_best["reference_id"], wse_reverse_best["query_id"]))
    ranked["wse_mutual_top1"] = [
        reference_id >= 0 and wse_reverse_lookup.get(reference_id) == query_id
        for query_id, reference_id in zip(ranked["query_id"], ranked["top_wse_reference_id"])
    ]

    truth = ranked.loc[ranked["library_truth_available"]]
    eligible = ranked.loc[ranked["query_id"].isin(eligible_query_ids)]
    report = {
        "pair": name,
        "query_features_with_nonempty_ms2": int(len(query_quant)),
        "query_features_in_frozen_qc_universe": int(len(eligible)),
        "reference_features_with_nonempty_ms2": int(len(ref_quant)),
        "queries_with_mass_candidate": int(ranked["candidate_count_10ppm"].gt(0).sum()),
        "mass_candidate_coverage": float(ranked["candidate_count_10ppm"].gt(0).mean()),
        "candidate_count_median": float(ranked["candidate_count_10ppm"].median()),
        "candidate_count_p95": float(ranked["candidate_count_10ppm"].quantile(0.95)),
        "bidirectional_mutual_top1": int(ranked["mutual_top1"].sum()),
        "wse_bidirectional_mutual_top1": int(ranked["wse_mutual_top1"].sum()),
        "thresholds": {
            str(threshold): {
                "top1_count": int(ranked["top_cosine"].ge(threshold).sum()),
                "top1_coverage": float(ranked["top_cosine"].ge(threshold).mean()),
                "bidirectional_mutual_count": int((ranked["top_cosine"].ge(threshold) & ranked["mutual_top1"]).sum()),
            } for threshold in (0.50, 0.70, 0.80, 0.90)
        },
        "frozen_qc_universe": {
            "queries_with_mass_candidate": int(eligible["candidate_count_10ppm"].gt(0).sum()),
            "mass_candidate_coverage": float(eligible["candidate_count_10ppm"].gt(0).mean()),
            "thresholds": {
                str(threshold): {
                    "top1_count": int(eligible["top_cosine"].ge(threshold).sum()),
                    "top1_coverage": float(eligible["top_cosine"].ge(threshold).mean()),
                    "bidirectional_mutual_count": int(
                        (eligible["top_cosine"].ge(threshold) & eligible["mutual_top1"]).sum()
                    ),
                } for threshold in (0.50, 0.70, 0.80, 0.90)
            },
        },
        "library_truth": {
            "query_library_features": int(ranked["query_library_key"].ne("").sum()),
            "queries_with_key_present_in_reference": int(len(truth)),
            "top1_correct": int(truth["library_top1_correct"].sum()),
            "top1_accuracy": float(truth["library_top1_correct"].mean()) if len(truth) else None,
            "wse_top1_correct": int(truth["library_wse_top1_correct"].sum()),
            "wse_top1_accuracy": float(truth["library_wse_top1_correct"].mean()) if len(truth) else None,
            "correct_at_cosine_070": int((truth["library_top1_correct"] & truth["top_cosine"].ge(0.70)).sum()),
        },
        "files": {
            "query_quantification": {"path": str(query_quant_path), "sha256": sha256(query_quant_path)},
            "query_mgf": {"path": str(query_mgf_path), "sha256": sha256(query_mgf_path)},
            "query_library": {"path": str(query_library_path), "sha256": sha256(query_library_path)},
            "reference_quantification": {"path": str(ref_quant_path), "sha256": sha256(ref_quant_path)},
            "reference_mgf": {"path": str(ref_mgf_path), "sha256": sha256(ref_mgf_path)},
            "reference_library": {"path": str(ref_library_path), "sha256": sha256(ref_library_path)},
        },
    }
    return report, ranked, pd.DataFrame(edge_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-root", type=Path, default=Path("data/reference/chemprop2_m1_20261011"))
    parser.add_argument("--endpoint-root", type=Path, default=Path("data/reference/chemprop_cross_design_20261011"))
    parser.add_argument("--query-audit", type=Path, default=Path("data/validation/chemprop2_response_m1_v2_20261011/feature_qc_audit.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/chemprop_cross_design_overlap_audit_v5_20261011"))
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--library-score", type=float, default=0.80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")
    reports: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    query_audit = pd.read_csv(args.query_audit)
    eligible_query_ids = set(query_audit.loc[query_audit["analysis_pass"], "row_id"].astype(int))
    for name, reference in (
        ("time_to_23drug", args.endpoint_root / "chemprop_23drug"),
        ("time_to_45drug", args.endpoint_root / "chemprop_45drug"),
    ):
        report, frame, edges = audit_pair(
            name,
            args.time_root / "quantification.csv",
            args.time_root / "consensus.mgf",
            args.time_root / "library_matches.tsv",
            reference,
            eligible_query_ids,
            args.ppm,
            args.fragment_tolerance,
            args.library_score,
        )
        reports.append(report); frames.append(frame); edge_frames.append(edges)
    args.output_dir.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "top_matches.csv.gz", index=False, compression="gzip")
    pd.concat(edge_frames, ignore_index=True).to_csv(
        args.output_dir / "candidate_edges_10ppm.csv.gz", index=False, compression="gzip",
    )
    output = {
        "status": "chemprop_cross_design_spectral_overlap_audit_complete",
        "formal": False,
        "evidence_role": "development_consumed",
        "candidate_generation": f"precursor mass +/- {args.ppm} ppm",
        "ranking": [
            f"sqrt-intensity raw MS/MS cosine, {args.fragment_tolerance} Da fragment tolerance",
            f"weighted spectral entropy, {args.fragment_tolerance} Da fragment tolerance",
        ],
        "weighted_entropy_backend": weighted_entropy_backend(),
        "label_use": "drug labels not read; library identity read only after ranking for incomplete evaluation",
        "candidate_edge_export": "all 10-ppm edges; downstream methods must filter the same reference-QC graph before ranking",
        "pairs": reports,
        "does_not_establish": [
            "cross-design response retrieval",
            "unknown-entity identity truth",
            "incremental value of DreaMS, Noise, P2b, or ChemAware",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
