#!/usr/bin/env python3
"""Score the frozen IBD Plate3 structure-candidate axis with WSE.

This is the baseline handoff for B and the integrated algorithm.  It reads no
phenotype or effect file and preserves queries with no usable spectrum or no
mass-compatible candidate as explicit zero-candidate/zero-availability groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity

from analyze_msv000088023_hfd_unknown_response import (
    cleaned_peaks,
    matched_peak_count,
    parse_selected_library_spectra,
)
from run_contrast_atlas_sparse_fragment_measure_baseline import iter_mgf, spectrum_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(path: Path, required: set[str]) -> tuple[dict[str, dict[str, object]], int]:
    spectra: dict[str, dict[str, object]] = {}
    total = 0
    for metadata, peaks in iter_mgf(path):
        total += 1
        raw_id = spectrum_id(metadata)
        if raw_id not in required:
            continue
        if raw_id in spectra:
            raise RuntimeError(f"duplicate query spectrum: {raw_id}")
        spectra[raw_id] = {
            "precursor_mz": float(metadata["PEPMASS"].split()[0]),
            "peaks": np.asarray(peaks, dtype=np.float32),
        }
    if set(spectra) != required:
        raise RuntimeError(f"query MGF is missing {len(required - set(spectra))} required spectra")
    return spectra, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True, type=Path)
    parser.add_argument("--query-mgf", required=True, type=Path)
    parser.add_argument("--library-mgf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError(args.output)
    queries_path = args.slice / "query_entities.tsv.gz"
    candidate_spectra_path = args.slice / "candidate_library_spectra.tsv.gz"
    candidate_structures_path = args.slice / "candidate_structures.tsv.gz"
    slice_report_path = args.slice / "report.json"
    for path in (
        queries_path, candidate_spectra_path, candidate_structures_path,
        slice_report_path, args.query_mgf, args.library_mgf,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    queries = pd.read_csv(queries_path, sep="\t", dtype={"raw_entity_id": str})
    candidate_spectra = pd.read_csv(candidate_spectra_path, sep="\t")
    candidate_structures = pd.read_csv(candidate_structures_path, sep="\t", dtype=str)
    if queries["entity_id"].duplicated().any():
        raise RuntimeError("query entity IDs are not unique")
    if candidate_structures.duplicated(["entity_id", "candidate_id"]).any():
        raise RuntimeError("structure candidate axis is not unique")
    required_raw = set(queries["raw_entity_id"].astype(str))
    query_spectra, query_mgf_total = load_queries(args.query_mgf, required_raw)

    selected_rows = set(candidate_spectra["row"].astype(int))
    library, library_total = parse_selected_library_spectra(args.library_mgf, selected_rows)
    if set(library) != selected_rows:
        raise RuntimeError("selected library spectra are incomplete")
    query_clean = {
        raw: cleaned_peaks(spectrum, args.fragment_da)
        for raw, spectrum in query_spectra.items()
    }
    library_clean = {
        row: cleaned_peaks(spectrum, args.fragment_da)
        for row, spectrum in library.items()
    }
    raw_by_entity = dict(zip(queries["entity_id"].astype(str), queries["raw_entity_id"].astype(str)))
    eligible_by_entity = dict(zip(queries["entity_id"].astype(str), queries["spectral_eligible"].astype(bool)))
    scored_rows: list[dict[str, object]] = []
    for row in candidate_spectra.itertuples(index=False):
        entity = str(row.entity_id)
        if not eligible_by_entity[entity]:
            continue
        query = query_clean[raw_by_entity[entity]]
        reference = library_clean[int(row.row)]
        score = float(calculate_entropy_similarity(
            query, reference, ms2_tolerance_in_da=args.fragment_da, clean_spectra=False,
        ))
        scored_rows.append({
            "entity_id": entity,
            "candidate_id": str(row.candidate_id),
            "library_row": int(row.row),
            "WSE_score": score,
            "matched_peaks": matched_peak_count(query, reference, args.fragment_da),
        })
    scored = pd.DataFrame(scored_rows)
    if scored.empty:
        best = pd.DataFrame(columns=["entity_id", "candidate_id", "WSE_score", "matched_peaks", "library_row"])
    else:
        best = (
            scored.sort_values(
                ["entity_id", "candidate_id", "WSE_score", "matched_peaks", "library_row"],
                ascending=[True, True, False, False, True],
            )
            .drop_duplicates(["entity_id", "candidate_id"], keep="first")
        )
    axis = candidate_structures[["entity_id", "candidate_id"]].copy()
    axis = axis.merge(best, on=["entity_id", "candidate_id"], how="left", validate="one_to_one")

    query_ptr = [0]
    candidate_ids: list[str] = []
    wse_scores: list[float] = []
    matched_peaks: list[int] = []
    availability: list[bool] = []
    axis_by_query = {str(key): block for key, block in axis.groupby("entity_id", sort=False)}
    for query in queries.itertuples(index=False):
        block = axis_by_query.get(str(query.entity_id))
        if block is not None:
            block = block.sort_values("candidate_id", kind="mergesort")
            candidate_ids.extend(block["candidate_id"].astype(str))
            score_values = pd.to_numeric(block["WSE_score"], errors="coerce").to_numpy(float)
            match_values = pd.to_numeric(block["matched_peaks"], errors="coerce").fillna(0).to_numpy(int)
            wse_scores.extend(score_values)
            matched_peaks.extend(match_values)
            availability.extend(np.isfinite(score_values))
        query_ptr.append(len(candidate_ids))
    if len(candidate_ids) != len(axis):
        raise RuntimeError("candidate bundle axis lost rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("ibd200_plate3_candidate_score_bundle_v1"),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        query_ids=queries["entity_id"].astype(str).to_numpy(dtype=str),
        candidate_ids=np.asarray(candidate_ids, dtype=str),
        WSE_score=np.asarray(wse_scores, dtype=np.float32),
        WSE_matched_peaks=np.asarray(matched_peaks, dtype=np.int16),
        WSE_availability=np.asarray(availability, dtype=bool),
    )
    available_query = np.asarray([
        any(availability[query_ptr[index]:query_ptr[index + 1]])
        for index in range(len(queries))
    ])
    report = {
        "schema": "ibd200_plate3_candidate_score_bundle_report_v1",
        "status": "DEVELOPMENT_CONSUMED_WSE_CANDIDATE_BUNDLE_COMPLETE",
        "labels_accessed": False,
        "method_id": "weighted_spectral_entropy",
        "parameters": {"fragment_da": args.fragment_da},
        "denominators": {
            "queries": len(queries),
            "candidate_structures": len(axis),
            "candidate_library_spectrum_edges": len(candidate_spectra),
            "scored_library_spectrum_edges": len(scored),
            "available_candidate_scores": int(np.sum(availability)),
            "queries_with_available_score": int(available_query.sum()),
            "queries_abstained": int((~available_query).sum()),
            "query_mgf_spectra": query_mgf_total,
            "library_mgf_spectra": library_total,
            "selected_library_spectra": len(selected_rows),
        },
        "input_sha256": {
            str(path): sha256(path)
            for path in (
                queries_path, candidate_spectra_path, candidate_structures_path,
                slice_report_path, args.query_mgf, args.library_mgf,
            )
        },
        "output_sha256": sha256(args.output),
        "handoff": (
            "query_ptr/query_ids/candidate_ids is the immutable biology candidate axis; "
            "B, integrated spectral and BioAware scores must preserve it exactly"
        ),
        "claim_boundary": (
            "This phenotype-blind bundle is a baseline score artifact, not an identification, "
            "biological result, or unknown-information increment."
        ),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
