#!/usr/bin/env python3
"""Freeze a truth-blind WSE shortlist for integrated-algorithm application.

The biology task is fixed to MSV000095331 versus the three ChemProp projects.
For every spectrum, this script retains up to K WSE candidates in each other
project and does so in both directions.  Drug names, response beta values, and
registered cross-project relations are not loaded.  B and the integrated model
must score this exact axis; WSE candidate generation is therefore held fixed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import FlashEntropySearch

from analyze_msv000095331_chemprop_open_retrieval import (
    CHEMPROP_MGF,
    NEW_STUDY,
    load_identity,
    load_raw_spectra,
)


NEW_MGF = Path("data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directional_topk(
    query_study: str,
    query_ids: np.ndarray,
    query_peaks: list[np.ndarray],
    candidate_study: str,
    candidate_ids: np.ndarray,
    candidate_peaks: list[np.ndarray],
    identity_status: dict[tuple[str, str], str],
    precursors: dict[str, dict[str, float]],
    topk: int,
    tolerance: float,
    minimum: float,
    minimum_matched: int,
    precursor_ppm: float,
) -> tuple[list[dict], list[dict]]:
    search = FlashEntropySearch(max_ms2_tolerance_in_da=max(tolerance, 0.024))
    indexed = search.build_index(
        [
            {"precursor_mz": 0.0, "peaks": peaks.copy(), "entity_id": str(entity)}
            for entity, peaks in zip(candidate_ids, candidate_peaks)
        ],
        precursor_ions_removal_da=None,
        clean_spectra=True,
    )
    indexed_ids = np.asarray([str(item["entity_id"]) for item in indexed], dtype=str)
    pair_rows, query_rows = [], []
    for query_id, peaks in zip(query_ids, query_peaks):
        query_id = str(query_id)
        cleaned = search.clean_spectrum_for_search(
            precursor_mz=0.0,
            peaks=peaks,
            precursor_ions_removal_da=None,
            noise_threshold=0.01,
            min_ms2_difference_in_da=0.05,
            max_peak_num=0,
        )
        selected: list[int] = []
        similarities = np.asarray([], dtype=float)
        matched = np.asarray([], dtype=int)
        if len(cleaned) and len(indexed_ids):
            similarities, matched = search.open_search(
                cleaned, tolerance, output_matched_peak_number=True
            )
            eligible = np.flatnonzero(
                (matched >= minimum_matched) & (similarities >= minimum)
            )
            if len(eligible):
                ordering = sorted(
                    eligible,
                    key=lambda index: (-float(similarities[index]), str(indexed_ids[index])),
                )
                selected = ordering[:topk]
        query_rows.append({
            "query_study_id": query_study,
            "query_raw_entity_id": query_id,
            "candidate_study_id": candidate_study,
            "candidate_count": len(selected),
            "query_status": "candidate_shortlist" if selected else "no_candidate",
        })
        for rank, index in enumerate(selected, start=1):
            candidate_id = str(indexed_ids[index])
            query_mz = precursors[query_study].get(query_id, np.nan)
            candidate_mz = precursors[candidate_study].get(candidate_id, np.nan)
            ppm = (
                abs(query_mz - candidate_mz) / ((query_mz + candidate_mz) / 2.0) * 1e6
                if np.isfinite(query_mz) and np.isfinite(candidate_mz)
                and query_mz > 0 and candidate_mz > 0 else np.nan
            )
            text = f"{query_study}::{query_id}|{candidate_study}::{candidate_id}"
            pair_rows.append({
                "candidate_pair_id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "query_id": f"{query_study}::{query_id}",
                "candidate_id": f"{candidate_study}::{candidate_id}",
                "query_study_id": query_study,
                "query_raw_entity_id": query_id,
                "candidate_study_id": candidate_study,
                "candidate_raw_entity_id": candidate_id,
                "wse_shortlist_rank": rank,
                "wse_score": float(similarities[index]),
                "wse_matched_fragments": int(matched[index]),
                "query_identity_status": identity_status[(query_study, query_id)],
                "candidate_identity_status": identity_status[(candidate_study, candidate_id)],
                "query_precursor_mz": query_mz,
                "candidate_precursor_mz": candidate_mz,
                "precursor_ppm_difference": ppm,
                "edge_track": (
                    "precursor_compatible_identity_hypothesis"
                    if np.isfinite(ppm) and ppm <= precursor_ppm
                    else "chemical_relation_only"
                ),
            })
    return pair_rows, query_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-response-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_drug_endpoint_interaction_responses_v3_20261023"
        ),
    )
    parser.add_argument(
        "--common-identity-root", type=Path,
        default=Path("data/validation/contrast_atlas_open_v9_common_identity_shards_20261023"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--minimum-wse", type=float, default=0.70)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--minimum-matched-fragments", type=int, default=6)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.topk < 2:
        raise ValueError("topk must preserve actual reranking competition")

    identity = load_identity(
        args.new_response_dir / "source_native_identity.tsv",
        args.common_identity_root,
        set(CHEMPROP_MGF),
    )
    mgfs = {NEW_STUDY: NEW_MGF, **CHEMPROP_MGF}
    spectra, precursors, loading = {}, {}, {}
    for study, path in mgfs.items():
        required = set(identity.loc[identity["study_id"].eq(study), "raw_entity_id"])
        ids, peaks, precursors[study], stats = load_raw_spectra(path, required)
        spectra[study] = (ids, peaks)
        loading[study] = stats
    status = {
        (str(row.study_id), str(row.raw_entity_id)): str(row.identity_status)
        for row in identity.itertuples(index=False)
    }

    pairs, queries = [], []
    new_ids, new_peaks = spectra[NEW_STUDY]
    for study in CHEMPROP_MGF:
        candidate_ids, candidate_peaks = spectra[study]
        block, registry = directional_topk(
            NEW_STUDY, new_ids, new_peaks, study, candidate_ids, candidate_peaks,
            status, precursors, args.topk, args.fragment_tolerance_da,
            args.minimum_wse, args.minimum_matched_fragments, args.precursor_ppm,
        )
        pairs.extend(block); queries.extend(registry)
        block, registry = directional_topk(
            study, candidate_ids, candidate_peaks, NEW_STUDY, new_ids, new_peaks,
            status, precursors, args.topk, args.fragment_tolerance_da,
            args.minimum_wse, args.minimum_matched_fragments, args.precursor_ppm,
        )
        pairs.extend(block); queries.extend(registry)

    pair_table = pd.DataFrame(pairs)
    query_table = pd.DataFrame(queries)
    entity_table = identity.copy()
    entity_table["mgf_path"] = entity_table["study_id"].map(
        {study: str(path) for study, path in mgfs.items()}
    )
    if pair_table["candidate_pair_id"].duplicated().any():
        raise ValueError("duplicate directed candidate pair")

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    pair_path = args.output_dir / "fixed_directional_candidate_pairs.tsv.gz"
    query_path = args.output_dir / "directional_query_registry.tsv.gz"
    entity_path = args.output_dir / "study_entity_registry.tsv.gz"
    pair_table.to_csv(pair_path, sep="\t", index=False, compression=compression)
    query_table.to_csv(query_path, sep="\t", index=False, compression=compression)
    entity_table.to_csv(entity_path, sep="\t", index=False, compression=compression)
    report = {
        "status": "MSV000095331_INTEGRATED_ALGORITHM_HANDOFF_COMPLETE",
        "labels_accessed": False,
        "response_effects_accessed": False,
        "registered_relations_accessed": False,
        "candidate_generation": {
            "method": "WSE directional top-K",
            "topk": args.topk,
            "minimum_wse": args.minimum_wse,
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "minimum_matched_fragments": args.minimum_matched_fragments,
        },
        "denominators": {
            "studies": len(mgfs),
            "directional_study_pairs": 6,
            "directional_queries": int(len(query_table)),
            "queries_with_candidates": int(query_table["candidate_count"].gt(0).sum()),
            "queries_without_candidates": int(query_table["candidate_count"].eq(0).sum()),
            "candidate_pairs": int(len(pair_table)),
            "identity_hypothesis_pairs": int(
                pair_table["edge_track"].eq("precursor_compatible_identity_hypothesis").sum()
            ),
            "chemical_relation_pairs": int(
                pair_table["edge_track"].eq("chemical_relation_only").sum()
            ),
        },
        "required_algorithm_output": [
            "candidate_pair_id", "query_id", "candidate_id", "method_name",
            "method_score", "evidence_available",
        ],
        "comparison_rule": (
            "WSE, B, and the integrated model must retain this exact query/candidate axis; "
            "each method selects directional Top-1 within a study pair, followed by mutual "
            "Top-1 response scoring with zero-evidence ties receiving no Top-1 credit"
        ),
        "files": {
            "candidate_pairs": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "query_registry": {"path": str(query_path), "sha256": sha256(query_path)},
            "entity_registry": {"path": str(entity_path), "sha256": sha256(entity_path)},
            "mgfs": {
                study: {"path": str(path), "sha256": sha256(path)}
                for study, path in mgfs.items()
            },
        },
        "spectrum_loading": loading,
        "claim_boundary": [
            "This handoff tests reranking within a fixed WSE shortlist; it cannot measure candidates outside that shortlist.",
            "No drug name, response effect, or cross-project truth enters candidate generation.",
            "A chemical-relation pair is not an identity assertion.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
