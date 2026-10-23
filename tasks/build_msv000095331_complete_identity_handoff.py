#!/usr/bin/env python3
"""Build the complete same-precursor identity-candidate axis for MSV000095331.

This is deliberately distinct from the open chemical-relation shortlist.  A
pair enters this axis only when its observed precursor m/z is within the fixed
ppm window.  No spectral-score threshold, drug name, response estimate, or
registered cross-study relation is used for admission.  WSE is evaluated for
every admitted pair as a baseline score, including zero-score pairs.

The output is an identity *hypothesis* axis, not proof of molecular identity.
It does not infer neutral masses or cross-adduct equivalence because those
fields are unavailable for most unresolved entities in these studies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity, clean_spectrum

from analyze_msv000095331_chemprop_open_retrieval import (
    CHEMPROP_MGF,
    NEW_STUDY,
    load_identity,
    load_raw_spectra,
)
from build_msv000095331_integrated_algorithm_handoff import NEW_MGF, sha256


STATUS = "MSV000095331_COMPLETE_IDENTITY_HANDOFF_COMPLETE"


def ppm_difference(left: float, right: np.ndarray) -> np.ndarray:
    return np.abs(right - left) / ((right + left) / 2.0) * 1e6


def clean(peaks: np.ndarray) -> np.ndarray:
    if not len(peaks):
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(
        clean_spectrum(
            peaks,
            noise_threshold=0.01,
            min_ms2_difference_in_da=0.05,
            max_peak_num=-1,
        ),
        dtype=np.float32,
    )


def matched_fragments(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    """Count unambiguous matches after 0.05-Da cleaning.

    Cleaned peaks within either spectrum are farther apart than twice the
    0.02-Da comparison tolerance, so a deterministic two-pointer count is
    sufficient and does not affect candidate admission.
    """
    i = j = count = 0
    while i < len(left) and j < len(right):
        delta = float(left[i, 0] - right[j, 0])
        if abs(delta) <= tolerance:
            count += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return count


def directional_block(
    query_study: str,
    query_ids: np.ndarray,
    query_peaks: list[np.ndarray],
    query_precursors: dict[str, float],
    candidate_study: str,
    candidate_ids: np.ndarray,
    candidate_peaks: list[np.ndarray],
    candidate_precursors: dict[str, float],
    identity_status: dict[tuple[str, str], str],
    precursor_ppm: float,
    fragment_tolerance_da: float,
) -> tuple[list[dict], list[dict]]:
    candidate_ids = np.asarray(candidate_ids, dtype=str)
    candidate_mz = np.asarray(
        [candidate_precursors[str(entity)] for entity in candidate_ids], dtype=float
    )
    candidate_clean = [clean(peaks) for peaks in candidate_peaks]
    order = np.argsort(candidate_mz, kind="mergesort")
    sorted_mz = candidate_mz[order]
    pair_rows: list[dict] = []
    query_rows: list[dict] = []

    for query_entity, raw_query_peaks in zip(query_ids, query_peaks):
        query_entity = str(query_entity)
        query_mz = float(query_precursors[query_entity])
        # These bounds are intentionally only a fast superset.  Exact symmetric
        # ppm filtering below defines the candidate universe.
        lower = query_mz / (1.0 + precursor_ppm * 1e-6)
        upper = query_mz / (1.0 - precursor_ppm * 1e-6)
        left = int(np.searchsorted(sorted_mz, lower, side="left"))
        right = int(np.searchsorted(sorted_mz, upper, side="right"))
        indices = order[left:right]
        if len(indices):
            exact = ppm_difference(query_mz, candidate_mz[indices]) <= precursor_ppm
            indices = indices[exact]
        indices = np.asarray(
            sorted(indices, key=lambda idx: (candidate_mz[idx], candidate_ids[idx])),
            dtype=int,
        )
        query_rows.append({
            "query_id": f"{query_study}::{query_entity}",
            "query_study_id": query_study,
            "query_raw_entity_id": query_entity,
            "candidate_study_id": candidate_study,
            "candidate_count": int(len(indices)),
            "query_status": (
                "no_mass_compatible_candidate" if len(indices) == 0 else
                "single_mass_compatible_candidate" if len(indices) == 1 else
                "multiple_mass_compatible_candidates"
            ),
        })
        query_clean = clean(raw_query_peaks)
        for index in indices:
            candidate_entity = str(candidate_ids[index])
            candidate_mz_value = float(candidate_mz[index])
            candidate_spectrum = candidate_clean[index]
            if len(query_clean) and len(candidate_spectrum):
                wse = float(calculate_entropy_similarity(
                    query_clean,
                    candidate_spectrum,
                    ms2_tolerance_in_da=fragment_tolerance_da,
                    clean_spectra=False,
                ))
                matched = matched_fragments(
                    query_clean, candidate_spectrum, fragment_tolerance_da
                )
            else:
                wse = 0.0
                matched = 0
            query_id = f"{query_study}::{query_entity}"
            candidate_id = f"{candidate_study}::{candidate_entity}"
            pair_text = f"{query_id}|{candidate_id}"
            pair_rows.append({
                "candidate_pair_id": hashlib.sha256(pair_text.encode("utf-8")).hexdigest(),
                "query_id": query_id,
                "candidate_id": candidate_id,
                "query_study_id": query_study,
                "query_raw_entity_id": query_entity,
                "candidate_study_id": candidate_study,
                "candidate_raw_entity_id": candidate_entity,
                "query_identity_status": identity_status[(query_study, query_entity)],
                "candidate_identity_status": identity_status[(candidate_study, candidate_entity)],
                "query_precursor_mz": query_mz,
                "candidate_precursor_mz": candidate_mz_value,
                "precursor_ppm_difference": float(
                    ppm_difference(query_mz, np.asarray([candidate_mz_value]))[0]
                ),
                "candidate_axis": "same_observed_precursor_10ppm_identity_hypothesis",
                "wse_score": wse,
                "wse_matched_fragments": int(matched),
            })
    return pair_rows, query_rows


def wse_decisions(pair_table: pd.DataFrame, query_table: pd.DataFrame) -> pd.DataFrame:
    decision_rows: list[dict] = []
    grouped = {
        key: block for key, block in pair_table.groupby(
            ["query_id", "candidate_study_id"], sort=False
        )
    }
    for query in query_table.itertuples(index=False):
        key = (str(query.query_id), str(query.candidate_study_id))
        block = grouped.get(key)
        base = {
            "method_name": "weighted_spectral_entropy",
            "query_id": str(query.query_id),
            "query_study_id": str(query.query_study_id),
            "query_raw_entity_id": str(query.query_raw_entity_id),
            "candidate_study_id": str(query.candidate_study_id),
            "candidate_count": int(query.candidate_count),
        }
        if block is None or block.empty:
            decision_rows.append({
                **base,
                "decision_status": "no_mass_compatible_candidate",
                "selected_candidate_id": "",
                "selected_candidate_raw_entity_id": "",
                "selected_score": np.nan,
                "selected_matched_fragments": 0,
                "top_score_tie_count": 0,
            })
            continue
        ordered = block.sort_values(
            ["wse_score", "wse_matched_fragments", "precursor_ppm_difference", "candidate_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        top = ordered.iloc[0]
        tie_count = int(np.isclose(
            ordered["wse_score"].to_numpy(float), float(top["wse_score"]),
            atol=1e-12, rtol=0.0,
        ).sum())
        has_spectral_evidence = (
            float(top["wse_score"]) > 0 and int(top["wse_matched_fragments"]) > 0
        )
        decision_rows.append({
            **base,
            "decision_status": (
                "ranked_with_spectral_evidence" if has_spectral_evidence
                else "mass_candidate_but_no_spectral_evidence"
            ),
            "selected_candidate_id": str(top["candidate_id"]),
            "selected_candidate_raw_entity_id": str(top["candidate_raw_entity_id"]),
            "selected_score": float(top["wse_score"]),
            "selected_matched_fragments": int(top["wse_matched_fragments"]),
            "top_score_tie_count": tie_count,
        })
    decisions = pd.DataFrame(decision_rows)
    answered = decisions["decision_status"].eq("ranked_with_spectral_evidence")
    reverse = {
        (str(row.query_id), str(row.selected_candidate_id))
        for row in decisions.loc[answered].itertuples(index=False)
    }
    decisions["mutual_top1"] = [
        bool(answered.iloc[index] and (str(row.selected_candidate_id), str(row.query_id)) in reverse)
        for index, row in enumerate(decisions.itertuples(index=False))
    ]
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-response-dir", type=Path,
        default=Path(
            "data/validation/msv000095331_drug_endpoint_interaction_responses_v3_20261023"
        ),
    )
    parser.add_argument(
        "--common-identity-root", type=Path,
        default=Path("data/validation/contrast_atlas_open_v9_common_identity_shards_20261023"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    identity = load_identity(
        args.new_response_dir / "source_native_identity.tsv",
        args.common_identity_root,
        set(CHEMPROP_MGF),
    )
    status = {
        (str(row.study_id), str(row.raw_entity_id)): str(row.identity_status)
        for row in identity.itertuples(index=False)
    }
    mgfs = {NEW_STUDY: NEW_MGF, **CHEMPROP_MGF}
    spectra: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    precursors: dict[str, dict[str, float]] = {}
    loading: dict[str, dict] = {}
    for study, path in mgfs.items():
        required = set(identity.loc[identity["study_id"].eq(study), "raw_entity_id"])
        ids, peaks, precursors[study], loading[study] = load_raw_spectra(path, required)
        spectra[study] = (ids, peaks)

    pairs: list[dict] = []
    queries: list[dict] = []
    new_ids, new_peaks = spectra[NEW_STUDY]
    for study in CHEMPROP_MGF:
        old_ids, old_peaks = spectra[study]
        block, registry = directional_block(
            NEW_STUDY, new_ids, new_peaks, precursors[NEW_STUDY],
            study, old_ids, old_peaks, precursors[study], status,
            args.precursor_ppm, args.fragment_tolerance_da,
        )
        pairs.extend(block)
        queries.extend(registry)
        block, registry = directional_block(
            study, old_ids, old_peaks, precursors[study],
            NEW_STUDY, new_ids, new_peaks, precursors[NEW_STUDY], status,
            args.precursor_ppm, args.fragment_tolerance_da,
        )
        pairs.extend(block)
        queries.extend(registry)

    pair_table = pd.DataFrame(pairs)
    query_table = pd.DataFrame(queries)
    if pair_table["candidate_pair_id"].duplicated().any():
        raise ValueError("duplicate directional candidate pair")
    if not pair_table["precursor_ppm_difference"].le(args.precursor_ppm + 1e-9).all():
        raise RuntimeError("candidate axis contains an out-of-window pair")
    decisions = wse_decisions(pair_table, query_table)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    pair_path = args.output_dir / "complete_identity_candidate_pairs.tsv.gz"
    query_path = args.output_dir / "directional_query_registry.tsv.gz"
    decision_path = args.output_dir / "wse_entity_decisions.tsv.gz"
    entity_path = args.output_dir / "study_entity_registry.tsv.gz"
    axis_path = args.output_dir / "wse_identity_inference_axis.npz"
    pair_table.to_csv(pair_path, sep="\t", index=False, compression=compression)
    query_table.to_csv(query_path, sep="\t", index=False, compression=compression)
    decisions.to_csv(decision_path, sep="\t", index=False, compression=compression)
    entity_table = identity.copy()
    entity_table["mgf_path"] = entity_table["study_id"].map(
        {study: str(path) for study, path in mgfs.items()}
    )
    entity_table.to_csv(entity_path, sep="\t", index=False, compression=compression)

    # Query-major ordering creates a direct, truth-free inference contract.
    ordered_pairs = {
        (str(query_id), str(candidate_study)): block
        for (query_id, candidate_study), block in pair_table.groupby(
            ["query_id", "candidate_study_id"], sort=False
        )
    }
    candidate_ids: list[str] = []
    candidate_pair_ids: list[str] = []
    wse_scores: list[float] = []
    ptr = [0]
    for row in query_table.itertuples(index=False):
        key = (str(row.query_id), str(row.candidate_study_id))
        if key in ordered_pairs:
            block = ordered_pairs[key]
            candidate_ids.extend(block["candidate_id"].astype(str))
            candidate_pair_ids.extend(block["candidate_pair_id"].astype(str))
            wse_scores.extend(block["wse_score"].astype(float))
        ptr.append(len(candidate_ids))
    np.savez_compressed(
        axis_path,
        schema=np.asarray("msv000095331_complete_identity_candidate_axis_v1"),
        query_ptr=np.asarray(ptr, dtype=np.int64),
        query_ids=(query_table["query_id"] + "|" + query_table["candidate_study_id"]).to_numpy(dtype="U"),
        candidate_ids=np.asarray(candidate_ids, dtype="U"),
        candidate_pair_ids=np.asarray(candidate_pair_ids, dtype="U"),
        wse_score=np.asarray(wse_scores, dtype=np.float32),
        wse_availability=np.ones(len(wse_scores), dtype=bool),
    )

    report = {
        "status": STATUS,
        "dataset": "MSV000095331 plus three ChemProp studies",
        "dataset_role": "development_consumed_not_external_validation",
        "labels_accessed": False,
        "response_effects_accessed": False,
        "drug_names_accessed": False,
        "registered_relations_accessed": False,
        "candidate_axis": {
            "track": "identity_hypothesis",
            "rule": "all pairs within symmetric observed-precursor ppm window",
            "precursor_ppm": args.precursor_ppm,
            "spectral_threshold_for_admission": None,
            "neutral_mass_or_cross_adduct_inference": False,
        },
        "wse_baseline": {
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "minimum_score_for_admission": None,
            "minimum_matched_fragments_for_admission": None,
        },
        "denominators": {
            "studies": len(mgfs),
            "directional_study_pairs": 6,
            "directional_queries": int(len(query_table)),
            "queries_without_mass_candidate": int(query_table["candidate_count"].eq(0).sum()),
            "queries_with_one_mass_candidate": int(query_table["candidate_count"].eq(1).sum()),
            "queries_with_multiple_mass_candidates": int(query_table["candidate_count"].gt(1).sum()),
            "candidate_pairs": int(len(pair_table)),
            "wse_queries_with_spectral_evidence": int(
                decisions["decision_status"].eq("ranked_with_spectral_evidence").sum()
            ),
            "wse_mutual_top1_directions": int(decisions["mutual_top1"].sum()),
        },
        "files": {
            "candidate_pairs": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "query_registry": {"path": str(query_path), "sha256": sha256(query_path)},
            "wse_entity_decisions": {"path": str(decision_path), "sha256": sha256(decision_path)},
            "entity_registry": {"path": str(entity_path), "sha256": sha256(entity_path)},
            "inference_axis": {"path": str(axis_path), "sha256": sha256(axis_path)},
        },
        "spectrum_loading": loading,
        "required_algorithm_return": [
            "candidate_pair_id", "method_name", "method_score", "evidence_available"
        ],
        "claim_boundary": [
            "The axis is complete for same-observed-precursor 10-ppm candidates, not for cross-adduct neutral entities.",
            "Mass compatibility defines identity hypotheses but does not prove identity or resolve isomers.",
            "The former WSE Top-20 axis remains a separate chemical-relation baseline and is not a universal admission gate.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
