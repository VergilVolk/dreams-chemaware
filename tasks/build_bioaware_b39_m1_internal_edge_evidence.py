#!/usr/bin/env python
"""Rebuild B3/B9 evidence at candidate--seed edge resolution for B39.

The historical B3/B9 caches aggregate all reaction neighbours before writing
one candidate-context row.  That is too coarse for B39.  This script reuses
their frozen matching and scoring functions, but writes one row per
query/candidate/rotation/seed edge and proves that re-aggregation reproduces
the historical caches.  It covers the four internal synthetic-rotation
sources only; ST and KGMN remain explicitly missing rather than zero-filled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b3_reaction_coabundance_action import (  # noqa: E402
    load_unit_observations,
    matched_controls as coabundance_matched_controls,
    stable_correlation,
    top3_mean,
)
from audit_bioaware_b9_reaction_spectral_specificity import (  # noqa: E402
    EXPECTED_UNITS,
    actual_seed_records,
    add_derived_features,
    enrich_seed_records,
    identity_profile,
    load_negative,
    load_positive,
    load_query_tensors,
    load_relations,
    load_seed_spectra,
    mean_views,
    select_controls,
)


VIEWS = (
    "truncated_direct", "neutral_loss", "modified_cosine", "dual_view",
    "kgmn_support", "matched_fragments",
)
FORBIDDEN_COLUMNS = {
    "truth_candidate_id", "truth_formula", "is_positive", "baseline_correct",
    "corrected", "introduced", "final_correct", "delta", "spectral_score",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary, index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_spectral_profiles(
    seeds: dict[str, list[dict]],
    polarity: str,
    adduct: str,
    allowed_identities: set[str],
) -> dict[str, tuple[list[dict], dict[str, Any]]]:
    profiles: dict[str, tuple[list[dict], dict[str, Any]]] = {}
    for identity, records in seeds.items():
        if str(identity) not in allowed_identities:
            continue
        local = [
            record for record in records
            if str(record["polarity"]) == polarity and str(record["adduct"]) == adduct
        ]
        if local:
            profiles[str(identity)] = (local, identity_profile(local))
    return profiles


def spectral_edge_evidence(
    *,
    candidate: str,
    candidate_formula: str,
    neighbour: str,
    query_id: str,
    query_tensor: np.ndarray,
    polarity: str,
    adduct: str,
    profiles: dict[str, tuple[list[dict], dict[str, Any]]],
    seeds: dict[str, list[dict]],
    relations: dict[tuple[str, str], list[tuple[str, str, str]]],
    relation_neighbours: dict[str, set[str]],
    pair_cache: dict[tuple[str, int], dict[str, float]],
) -> dict[str, Any]:
    real_records = actual_seed_records(
        candidate, neighbour, candidate_formula, polarity, adduct, relations, seeds
    )
    if not real_records:
        return {"spectral_available": False, "spectral_control_tier": -1}
    selected = select_controls(
        real_records,
        profiles,
        forbidden=set(relation_neighbours.get(candidate, set())) | {candidate, neighbour},
    )
    if selected is None:
        return {"spectral_available": False, "spectral_control_tier": -1}
    controls, tier = selected
    real = mean_views(query_id, query_tensor, real_records, pair_cache)
    control_values = [
        mean_views(query_id, query_tensor, records, pair_cache)
        for _identity, records in controls
    ]
    control = {
        view: float(np.mean([item[view] for item in control_values]))
        for view in real
    }
    output: dict[str, Any] = {
        "spectral_available": True,
        "spectral_control_tier": int(tier),
        "spectral_real_record_count": int(len(real_records)),
        "spectral_control_identities": ";".join(identity for identity, _records in controls),
    }
    for view in VIEWS:
        output[f"spectral_real_{view}"] = float(real[view])
        output[f"spectral_control_{view}"] = float(control[view])
        output[f"spectral_excess_{view}"] = float(real[view] - control[view])
    return output


def coabundance_edge_evidence(
    *,
    candidate: str,
    neighbour: str,
    query_profile: np.ndarray,
    polarity: str,
    fold_seeds: set[str],
    seed_profiles: dict[str, dict],
    relation_neighbours: dict[str, set[str]],
    degree: dict[str, int],
) -> dict[str, Any]:
    key = f"{neighbour}|{polarity}"
    if key not in seed_profiles:
        return {"coabundance_available": False}
    controls = coabundance_matched_controls(
        neighbour, candidate, polarity, fold_seeds, seed_profiles,
        relation_neighbours, degree, controls=3,
    )
    if len(controls) != 3:
        return {"coabundance_available": False}
    real = stable_correlation(query_profile, seed_profiles[key]["profile"])
    control_values = [
        stable_correlation(query_profile, seed_profiles[f"{identity}|{polarity}"]["profile"])
        for identity in controls
    ]
    output: dict[str, Any] = {
        "coabundance_available": True,
        "coabundance_control_identities": ";".join(controls),
    }
    for name in ("signed", "absolute", "positive", "negative", "sign_stability"):
        control_mean = float(np.mean([value[name] for value in control_values]))
        output[f"coabundance_real_{name}"] = float(real[name])
        output[f"coabundance_control_{name}"] = control_mean
        output[f"coabundance_excess_{name}"] = float(real[name] - control_mean)
        for index, value in enumerate(control_values):
            output[f"coabundance_control_{index}_{name}"] = float(value[name])
    return output


def aggregate_spectral(rows: list[dict[str, Any]]) -> dict[str, float]:
    available = [row for row in rows if row.get("spectral_available", False)]
    output = {
        "edge_available": float(bool(available)),
        "matched_edges": float(len(available)),
        "control_tier0_fraction": float(np.mean([row["spectral_control_tier"] == 0 for row in available])) if available else 0.0,
        "control_tier1_fraction": float(np.mean([row["spectral_control_tier"] == 1 for row in available])) if available else 0.0,
    }
    for view in VIEWS:
        for kind in ("real", "control", "excess"):
            values = [row[f"spectral_{kind}_{view}"] for row in available]
            output[f"{view}_{kind}_mean"] = float(np.mean(values)) if values else 0.0
        values = [row[f"spectral_excess_{view}"] for row in available]
        output[f"{view}_excess_best"] = float(max(values)) if values else 0.0
    return output


def aggregate_coabundance(rows: list[dict[str, Any]]) -> dict[str, float]:
    available = [row for row in rows if row.get("coabundance_available", False)]
    names = ("absolute", "positive", "negative", "sign_stability")
    actual = {
        name: top3_mean([row[f"coabundance_real_{name}"] for row in available])
        for name in names
    }
    random = {
        name: float(np.mean([
            top3_mean([row[f"coabundance_control_{index}_{name}"] for row in available])
            for index in range(3)
        ])) if available else 0.0
        for name in names
    }
    return {
        "available": float(bool(available)),
        "neighbours": float(len(available)),
        **{f"actual_{name}": float(value) for name, value in actual.items()},
        **{f"random_{name}": float(value) for name, value in random.items()},
        "abs_excess": float(actual["absolute"] - random["absolute"]),
        "positive_excess": float(actual["positive"] - random["positive"]),
        "negative_excess": float(actual["negative"] - random["negative"]),
        "multiwitness": float(len(available) >= 2),
    }


def compare_replay(
    calculated: pd.DataFrame,
    frozen_path: Path,
    columns: list[str],
    label: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    frozen = pd.read_csv(frozen_path)
    keys = ["query_id", "candidate_id", "fold"]
    merged = calculated[keys + columns].merge(
        frozen[keys + columns], on=keys, suffixes=("_new", "_old"), validate="one_to_one"
    )
    if len(merged) != len(calculated):
        raise RuntimeError(f"{label} frozen replay lost rows")
    maximum = 0.0
    mismatches = 0
    for column in columns:
        difference = np.abs(
            merged[f"{column}_new"].to_numpy(float)
            - merged[f"{column}_old"].to_numpy(float)
        )
        maximum = max(maximum, float(difference.max(initial=0.0)))
        mismatches += int((difference > tolerance).sum())
    if mismatches:
        raise RuntimeError(
            f"{label} candidate-context replay failed: mismatches={mismatches} max={maximum}"
        )
    return {"rows": int(len(merged)), "maximum_absolute_error": float(maximum)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b39-m0-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_v3_v1",
    )
    parser.add_argument(
        "--negative-query-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_units_v2",
    )
    parser.add_argument(
        "--negative-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--negative-path-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_paths_v1",
    )
    parser.add_argument(
        "--negative-edge0-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_edge_step0_v1",
    )
    parser.add_argument(
        "--manifest-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1",
    )
    parser.add_argument(
        "--network-dir", type=Path,
        default=ROOT / "data/reference/metdna2_emrn_network_20260828",
    )
    parser.add_argument(
        "--rhea-pairs", type=Path,
        default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz",
    )
    parser.add_argument(
        "--b3-rotations", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/reaction_coabundance_rotations.csv.gz",
    )
    parser.add_argument(
        "--b9-rotations", type=Path,
        default=ROOT / "data/validation/bioaware_b9_local_fullcheck_20260907_v1/reaction_spectral_rotations.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--max-queries-per-unit", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    event_path = args.b39_m0_dir / "atomic_reaction_events.csv.gz"
    report_path = args.b39_m0_dir / "report.json"
    required = [
        event_path, report_path, args.b3_rotations, args.b9_rotations,
        args.network_dir / "metdna2_emrn_edges.csv.gz", args.rhea_pairs,
    ]
    for unit in EXPECTED_UNITS:
        required.append(args.manifest_root / unit / "identity_splits.csv.gz")
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    b39_report = json.loads(report_path.read_text(encoding="utf-8"))
    if b39_report.get("status") != "bioaware_b39_m0_atomic_event_ledger_complete" or not b39_report.get("pass_to_b39_m1"):
        raise RuntimeError("B39-M1 requires a passing B39-M0 ledger")
    if b39_report.get("provenance", {}).get("atomic_events") != sha256(event_path):
        raise RuntimeError("B39-M0 event provenance mismatch")

    atomic_events = pd.read_csv(event_path, dtype={"reaction_id": str}, low_memory=False)
    internal_events = atomic_events.loc[atomic_events["context_class"].eq("S")].copy()
    internal_events["fold"] = pd.to_numeric(
        internal_events["seed_stratum"].str.extract(r"fold=(\d+)")[0], errors="raise"
    ).astype(int)
    event_pairs = internal_events[
        ["source", "unit_id", "polarity", "query_id", "candidate_id", "fold", "seed_identity"]
    ].drop_duplicates()
    pair_keys = set(
        event_pairs[["query_id", "candidate_id", "fold", "seed_identity"]]
        .astype({"query_id": str, "candidate_id": str, "fold": int, "seed_identity": str})
        .itertuples(index=False, name=None)
    )
    neighbours_by_rotation: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for query_id, candidate_id, fold, seed_identity in sorted(pair_keys):
        neighbours_by_rotation[(query_id, candidate_id, int(fold))].append(seed_identity)

    positive, positive_provenance = load_positive(args.positive_root)
    negative, negative_provenance = load_negative(
        args.negative_candidates, args.negative_path_root, args.negative_edge0_root
    )
    candidates = add_derived_features(pd.concat([positive, negative], ignore_index=True))
    relations, relation_report = load_relations(args.network_dir, args.rhea_pairs)
    relation_neighbours: dict[str, set[str]] = defaultdict(set)
    for left, right in relations:
        relation_neighbours[str(left)].add(str(right))
    degree = {identity: len(neighbours) for identity, neighbours in relation_neighbours.items()}

    evidence_rows: list[dict[str, Any]] = []
    replay_spectral_rows: list[dict[str, Any]] = []
    replay_coabundance_rows: list[dict[str, Any]] = []
    pair_cache: dict[tuple[str, int], dict[str, float]] = {}
    unit_reports: dict[str, Any] = {}
    for unit in EXPECTED_UNITS:
        source = unit.split("__", 1)[0]
        positive_unit = args.positive_root / unit
        negative_unit = args.negative_query_root / unit
        manifest_unit = args.manifest_root / unit
        splits = pd.read_csv(manifest_unit / "identity_splits.csv.gz")
        query_meta, query_tensors = load_query_tensors(positive_unit, negative_unit)
        query_meta["query_id"] = query_meta["query_id"].astype(str)
        query_meta_index = query_meta.set_index("query_id")
        raw_seeds, _seed_report = load_seed_spectra(
            positive_unit, manifest_unit, args.ppm, args.rt_sec
        )
        spectral_seeds = enrich_seed_records(raw_seeds, relation_neighbours)
        spectral_profile_cache: dict[
            tuple[int, str, str], dict[str, tuple[list[dict], dict[str, Any]]]
        ] = {}
        _abundance_meta, query_profiles, abundance_seeds = load_unit_observations(
            manifest_unit, positive_unit, negative_unit, source
        )
        local = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        if args.max_queries_per_unit > 0:
            keep = sorted(local["query_id"].astype(str).unique())[: args.max_queries_per_unit]
            local = local.loc[local["query_id"].astype(str).isin(keep)].copy()
        local_pair_attempts = local_pairs_written = 0
        for fold in range(10):
            fold_seeds = set(splits.loc[
                splits["fold"].eq(fold) & splits["role"].eq("seed"), "ik14"
            ].astype(str))
            heldout = set(splits.loc[
                splits["fold"].eq(fold) & splits["role"].eq("heldout"), "ik14"
            ].astype(str))
            fold_candidates = local.loc[local["truth_candidate_id"].astype(str).isin(heldout)]
            for row in fold_candidates.itertuples(index=False):
                query_id = str(row.query_id)
                candidate = str(row.candidate_id)
                meta = query_meta_index.loc[query_id]
                polarity = str(meta.polarity)
                adduct = str(meta.adduct)
                profile_key = (fold, polarity, adduct)
                if profile_key not in spectral_profile_cache:
                    spectral_profile_cache[profile_key] = build_spectral_profiles(
                        spectral_seeds, polarity, adduct, fold_seeds
                    )
                profiles = spectral_profile_cache[profile_key]
                neighbours = neighbours_by_rotation.get((query_id, candidate, fold), [])
                rows_for_candidate: list[dict[str, Any]] = []
                for neighbour in neighbours:
                    local_pair_attempts += 1
                    spectral = spectral_edge_evidence(
                        candidate=candidate,
                        candidate_formula=str(row.truth_formula),
                        neighbour=neighbour,
                        query_id=query_id,
                        query_tensor=query_tensors[query_id],
                        polarity=polarity,
                        adduct=adduct,
                        profiles=profiles,
                        seeds=spectral_seeds,
                        relations=relations,
                        relation_neighbours=relation_neighbours,
                        pair_cache=pair_cache,
                    )
                    coabundance = coabundance_edge_evidence(
                        candidate=candidate,
                        neighbour=neighbour,
                        query_profile=query_profiles[query_id]["profile"],
                        polarity=polarity,
                        fold_seeds=fold_seeds,
                        seed_profiles=abundance_seeds,
                        relation_neighbours=relation_neighbours,
                        degree=degree,
                    )
                    edge_row = {
                        "source": source,
                        "unit_id": unit,
                        "polarity": str(row.polarity),
                        "query_id": query_id,
                        "candidate_id": candidate,
                        "seed_stratum": f"fold={fold}",
                        "fold": int(fold),
                        "seed_identity": neighbour,
                        "evidence_resolution": "candidate_seed_edge_not_reaction_id_specific",
                        **spectral,
                        **coabundance,
                    }
                    evidence_rows.append(edge_row)
                    rows_for_candidate.append(edge_row)
                    local_pairs_written += 1
                replay_spectral_rows.append({
                    "query_id": query_id, "candidate_id": candidate, "fold": int(fold),
                    **aggregate_spectral(rows_for_candidate),
                })
                replay_coabundance_rows.append({
                    "query_id": query_id, "candidate_id": candidate, "fold": int(fold),
                    **aggregate_coabundance(rows_for_candidate),
                })
        unit_reports[unit] = {
            "candidate_queries": int(local["query_id"].nunique()),
            "candidate_rows": int(len(local)),
            "edge_pairs_attempted": int(local_pair_attempts),
            "edge_pairs_written": int(local_pairs_written),
        }
        print(
            f"[B39-M1 {unit}] pairs={local_pairs_written:,} queries={local['query_id'].nunique():,}",
            flush=True,
        )

    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        raise RuntimeError("B39-M1 produced no internal edge evidence")
    if evidence.duplicated(["query_id", "candidate_id", "fold", "seed_identity"]).any():
        raise RuntimeError("B39-M1 produced duplicate candidate-seed edge rows")
    if FORBIDDEN_COLUMNS & set(evidence.columns):
        raise RuntimeError("ranking outcome leaked into B39-M1 edge evidence")

    spectral_replay = pd.DataFrame(replay_spectral_rows)
    coabundance_replay = pd.DataFrame(replay_coabundance_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectral_replay_path = args.output_dir / "spectral_candidate_context_replay.csv.gz"
    coabundance_replay_path = args.output_dir / "coabundance_candidate_context_replay.csv.gz"
    atomic_csv_gzip(spectral_replay_path, spectral_replay)
    atomic_csv_gzip(coabundance_replay_path, coabundance_replay)
    spectral_columns = [
        "edge_available", "matched_edges", "control_tier0_fraction", "control_tier1_fraction",
        *[
            f"{view}_{kind}_mean"
            for view in ("truncated_direct", "neutral_loss", "modified_cosine", "dual_view", "kgmn_support")
            for kind in ("real", "control", "excess")
        ],
        *[
            f"{view}_excess_best"
            for view in ("truncated_direct", "neutral_loss", "modified_cosine", "dual_view", "kgmn_support")
        ],
    ]
    coabundance_columns = [
        "available", "neighbours", "actual_absolute", "actual_positive", "actual_negative",
        "actual_sign_stability", "random_absolute", "random_positive", "random_negative",
        "random_sign_stability", "abs_excess", "positive_excess", "negative_excess", "multiwitness",
    ]
    if args.max_queries_per_unit == 0:
        spectral_reproduction = compare_replay(
            spectral_replay, args.b9_rotations, spectral_columns, "B9 spectral"
        )
        coabundance_reproduction = compare_replay(
            coabundance_replay, args.b3_rotations, coabundance_columns, "B3 co-abundance"
        )
    else:
        # Smoke mode uses a strict subset, so compare_replay's one-to-one row
        # count contract is intentionally not invoked.
        spectral_reproduction = {"rows": int(len(spectral_replay)), "smoke_subset": True}
        coabundance_reproduction = {"rows": int(len(coabundance_replay)), "smoke_subset": True}

    joined = internal_events.merge(
        evidence,
        on=["source", "unit_id", "polarity", "query_id", "candidate_id", "seed_stratum", "fold", "seed_identity"],
        how="left", validate="many_to_one", suffixes=("", "_edge"),
    )
    joined["edge_evidence_row_available"] = joined["evidence_resolution"].notna()
    joined["spectral_available"] = joined["spectral_available"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    joined["coabundance_available"] = joined["coabundance_available"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    joined["both_experimental_layers_available"] = (
        joined["spectral_available"] & joined["coabundance_available"]
    )
    joined["event_specific_spectral_evidence_available"] = joined["spectral_available"]
    joined["event_specific_coabundance_evidence_available"] = joined["coabundance_available"]
    joined["experimental_evidence_resolution"] = np.where(
        joined["edge_evidence_row_available"],
        "candidate_seed_edge_not_reaction_id_specific",
        "absent",
    )
    drop_legacy = [
        column for column in joined.columns
        if column.startswith("legacy_spectral_") or column.startswith("legacy_coabundance_")
    ]
    joined = joined.drop(columns=drop_legacy)
    if FORBIDDEN_COLUMNS & set(joined.columns):
        raise RuntimeError("ranking outcome leaked into B39-M1 joined ledger")

    edge_path = args.output_dir / "candidate_seed_edge_evidence.csv.gz"
    joined_path = args.output_dir / "internal_atomic_events_with_edge_evidence.csv.gz"
    atomic_csv_gzip(edge_path, evidence)
    atomic_csv_gzip(joined_path, joined)
    report = {
        "status": "bioaware_b39_m1_internal_edge_evidence_complete",
        "formal": args.max_queries_per_unit == 0,
        "model_fitted": False,
        "embedding_values_read": False,
        "ranking_outcomes_read": False,
        "sources": ["BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"],
        "edge_pairs": int(len(evidence)),
        "atomic_event_rows": int(len(joined)),
        "spectral_edge_pairs": int(evidence["spectral_available"].fillna(False).sum()),
        "coabundance_edge_pairs": int(evidence["coabundance_available"].fillna(False).sum()),
        "both_layer_edge_pairs": int(
            (evidence["spectral_available"].fillna(False) & evidence["coabundance_available"].fillna(False)).sum()
        ),
        "atomic_events_with_spectral_evidence": int(joined["spectral_available"].sum()),
        "atomic_events_with_coabundance_evidence": int(joined["coabundance_available"].sum()),
        "atomic_events_with_both_layers": int(joined["both_experimental_layers_available"].sum()),
        "historical_cache_reproduction": {
            "B9_spectral": spectral_reproduction,
            "B3_coabundance": coabundance_reproduction,
        },
        "unit_reports": unit_reports,
        "relation_catalog": relation_report,
        "external_source_status": {
            "ST001154_same_formula_10ppm": "not_computed_in_M1_internal; raw sample-local edge evidence required",
            "KGMN200STD_hidden_seed": "not_computed_in_M1_internal; raw hidden-standard spectral edge evidence required",
        },
        "pass_to_b39_m2_internal_fixed_action": bool(
            args.max_queries_per_unit == 0
            and int(joined["both_experimental_layers_available"].sum()) > 0
        ),
        "contracts": {
            "event_resolution": "candidate-seed edge; repeated reaction IDs remain one dependent physical pair",
            "matched_non_neighbour_controls_per_edge": 3,
            "truth_used_only_to_replay_existing_heldout_rotations": True,
            "ranking_outcomes_used": False,
            "external_missingness_zero_imputed": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "b39_m0_report": sha256(report_path),
            "b39_m0_events": sha256(event_path),
            "positive": positive_provenance,
            "negative": negative_provenance,
            "network_edges": sha256(args.network_dir / "metdna2_emrn_edges.csv.gz"),
            "rhea_pairs": sha256(args.rhea_pairs),
            "b3_rotations": sha256(args.b3_rotations),
            "b9_rotations": sha256(args.b9_rotations),
            "spectral_candidate_context_replay": sha256(spectral_replay_path),
            "coabundance_candidate_context_replay": sha256(coabundance_replay_path),
            "edge_evidence": sha256(edge_path),
            "joined_events": sha256(joined_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B39-M1 reconstructs internal candidate-seed edge evidence and matched controls. "
            "It is not an action outcome, a prospective sample test, embedding improvement, "
            "biological causality, or SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
