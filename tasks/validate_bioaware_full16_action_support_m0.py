#!/usr/bin/env python
"""Independent, fail-closed validation of BioAware full-16 M0 outputs.

The validator recomputes upstream hashes, truth rows, candidate membership,
network summaries, source overlap, LOSO purges, and every scientific gate.
It intentionally fits no model and reads no embedding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = ("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma")
EXPECTED_SEPARATIONS = ("hilic", "rplc")
EXPECTED_POLARITIES = ("negative", "positive")
EXPECTED_PANELS = {
    f"{source}__{separation}__{polarity}"
    for source in EXPECTED_SOURCES
    for separation in EXPECTED_SEPARATIONS
    for polarity in EXPECTED_POLARITIES
}
NETWORKS = ("rhea", "kegg", "union")
AMBIGUOUS_BOOL = tuple(
    f"{network}_{suffix}"
    for network in NETWORKS
    for suffix in (
        "any_candidate_node", "any_candidate_supported", "support_discriminative",
        "truth_advantaged", "wrong_advantaged",
    )
)
CANDIDATE_BOOL = (
    "truth_present", "ambiguous", "is_truth", "rhea_node", "kegg_node", "union_node",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: Iterable[object]) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value)
        for value in values
    ], dtype=object)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def require_real_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{label} must be a JSON boolean")
    return value


def require_bool_column(frame: pd.DataFrame, column: str, *, allow_na: bool = False) -> None:
    if column not in frame:
        raise RuntimeError(f"missing boolean column: {column}")
    if not allow_na and frame[column].isna().any():
        raise RuntimeError(f"null values in boolean column: {column}")
    if not all(isinstance(value, (bool, np.bool_)) for value in frame[column].dropna().tolist()):
        raise RuntimeError(f"non-boolean values in column: {column}")


def assert_nested_equal(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"{label}: dictionary keys differ")
        for key in expected:
            assert_nested_equal(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"{label}: list differs")
        for position, (left, right) in enumerate(zip(observed, expected)):
            assert_nested_equal(left, right, f"{label}[{position}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not np.isclose(float(observed), float(expected), rtol=1e-10, atol=1e-12):
            raise RuntimeError(f"{label}: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise RuntimeError(f"{label}: {observed!r} != {expected!r}")


def finite_quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": int(len(array)), "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)), "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)), "max": float(array.max()),
    }


def network_summary(ambiguous: pd.DataFrame, network: str) -> dict[str, object]:
    outcomes = ambiguous[f"{network}_outcome"].value_counts().to_dict()
    supported = ambiguous.loc[ambiguous[f"{network}_any_candidate_supported"]]
    return {
        "queries_with_any_candidate_node": int(ambiguous[f"{network}_any_candidate_node"].sum()),
        "queries_with_strict_seed_support": int(ambiguous[f"{network}_any_candidate_supported"].sum()),
        "identities_with_strict_seed_support": int(supported.truth_ik14.nunique()),
        "formulas_with_strict_seed_support": int(supported.truth_formula.nunique()),
        "support_discriminative_queries": int(ambiguous[f"{network}_support_discriminative"].sum()),
        "truth_advantaged_queries": int(outcomes.get("truth_advantaged", 0)),
        "wrong_advantaged_queries": int(outcomes.get("wrong_advantaged", 0)),
        "tied_queries": int(outcomes.get("tied", 0)),
        "truth_advantage_headroom_rate_all_ambiguous": float(
            ambiguous[f"{network}_truth_advantaged"].mean()
        ),
        "truth_advantage_rate_among_supported": float(
            supported[f"{network}_truth_advantaged"].mean()
        ) if len(supported) else 0.0,
    }


def subgroup_summary(frame: pd.DataFrame) -> dict[str, object]:
    ambiguous = frame.loc[frame.ambiguous]
    payload: dict[str, object] = {
        "level1_rows": int(len(frame)),
        "level1_identities": int(frame.truth_ik14.nunique()),
        "level1_formulas": int(frame.truth_formula.nunique()),
        "truth_covered_rows": int(frame.truth_present.sum()),
        "truth_covered_identities": int(frame.loc[frame.truth_present, "truth_ik14"].nunique()),
        "ambiguous_rows": int(len(ambiguous)),
        "ambiguous_identities": int(ambiguous.truth_ik14.nunique()),
        "ambiguous_formulas": int(ambiguous.truth_formula.nunique()),
        "candidate_molecules": finite_quantiles(ambiguous.candidate_molecules),
    }
    payload["network"] = {
        network: network_summary(ambiguous, network) for network in NETWORKS
    } if len(ambiguous) else {}
    return payload


def recompute_overlap(ambiguous: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    by_source = {str(key): value for key, value in ambiguous.groupby("source", sort=True)}
    empty = ambiguous.iloc[0:0]
    for source_a in EXPECTED_SOURCES:
        for source_b in EXPECTED_SOURCES:
            left = by_source.get(source_a, empty)
            right = by_source.get(source_b, empty)
            records.append({
                "source_a": source_a, "source_b": source_b,
                "identity_overlap": int(len(set(left.truth_ik14) & set(right.truth_ik14))),
                "formula_overlap": int(len(set(left.truth_formula) & set(right.truth_formula))),
            })
    return pd.DataFrame(records)


def recompute_loso(ambiguous: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for held_source in EXPECTED_SOURCES:
        held = ambiguous.loc[ambiguous.source == held_source]
        held_full = full.loc[full.source == held_source]
        train = ambiguous.loc[ambiguous.source != held_source]
        purged = train.loc[
            ~train.truth_ik14.isin(set(held_full.truth_ik14))
            & ~train.truth_formula.isin(set(held_full.truth_formula))
        ]
        supported = purged.loc[purged.union_any_candidate_supported]
        records.append({
            "held_source": held_source,
            "held_rows": int(len(held)),
            "held_identities": int(held.truth_ik14.nunique()),
            "held_formulas": int(held.truth_formula.nunique()),
            "held_full_level1_rows_used_for_purge": int(len(held_full)),
            "held_full_level1_identities_used_for_purge": int(held_full.truth_ik14.nunique()),
            "held_full_level1_formulas_used_for_purge": int(held_full.truth_formula.nunique()),
            "train_rows_before_purge": int(len(train)),
            "train_rows_after_identity_formula_purge": int(len(purged)),
            "train_identities_after_purge": int(purged.truth_ik14.nunique()),
            "train_formulas_after_purge": int(purged.truth_formula.nunique()),
            "train_sources_after_purge": int(purged.source.nunique()),
            "train_panels_after_purge": int(purged.panel_id.nunique()),
            "train_union_supported_rows_after_purge": int(len(supported)),
            "train_union_supported_identities_after_purge": int(supported.truth_ik14.nunique()),
            "train_union_supported_formulas_after_purge": int(supported.truth_formula.nunique()),
        })
    return pd.DataFrame(records)


def sorted_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.sort_values(columns).reset_index(drop=True).sort_index(axis=1)


def load_expected_truth(manifest_root: Path, manifest_report: dict[str, object]) -> pd.DataFrame:
    parts = []
    for unit_id in sorted(manifest_report.get("units", {})):
        path = manifest_root / unit_id / "external_level1.csv.gz"
        require_file(path)
        frame = pd.read_csv(path)
        frame["manifest_row"] = np.arange(len(frame), dtype=np.int64)
        frame["query_id"] = [
            f"{unit_id}|{panel}|{position:05d}"
            for panel, position in zip(frame.panel_id.astype(str), frame.manifest_row)
        ]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def build_reference_index(
    hdf5_path: Path, negative_manifest_path: Path, approved_path: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
    with h5py.File(hdf5_path, "r") as handle:
        required = {"precursor_mz", "adduct", "INCHIKEY", "FORMULA"}
        if not required.issubset(handle.keys()):
            raise RuntimeError("reference HDF5 schema changed")
        pos_mz = np.asarray(handle["precursor_mz"][:], dtype=float)
        pos_adduct = decode(handle["adduct"][:])
        pos_ik14 = np.asarray([value[:14].upper() for value in decode(handle["INCHIKEY"][:])])
        pos_formula = decode(handle["FORMULA"][:])
    pos_valid = (
        np.isfinite(pos_mz) & (pos_mz > 0)
        & (np.char.str_len(pos_ik14.astype(str)) == 14)
        & (pos_adduct.astype(str) != "nan")
    )
    negative = pd.read_csv(negative_manifest_path)
    if not {"inchikey", "precursor_mz"}.issubset(negative):
        raise RuntimeError("negative manifest schema changed")
    neg_mz_all = pd.to_numeric(negative.precursor_mz, errors="coerce").to_numpy(float)
    neg_ik14_all = negative.inchikey.fillna("").astype(str).str[:14].str.upper().to_numpy(object)
    approved = np.load(approved_path, allow_pickle=False)
    if approved.ndim != 1 or not np.issubdtype(approved.dtype, np.integer):
        raise RuntimeError("negative approved-row index schema changed")
    if len(approved) == 0 or approved.min() < 0 or approved.max() >= len(negative):
        raise RuntimeError("negative approved-row index is out of range")
    if len(np.unique(approved)) != len(approved):
        raise RuntimeError("negative approved-row index contains duplicates")
    neg_valid = np.zeros(len(negative), dtype=bool)
    neg_valid[approved.astype(np.int64)] = True
    neg_valid &= (
        np.isfinite(neg_mz_all) & (neg_mz_all >= 50.0) & (neg_mz_all <= 1500.0)
        & (np.char.str_len(neg_ik14_all.astype(str)) == 14)
    )
    mz = np.concatenate([pos_mz[pos_valid], neg_mz_all[neg_valid]])
    adduct = np.concatenate([
        pos_adduct[pos_valid], np.full(int(neg_valid.sum()), "[M-H]-", dtype=object),
    ])
    ik14 = np.concatenate([pos_ik14[pos_valid], neg_ik14_all[neg_valid]])
    formula = np.concatenate([
        pos_formula[pos_valid], np.full(int(neg_valid.sum()), "", dtype=object),
    ])
    source = np.concatenate([
        np.full(int(pos_valid.sum()), "massspecgym_hdf5", dtype=object),
        np.full(int(neg_valid.sum()), "mona_negative_manifest", dtype=object),
    ])
    index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for value in sorted(set(adduct.astype(str))):
        positions = np.flatnonzero(adduct.astype(str) == value)
        order = np.argsort(mz[positions], kind="stable")
        positions = positions[order]
        index[value] = (mz[positions], ik14[positions], formula[positions], source[positions])
    return index, {
        "reference_rows": int(len(mz)),
        "identities": int(len(set(ik14))),
        "formula_values_available": int(len({str(x) for x in formula if str(x) not in {"", "nan"}})),
        "supported_adducts": sorted(index),
        "positive_input_rows": int(len(pos_mz)),
        "positive_valid_rows": int(pos_valid.sum()),
        "positive_identities": int(len(set(pos_ik14[pos_valid]))),
        "positive_adducts": sorted(set(pos_adduct[pos_valid].astype(str))),
        "negative_input_rows": int(len(negative)),
        "negative_valid_rows": int(neg_valid.sum()),
        "negative_approved_rows": int(len(approved)),
        "negative_identities": int(len(set(neg_ik14_all[neg_valid]))),
    }


def expected_candidates(
    index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    mz: float, adduct: str, ppm: float,
) -> dict[str, tuple[str, int, str, float]]:
    if adduct not in index or not np.isfinite(mz) or mz <= 0:
        return {}
    masses, identities, formulas, sources = index[adduct]
    tolerance = mz * ppm * 1e-6
    left = int(np.searchsorted(masses, mz - tolerance, side="left"))
    right = int(np.searchsorted(masses, mz + tolerance, side="right"))
    grouped: dict[str, list[int]] = defaultdict(list)
    for position in range(left, right):
        grouped[str(identities[position])].append(position)
    result: dict[str, tuple[str, int, str, float]] = {}
    for identity, positions in grouped.items():
        observed_formulas = {
            str(formulas[position]) for position in positions
            if str(formulas[position]) not in {"", "nan", "None"}
        }
        if len(observed_formulas) > 1:
            raise RuntimeError(f"reference identity has conflicting formulas: {identity}")
        result[identity] = (
            next(iter(observed_formulas), ""), len(positions),
            "|".join(sorted({str(sources[position]) for position in positions})),
            min(abs(float(masses[position]) - mz) for position in positions) / mz * 1e6,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1")
    parser.add_argument("--reference-hdf5", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--negative-library-manifest", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv")
    parser.add_argument("--negative-library-report", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/report.json")
    parser.add_argument("--negative-library-integrity-report", type=Path, default=ROOT / "data/validation/mona_negative_library_chemical_integrity_v1/report.json")
    parser.add_argument("--negative-approved-rows", type=Path, default=ROOT / "data/validation/mona_negative_library_chemical_integrity_v1/approved_m_h_library_rows.npy")
    parser.add_argument("--rhea-relations", type=Path, default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz")
    parser.add_argument("--kegg-edges", type=Path, default=ROOT / "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz")
    parser.add_argument("--audit-script", type=Path, default=ROOT / "tasks/audit_bioaware_full16_action_support_m0.py")
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()
    if not np.isclose(args.ppm, 10.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("validator accepts only the frozen 10 ppm protocol")

    report_path = args.result_dir / "report.json"
    queries_path = args.result_dir / "queries.csv.gz"
    candidates_path = args.result_dir / "candidate_identities.csv.gz"
    overlap_path = args.result_dir / "source_overlap.csv"
    loso_path = args.result_dir / "source_loso.csv"
    upstream = (
        args.manifest_root / "report.json", args.reference_hdf5,
        args.negative_library_manifest, args.negative_library_report,
        args.negative_library_integrity_report, args.negative_approved_rows,
        args.rhea_relations, args.kegg_edges, args.audit_script,
    )
    for path in (report_path, queries_path, candidates_path, overlap_path, loso_path, *upstream):
        require_file(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "bioaware_full16_m0_v1":
        raise RuntimeError("unexpected or missing M0 schema_version")
    if report.get("status") != "bioaware_full16_action_support_m0_complete":
        raise RuntimeError("unexpected M0 status")
    for key, expected in (
        ("formal", True), ("model_fitted", False), ("embedding_values_read", False),
        ("opened_external_manifest_is_development_data", True),
    ):
        if require_real_bool(report.get(key), key) is not expected:
            raise RuntimeError(f"M0 contract mismatch: {key}")

    provenance = report.get("provenance", {})
    paths_by_key = {
        "manifest_report_sha256": args.manifest_root / "report.json",
        "reference_hdf5_sha256": args.reference_hdf5,
        "negative_library_manifest_sha256": args.negative_library_manifest,
        "negative_library_report_sha256": args.negative_library_report,
        "negative_library_integrity_report_sha256": args.negative_library_integrity_report,
        "negative_approved_rows_sha256": args.negative_approved_rows,
        "rhea_relations_sha256": args.rhea_relations,
        "kegg_edges_sha256": args.kegg_edges,
        "script_sha256": args.audit_script,
        "queries_sha256": queries_path,
        "candidate_identities_sha256": candidates_path,
        "source_overlap_sha256": overlap_path,
        "source_loso_sha256": loso_path,
    }
    for key, path in paths_by_key.items():
        if provenance.get(key) != sha256(path):
            raise RuntimeError(f"current-file provenance mismatch: {key}")

    manifest_report = json.loads((args.manifest_root / "report.json").read_text(encoding="utf-8"))
    if manifest_report.get("status") != "bioaware_metdna3_external_manifest_frozen":
        raise RuntimeError("external manifest status changed")
    expected_units = {
        f"{source}__{separation}"
        for source in EXPECTED_SOURCES for separation in EXPECTED_SEPARATIONS
    }
    if set(manifest_report.get("units", {})) != expected_units:
        raise RuntimeError("external manifest units changed")
    truth_hashes = {}
    for unit_id in sorted(expected_units):
        path = args.manifest_root / unit_id / "external_level1.csv.gz"
        require_file(path)
        truth_hashes[unit_id] = sha256(path)
        if manifest_report["units"][unit_id].get("truth_sha256") != truth_hashes[unit_id]:
            raise RuntimeError(f"manifest truth hash changed: {unit_id}")
    assert_nested_equal(provenance.get("truth_file_sha256"), truth_hashes, "truth_file_sha256")

    negative_report = json.loads(args.negative_library_report.read_text(encoding="utf-8"))
    integrity = json.loads(args.negative_library_integrity_report.read_text(encoding="utf-8"))
    if negative_report.get("status") != "bioaware_metdna3_external_negative_dreams_complete":
        raise RuntimeError("negative benchmark status changed")
    if negative_report.get("library_audit", {}).get("chemically_filtered_for_m_h") is not True:
        raise RuntimeError("negative benchmark is not chemically filtered")
    if integrity.get("status") != "mona_negative_library_chemical_integrity_complete":
        raise RuntimeError("negative integrity status changed")
    if integrity.get("declared_adduct_scope") != "[M-H]-":
        raise RuntimeError("negative adduct scope changed")
    negative_manifest_hash = sha256(args.negative_library_manifest)
    approved_hash = sha256(args.negative_approved_rows)
    if negative_report.get("provenance", {}).get("library_manifest_sha256") != negative_manifest_hash:
        raise RuntimeError("negative manifest disagrees with benchmark")
    if integrity.get("provenance", {}).get("manifest_sha256") != negative_manifest_hash:
        raise RuntimeError("negative manifest disagrees with integrity audit")
    if integrity.get("provenance", {}).get("approved_rows_sha256") != approved_hash:
        raise RuntimeError("approved rows disagree with integrity audit")
    if negative_report.get("provenance", {}).get("approved_library_rows_sha256") != approved_hash:
        raise RuntimeError("approved rows disagree with negative benchmark")

    queries = pd.read_csv(queries_path)
    candidates = pd.read_csv(candidates_path, keep_default_na=False)
    overlap = pd.read_csv(overlap_path)
    loso = pd.read_csv(loso_path)
    required_query = {
        "query_id", "source", "unit_id", "panel_id", "separation", "polarity",
        "precursor_mz", "adduct", "truth_ik14", "truth_formula", "truth_present",
        "candidate_molecules", "ambiguous", *AMBIGUOUS_BOOL,
    }
    required_candidate = {
        "query_id", "source", "unit_id", "panel_id", "separation", "polarity",
        "precursor_mz", "adduct", "truth_ik14", "truth_formula", "candidate_molecules",
        "candidate_ik14", "candidate_formula", "candidate_spectra",
        "candidate_reference_sources", "candidate_nearest_mass_error_ppm", *CANDIDATE_BOOL,
    }
    if not required_query.issubset(queries):
        raise RuntimeError(f"query schema incomplete: {sorted(required_query-set(queries))}")
    if not required_candidate.issubset(candidates):
        raise RuntimeError(f"candidate schema incomplete: {sorted(required_candidate-set(candidates))}")
    for column in ("truth_present", "ambiguous"):
        require_bool_column(queries, column)
    for column in AMBIGUOUS_BOOL:
        require_bool_column(queries.loc[queries.ambiguous], column)
    for column in CANDIDATE_BOOL:
        require_bool_column(candidates, column)

    if queries.query_id.duplicated().any():
        raise RuntimeError("duplicate query IDs")
    if set(queries.source.astype(str)) != set(EXPECTED_SOURCES):
        raise RuntimeError("source labels changed")
    if set(queries.panel_id.astype(str)) != EXPECTED_PANELS:
        raise RuntimeError("panel labels changed")
    expected_unit = queries.source.astype(str) + "__" + queries.separation.astype(str)
    expected_panel = expected_unit + "__" + queries.polarity.astype(str)
    if not queries.unit_id.astype(str).equals(expected_unit):
        raise RuntimeError("query unit/source/separation mismatch")
    if not queries.panel_id.astype(str).equals(expected_panel):
        raise RuntimeError("query panel/unit/polarity mismatch")
    expected_sign = queries.polarity.map({"positive": "+", "negative": "-"})
    if expected_sign.isna().any() or not queries.adduct.astype(str).str[-1].equals(expected_sign):
        raise RuntimeError("query polarity/adduct mismatch")

    expected_truth = load_expected_truth(args.manifest_root, manifest_report)
    truth_projection = expected_truth[[
        "query_id", "sample_type", "unit_id", "panel_id", "separation", "polarity",
        "peak_name", "mz", "adduct", "ik14", "formula",
    ]].rename(columns={
        "sample_type": "source", "mz": "precursor_mz", "ik14": "truth_ik14",
        "formula": "truth_formula",
    })
    query_projection = queries[list(truth_projection.columns)]
    pd.testing.assert_frame_equal(
        sorted_frame(query_projection, ["query_id"]),
        sorted_frame(truth_projection, ["query_id"]),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )

    reference_index, library = build_reference_index(
        args.reference_hdf5, args.negative_library_manifest, args.negative_approved_rows,
    )
    library_report = report["candidate_library"]
    for key in ("reference_rows", "identities", "formula_values_available", "supported_adducts"):
        assert_nested_equal(library_report.get(key), library[key], f"candidate_library.{key}")
    positive = library_report["reference_sources"]["massspecgym_hdf5"]
    negative = library_report["reference_sources"]["mona_negative_manifest"]
    for observed, key in (
        (positive.get("input_rows"), "positive_input_rows"),
        (positive.get("valid_rows"), "positive_valid_rows"),
        (positive.get("identities"), "positive_identities"),
        (positive.get("adducts"), "positive_adducts"),
        (negative.get("input_rows"), "negative_input_rows"),
        (negative.get("valid_rows"), "negative_valid_rows"),
        (negative.get("chemically_approved_rows"), "negative_approved_rows"),
        (negative.get("identities"), "negative_identities"),
    ):
        assert_nested_equal(observed, library[key], f"candidate_library.{key}")

    if candidates.duplicated(["query_id", "candidate_ik14"]).any():
        raise RuntimeError("duplicate query/candidate identity pairs")
    ambiguous = queries.loc[queries.ambiguous].copy()
    if set(candidates.query_id.astype(str)) != set(ambiguous.query_id.astype(str)):
        raise RuntimeError("candidate/query membership mismatch")
    if not (
        candidates.is_truth
        == (candidates.candidate_ik14.astype(str) == candidates.truth_ik14.astype(str))
    ).all():
        raise RuntimeError("is_truth is inconsistent with candidate/truth identity")
    grouped = candidates.groupby("query_id", sort=False)
    if not (grouped.is_truth.sum() == 1).all():
        raise RuntimeError("candidate groups do not contain exactly one truth")
    group_sizes = grouped.candidate_ik14.nunique()
    declared = grouped.candidate_molecules.first().astype(int)
    if not group_sizes.equals(declared) or (group_sizes < 2).any():
        raise RuntimeError("candidate group sizes are inconsistent")
    if not np.isfinite(candidates.candidate_nearest_mass_error_ppm).all():
        raise RuntimeError("candidate mass error contains non-finite values")
    if (candidates.candidate_nearest_mass_error_ppm < -1e-12).any() or (
        candidates.candidate_nearest_mass_error_ppm > args.ppm + 1e-8
    ).any():
        raise RuntimeError("candidate outside strict mass window")
    allowed_sources = {"massspecgym_hdf5", "mona_negative_manifest"}
    parsed_sources = candidates.candidate_reference_sources.map(lambda x: set(str(x).split("|")))
    if any(not values or not values.issubset(allowed_sources) for values in parsed_sources):
        raise RuntimeError("unknown candidate reference source")
    negative_rows = candidates.adduct == "[M-H]-"
    if any(values != {"mona_negative_manifest"} for values in parsed_sources.loc[negative_rows]):
        raise RuntimeError("[M-H]- candidate is not exclusively from approved MONA rows")
    if any(values != {"massspecgym_hdf5"} for values in parsed_sources.loc[~negative_rows]):
        raise RuntimeError("positive candidate is not exclusively from MassSpecGym HDF5")

    candidate_groups = {key: value for key, value in candidates.groupby("query_id", sort=False)}
    for row in queries.itertuples(index=False):
        expected = expected_candidates(reference_index, float(row.precursor_mz), str(row.adduct), args.ppm)
        expected_present = str(row.truth_ik14) in expected
        expected_ambiguous = expected_present and len(expected) >= 2
        if bool(row.truth_present) != expected_present or bool(row.ambiguous) != expected_ambiguous:
            raise RuntimeError(f"truth/ambiguity replay mismatch: {row.query_id}")
        if int(row.candidate_molecules) != len(expected):
            raise RuntimeError(f"candidate count replay mismatch: {row.query_id}")
        if not expected_ambiguous:
            continue
        group = candidate_groups[str(row.query_id)].set_index("candidate_ik14")
        if set(group.index.astype(str)) != set(expected):
            raise RuntimeError(f"candidate identity replay mismatch: {row.query_id}")
        for identity, payload in expected.items():
            expected_formula, expected_spectra, expected_sources, expected_error = payload
            observed = group.loc[identity]
            if str(observed.candidate_formula) != expected_formula:
                raise RuntimeError(f"candidate formula replay mismatch: {row.query_id}/{identity}")
            if int(observed.candidate_spectra) != expected_spectra:
                raise RuntimeError(f"candidate spectra replay mismatch: {row.query_id}/{identity}")
            if str(observed.candidate_reference_sources) != expected_sources:
                raise RuntimeError(f"candidate source replay mismatch: {row.query_id}/{identity}")
            if not np.isclose(float(observed.candidate_nearest_mass_error_ppm), expected_error, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"candidate mass-error replay mismatch: {row.query_id}/{identity}")

    metadata = [
        "source", "unit_id", "panel_id", "separation", "polarity", "peak_name",
        "precursor_mz", "adduct", "truth_ik14", "truth_formula", "truth_present",
        "candidate_molecules", "ambiguous", "strict_source_seed_identities",
    ]
    candidate_first = grouped[metadata].first().reset_index()
    query_metadata = ambiguous[["query_id", *metadata]]
    pd.testing.assert_frame_equal(
        sorted_frame(candidate_first, ["query_id"]), sorted_frame(query_metadata, ["query_id"]),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )

    for network in NETWORKS:
        truth_support = candidates.loc[candidates.is_truth].set_index("query_id")[f"{network}_strict_seed_support"]
        wrong = candidates.loc[~candidates.is_truth]
        wrong_max = wrong.groupby("query_id")[f"{network}_strict_seed_support"].max()
        any_node = candidates.groupby("query_id")[f"{network}_node"].any()
        any_supported = candidates.groupby("query_id")[f"{network}_strict_seed_support"].max() > 0
        discriminative = candidates.groupby("query_id")[f"{network}_strict_seed_support"].nunique() > 1
        check = ambiguous.set_index("query_id")
        if not check[f"{network}_truth_support"].astype(int).equals(truth_support.reindex(check.index).astype(int)):
            raise RuntimeError(f"{network} truth support mismatch")
        if not check[f"{network}_max_wrong_support"].astype(int).equals(wrong_max.reindex(check.index).astype(int)):
            raise RuntimeError(f"{network} wrong support mismatch")
        if not np.array_equal(
            check[f"{network}_any_candidate_node"].to_numpy(dtype=bool),
            any_node.reindex(check.index).to_numpy(dtype=bool),
        ):
            raise RuntimeError(f"{network} node flag mismatch")
        if not np.array_equal(
            check[f"{network}_any_candidate_supported"].to_numpy(dtype=bool),
            any_supported.reindex(check.index).to_numpy(dtype=bool),
        ):
            raise RuntimeError(f"{network} support flag mismatch")
        if not np.array_equal(
            check[f"{network}_support_discriminative"].to_numpy(dtype=bool),
            discriminative.reindex(check.index).to_numpy(dtype=bool),
        ):
            raise RuntimeError(f"{network} discriminative flag mismatch")
        truth_array = truth_support.reindex(check.index).to_numpy()
        wrong_array = wrong_max.reindex(check.index).to_numpy()
        outcome = pd.Series(np.where(
            truth_array > wrong_array, "truth_advantaged",
            np.where(truth_array < wrong_array, "wrong_advantaged", "tied"),
        ), index=check.index)
        if not np.array_equal(check[f"{network}_outcome"].astype(str).to_numpy(), outcome.to_numpy()):
            raise RuntimeError(f"{network} outcome mismatch")
        if not np.array_equal(
            check[f"{network}_truth_advantaged"].to_numpy(dtype=bool),
            (outcome == "truth_advantaged").to_numpy(dtype=bool),
        ):
            raise RuntimeError(f"{network} truth-advantage mismatch")
        if not np.array_equal(
            check[f"{network}_wrong_advantaged"].to_numpy(dtype=bool),
            (outcome == "wrong_advantaged").to_numpy(dtype=bool),
        ):
            raise RuntimeError(f"{network} wrong-advantage mismatch")

    overall = subgroup_summary(queries)
    by_source = {str(key): subgroup_summary(value) for key, value in queries.groupby("source", sort=True)}
    by_polarity = {str(key): subgroup_summary(value) for key, value in queries.groupby("polarity", sort=True)}
    by_panel = {str(key): subgroup_summary(value) for key, value in queries.groupby("panel_id", sort=True)}
    assert_nested_equal(report["strict_10ppm_same_adduct"], overall, "strict summary")
    assert_nested_equal(report["by_source"], by_source, "source summary")
    assert_nested_equal(report["by_polarity"], by_polarity, "polarity summary")
    assert_nested_equal(report["by_panel"], by_panel, "panel summary")
    expected_manifest = {
        "sources": int(queries.source.nunique()), "units": int(queries.unit_id.nunique()),
        "panels": int(queries.panel_id.nunique()), "level1_rows": int(len(queries)),
        "level1_identities": int(queries.truth_ik14.nunique()),
        "level1_formulas": int(queries.truth_formula.nunique()),
    }
    assert_nested_equal(report["manifest"], expected_manifest, "manifest summary")

    expected_overlap = recompute_overlap(ambiguous)
    expected_loso = recompute_loso(ambiguous, queries)
    pd.testing.assert_frame_equal(
        sorted_frame(overlap, ["source_a", "source_b"]),
        sorted_frame(expected_overlap, ["source_a", "source_b"]), check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        sorted_frame(loso, ["held_source"]), sorted_frame(expected_loso, ["held_source"]),
        check_dtype=False,
    )
    assert_nested_equal(report["source_overlap"], expected_overlap.to_dict(orient="records"), "source_overlap report")
    assert_nested_equal(report["leave_one_biological_source_out"], expected_loso.to_dict(orient="records"), "LOSO report")

    supported_adducts = set(library["supported_adducts"])
    truth_adduct_counts = {str(k): int(v) for k, v in queries.adduct.value_counts().sort_index().items()}
    covered_adduct_counts = {
        str(k): int(v) for k, v in queries.loc[queries.truth_present, "adduct"].value_counts().sort_index().items()
    }
    supported_panels = sorted(key for key, value in by_panel.items() if value["truth_covered_rows"] > 0)
    supported_polarities = sorted(key for key, value in by_polarity.items() if value["truth_covered_rows"] > 0)
    unsupported_adducts = sorted(set(queries.adduct.astype(str)) - supported_adducts)
    evaluability = report["candidate_protocol_evaluability"]
    for key, expected in (
        ("supported_panels", supported_panels),
        ("unsupported_panels", sorted(EXPECTED_PANELS - set(supported_panels))),
        ("supported_polarities", supported_polarities),
        ("truth_rows_by_adduct", truth_adduct_counts),
        ("truth_covered_rows_by_adduct", covered_adduct_counts),
        ("unsupported_truth_adducts", unsupported_adducts),
        ("unsupported_truth_rows", int(queries.adduct.astype(str).isin(unsupported_adducts).sum())),
    ):
        assert_nested_equal(evaluability.get(key), expected, f"evaluability.{key}")

    per_source_supported = {
        source: int(ambiguous.loc[
            (ambiguous.source == source) & ambiguous.union_any_candidate_supported, "truth_ik14"
        ].nunique()) for source in EXPECTED_SOURCES
    }
    assert_nested_equal(report["per_source_union_supported_identities"], per_source_supported, "per-source network support")
    gates = {
        "exactly_four_biological_sources": queries.source.nunique() == 4,
        "exactly_sixteen_panels": queries.panel_id.nunique() == 16,
        "candidate_library_supports_both_polarities": (
            any(value.endswith("+") for value in supported_adducts)
            and any(value.endswith("-") for value in supported_adducts)
        ),
        "all_truth_adducts_supported": len(unsupported_adducts) == 0,
        "every_panel_has_truth_coverage": all(value["truth_covered_rows"] > 0 for value in by_panel.values()),
        "ambiguous_identities_ge_600": overall["ambiguous_identities"] >= 600,
        "ambiguous_formulas_ge_450": overall["ambiguous_formulas"] >= 450,
        "each_source_ambiguous_identities_ge_100": all(
            by_source[source]["ambiguous_identities"] >= 100 for source in EXPECTED_SOURCES
        ),
        "each_source_union_supported_identities_ge_100": all(
            per_source_supported[source] >= 100 for source in EXPECTED_SOURCES
        ),
        "each_loso_train_identities_ge_400_after_purge": bool(
            (expected_loso.train_identities_after_purge >= 400).all()
        ),
        "each_loso_train_formulas_ge_300_after_purge": bool(
            (expected_loso.train_formulas_after_purge >= 300).all()
        ),
    }
    observed_gates = report.get("gates")
    if not isinstance(observed_gates, dict) or set(observed_gates) != set(gates):
        raise RuntimeError("gate registry changed")
    for key, expected in gates.items():
        if require_real_bool(observed_gates[key], f"gates.{key}") != bool(expected):
            raise RuntimeError(f"gate was not independently reproduced: {key}")
    decision = require_real_bool(
        report.get("pass_to_reaction_specific_action_discovery"),
        "pass_to_reaction_specific_action_discovery",
    )
    if decision != all(gates.values()):
        raise RuntimeError("overall scientific decision disagrees with recomputed gates")

    print(
        "[validate_bioaware_full16_action_support_m0] PASS "
        f"ambiguous={overall['ambiguous_identities']} IDs/{overall['ambiguous_formulas']} formulas "
        f"union-supported={overall['network']['union']['identities_with_strict_seed_support']} IDs "
        f"scientific_pass={decision}"
    )


if __name__ == "__main__":
    main()
