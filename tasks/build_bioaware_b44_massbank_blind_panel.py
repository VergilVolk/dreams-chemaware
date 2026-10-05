#!/usr/bin/env python
"""Seal a MassBank same-formula retrieval panel before any BioAware scoring.

The builder reads only record metadata and spectrum availability.  It never
loads a DreaMS checkpoint, embedding, BioAware model, or outcome score.  Query
truth identities and formulas are disjoint from the complete opened B42
candidate universe.  One spectrum per truth identity is held out globally as
query; every candidate molecule has at least one remaining reference spectrum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402
from evaluate_bioaware_b39_m2_fixed_action import atomic_csv_gzip  # noqa: E402


META_COLUMNS = [
    "InChIKey", "InChI", "MS_TYPE", "MS_ION_MODE", "PRECURSOR_MZ",
    "PRECURSOR_TYPE_ADDUCT",
]


def stable_number(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_spectrum_index(hdf5_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[int] = []
    invalid_reasons: dict[str, int] = {}
    with h5py.File(hdf5_path, "r") as handle:
        if set(handle) != {"data"}:
            raise RuntimeError(f"unexpected MassBank HDF5 datasets: {sorted(handle)}")
        dataset = handle["data"]
        for hdf5_row in range(len(dataset)):
            payload = json.loads(dataset[hdf5_row])
            if len(payload) != 7:
                invalid_rows.append(int(hdf5_row))
                invalid_reasons["payload_length"] = invalid_reasons.get("payload_length", 0) + 1
                continue
            metadata_row, ik14, mz, neutral_loss, intensity, smiles, precursor_mz = payload
            if not (len(mz) == len(intensity) and len(mz) > 0):
                invalid_rows.append(int(hdf5_row))
                reason = "empty_peaks" if len(mz) == 0 and len(intensity) == 0 else "peak_intensity_length_mismatch"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue
            rows.append({
                "hdf5_row": int(hdf5_row),
                "metadata_row": int(metadata_row),
                "ik14_payload": str(ik14).strip(),
                "smiles_payload": str(smiles).strip(),
                "precursor_mz_payload": float(precursor_mz),
                "peak_count": int(len(mz)),
                "neutral_loss_count": int(len(neutral_loss)),
            })
    audit = {
        "payload_rows": int(len(rows) + len(invalid_rows)),
        "valid_peak_rows": int(len(rows)),
        "invalid_peak_rows": int(len(invalid_rows)),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_row_sha256": hashlib.sha256(
            ",".join(map(str, invalid_rows)).encode("utf-8")
        ).hexdigest(),
        "invalid_row_preview": invalid_rows[:20],
    }
    if not rows:
        raise RuntimeError("MassBank HDF5 contains no valid peak spectra")
    return pd.DataFrame(rows), audit


def load_mgf_headers(mgf_path: Path, spectrum_index: pd.DataFrame) -> pd.DataFrame:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    with mgf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                if current is not None:
                    raise RuntimeError("nested BEGIN IONS in MassBank MGF")
                current = {}
            elif line == "END IONS":
                if current is None:
                    raise RuntimeError("END IONS without BEGIN IONS in MassBank MGF")
                entries.append(current)
                current = None
            elif current is not None and "=" in line and not line[:1].isdigit():
                key, value = line.split("=", 1)
                current[key.strip().upper()] = value.strip()
    if current is not None:
        raise RuntimeError("unterminated MassBank MGF record")
    if len(entries) != int(spectrum_index["hdf5_row"].max()) + 1:
        raise RuntimeError(
            f"MassBank MGF/HDF5 row-count mismatch: {len(entries)} versus "
            f"{int(spectrum_index['hdf5_row'].max()) + 1}"
        )
    rows: list[dict[str, Any]] = []
    alignment_mismatches = 0
    precursor_mismatches = 0
    by_hdf5_row = spectrum_index.set_index("hdf5_row")
    for hdf5_row in spectrum_index["hdf5_row"].astype(int):
        entry = entries[hdf5_row]
        full_key = str(entry.get("INCHIKEY", "")).strip()
        precursor = pd.to_numeric(entry.get("PEPMASS", ""), errors="coerce")
        payload = by_hdf5_row.loc[hdf5_row]
        if full_key[:14] != str(payload["ik14_payload"]):
            alignment_mismatches += 1
        if not np.isfinite(precursor) or abs(float(precursor) - float(payload["precursor_mz_payload"])) > 1e-6:
            precursor_mismatches += 1
        rows.append({
            "hdf5_row": int(hdf5_row),
            "full_inchikey": full_key,
            "MS_ION_MODE": str(entry.get("IONMODE", "")).strip().upper(),
            "MS_TYPE": str(entry.get("MSLEVEL", "")).strip().upper(),
            "PRECURSOR_MZ": float(precursor) if np.isfinite(precursor) else np.nan,
            "record_id": f"MassBankHDF5:{hdf5_row}",
        })
    if alignment_mismatches or precursor_mismatches:
        raise RuntimeError(
            "MassBank MGF is not the exact row-aligned source of the HDF5: "
            f"identity={alignment_mismatches}, precursor={precursor_mismatches}"
        )
    return pd.DataFrame(rows)


def canonical_metadata(
    csv_path: Path,
    mgf_path: Path,
    spectrum_index: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mgf = load_mgf_headers(mgf_path, spectrum_index)
    metadata = pd.read_csv(
        csv_path, usecols=META_COLUMNS, low_memory=False,
        encoding="utf-8", encoding_errors="replace",
    )
    metadata["full_inchikey"] = metadata["InChIKey"].fillna("").astype(str).str.strip()
    metadata["MS_ION_MODE"] = metadata["MS_ION_MODE"].fillna("").astype(str).str.strip().str.upper()
    metadata["PRECURSOR_MZ"] = pd.to_numeric(metadata["PRECURSOR_MZ"], errors="coerce")
    metadata["precursor_4dp"] = metadata["PRECURSOR_MZ"].round(4)
    metadata["PRECURSOR_TYPE_ADDUCT"] = (
        metadata["PRECURSOR_TYPE_ADDUCT"].fillna("").astype(str).str.strip()
    )
    metadata["formula"] = (
        metadata["InChI"].fillna("").astype(str)
        .str.extract(r"^InChI=\d*S?/([^/]+)", expand=False).fillna("")
    )
    metadata = metadata.loc[
        metadata["MS_TYPE"].fillna("").astype(str).str.upper().eq("MS2")
        & metadata["full_inchikey"].ne("")
        & metadata["formula"].ne("")
    ].copy()
    formula_sets = metadata.groupby("full_inchikey")["formula"].agg(lambda x: sorted(set(x)))
    formula_lookup = {key: values[0] for key, values in formula_sets.items() if len(values) == 1}
    adduct_sets = metadata.groupby(
        ["full_inchikey", "MS_ION_MODE", "precursor_4dp"], dropna=False
    )["PRECURSOR_TYPE_ADDUCT"].agg(lambda x: sorted({v for v in x if v and v.lower() != "nan"}))
    adduct_lookup = {key: values[0] for key, values in adduct_sets.items() if len(values) == 1}

    result = pd.concat(
        [spectrum_index.reset_index(drop=True), mgf.drop(columns="hdf5_row").reset_index(drop=True)],
        axis=1,
    )
    result["ik14"] = result["full_inchikey"].str.slice(0, 14)
    result["formula"] = result["full_inchikey"].map(formula_lookup).fillna("")
    lookup_keys = list(zip(
        result["full_inchikey"], result["MS_ION_MODE"], result["PRECURSOR_MZ"].round(4)
    ))
    result["PRECURSOR_TYPE_ADDUCT"] = [adduct_lookup.get(key, "") for key in lookup_keys]
    valid = (
        result["ik14"].str.len().eq(14)
        & result["ik14"].eq(result["ik14_payload"])
        & result["formula"].ne("")
        & result["MS_TYPE"].eq("2")
        & result["MS_ION_MODE"].str.upper().isin({"POSITIVE", "NEGATIVE"})
        & result["PRECURSOR_TYPE_ADDUCT"].ne("")
        & np.isfinite(result["PRECURSOR_MZ"])
        & result["PRECURSOR_MZ"].gt(0)
    )
    output = result.loc[valid].copy()
    if output["record_id"].duplicated().any() or output["hdf5_row"].duplicated().any():
        raise RuntimeError("MassBank record identity is not unique")
    audit = {
        "row_aligned_mgf_records": int(len(result)),
        "records_with_unique_formula_mapping": int(result["formula"].ne("").sum()),
        "records_with_unique_adduct_mapping": int(result["PRECURSOR_TYPE_ADDUCT"].ne("").sum()),
        "eligible_records": int(len(output)),
        "excluded_records": int(len(result) - len(output)),
        "formula_lookup_keys": int(len(formula_lookup)),
        "unique_adduct_lookup_keys": int(len(adduct_lookup)),
    }
    return output, audit


def build_panel(
    records: pd.DataFrame,
    excluded_identities: set[str],
    excluded_formulas: set[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records = records.copy()
    records["stratum"] = (
        records["formula"].astype(str) + "|" +
        records["MS_ION_MODE"].astype(str).str.upper() + "|" +
        records["PRECURSOR_TYPE_ADDUCT"].astype(str)
    )
    identity_stratum_counts = records.groupby(["ik14", "stratum"])["hdf5_row"].nunique()
    eligible_identity_strata = {
        (str(identity), str(stratum))
        for (identity, stratum), count in identity_stratum_counts.items()
        if count >= 2 and str(identity) not in excluded_identities
    }
    identities_per_stratum = records.groupby("stratum")["ik14"].nunique()
    competitive_strata = {
        str(stratum) for stratum, count in identities_per_stratum.items() if count >= 2
    }
    eligible_pair = pd.Series(
        [
            (str(identity), str(stratum)) in eligible_identity_strata
            for identity, stratum in zip(records["ik14"], records["stratum"])
        ],
        index=records.index,
    )
    query_candidates = records.loc[
        eligible_pair
        & records["stratum"].isin(competitive_strata)
        & ~records["formula"].isin(excluded_formulas)
    ].copy()
    query_candidates["selection_key"] = [
        stable_number(seed, identity, record_id)
        for identity, record_id in zip(query_candidates["ik14"], query_candidates["record_id"])
    ]
    selected_queries = (
        query_candidates.sort_values(["ik14", "selection_key", "record_id"], kind="stable")
        .groupby("ik14", sort=True, as_index=False).first()
    )
    held_query_rows = set(selected_queries["hdf5_row"].astype(int))
    library = records.loc[~records["hdf5_row"].isin(held_query_rows)].copy()
    library_identity_by_stratum = (
        library.groupby("stratum")["ik14"].agg(lambda x: sorted(set(x.astype(str)))).to_dict()
    )
    library_rows_by_stratum_identity = {
        (stratum, identity): sorted(group["hdf5_row"].astype(int).tolist())
        for (stratum, identity), group in library.groupby(["stratum", "ik14"], sort=True)
    }

    query_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in selected_queries.itertuples(index=False):
        identities = library_identity_by_stratum.get(str(row.stratum), [])
        if str(row.ik14) not in identities or len(identities) < 2:
            continue
        query_id = str(row.record_id)
        query_rows.append({
            "query_id": query_id,
            "query_hdf5_row": int(row.hdf5_row),
            "query_accession": str(row.record_id),
            "truth_ik14": str(row.ik14),
            "truth_formula": str(row.formula),
            "ion_mode": str(row.MS_ION_MODE).upper(),
            "adduct": str(row.PRECURSOR_TYPE_ADDUCT),
            "precursor_mz": float(row.PRECURSOR_MZ),
            "candidate_identities": int(len(identities)),
        })
        for identity in identities:
            references = library_rows_by_stratum_identity[(str(row.stratum), identity)]
            for reference_row in references:
                candidate_rows.append({
                    "query_id": query_id,
                    "candidate_ik14": str(identity),
                    "reference_hdf5_row": int(reference_row),
                    "is_positive": bool(str(identity) == str(row.ik14)),
                })
    queries = pd.DataFrame(query_rows)
    candidates = pd.DataFrame(candidate_rows)
    if queries.empty:
        raise RuntimeError("MassBank external panel has no evaluable query")
    positives = candidates.groupby("query_id")["is_positive"].sum()
    if not positives.ge(1).all() or len(positives) != len(queries):
        raise RuntimeError("MassBank query lacks positive reference spectra")
    identity_candidates = candidates.groupby("query_id")["candidate_ik14"].nunique()
    if not identity_candidates.ge(2).all():
        raise RuntimeError("MassBank query lacks an isomeric identity competitor")
    if set(queries["truth_ik14"]) & excluded_identities:
        raise RuntimeError("B44 query identity overlaps opened B42 universe")
    if set(queries["truth_formula"]) & excluded_formulas:
        raise RuntimeError("B44 query formula overlaps opened B42 universe")
    if set(queries["query_hdf5_row"]) & set(candidates["reference_hdf5_row"]):
        raise RuntimeError("B44 held query row appears in reference library")
    library_manifest = library.loc[
        library["hdf5_row"].isin(candidates["reference_hdf5_row"].unique()),
        [
            "hdf5_row", "metadata_row", "record_id", "full_inchikey", "ik14", "formula",
            "MS_ION_MODE", "PRECURSOR_TYPE_ADDUCT", "PRECURSOR_MZ", "peak_count",
        ],
    ].copy()
    report = {
        "input_valid_spectra": int(len(records)),
        "eligible_identity_strata": int(len(eligible_identity_strata)),
        "query_candidates_before_candidate_graph": int(len(selected_queries)),
        "queries": int(len(queries)),
        "truth_identities": int(queries["truth_ik14"].nunique()),
        "truth_formulas": int(queries["truth_formula"].nunique()),
        "candidate_reference_rows": int(len(library_manifest)),
        "candidate_identities": int(library_manifest["ik14"].nunique()),
        "candidate_formulas": int(library_manifest["formula"].nunique()),
        "candidate_rows": int(len(candidates)),
        "candidate_identities_per_query": {
            "minimum": int(identity_candidates.min()),
            "median": float(identity_candidates.median()),
            "p90": float(identity_candidates.quantile(0.9)),
            "maximum": int(identity_candidates.max()),
        },
        "by_ion_mode": queries["ion_mode"].value_counts().sort_index().astype(int).to_dict(),
    }
    return queries, candidates, library_manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massbank-hdf5", type=Path, default=ROOT / "data/massbank/massbank_full.hdf5")
    parser.add_argument("--massbank-mgf", type=Path, default=ROOT / "data/massbank/massbank_full.mgf")
    parser.add_argument("--massbank-metadata", type=Path, default=ROOT / "data/massbank/massbank_202406_msms.csv")
    parser.add_argument("--b42-dir", type=Path, default=ROOT / "data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    b42_features = args.b42_dir / "candidate_catalog_features.csv.gz"
    b42_report = args.b42_dir / "report.json"
    for path in (args.massbank_hdf5, args.massbank_mgf, args.massbank_metadata, b42_features, b42_report):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    opened = pd.read_csv(
        b42_features, usecols=["candidate_id", "truth_formula"], low_memory=False
    )
    excluded_identities = set(opened["candidate_id"].dropna().astype(str))
    excluded_formulas = set(opened["truth_formula"].dropna().astype(str))
    spectrum_index, spectrum_audit = load_spectrum_index(args.massbank_hdf5)
    records, metadata_audit = canonical_metadata(
        args.massbank_metadata, args.massbank_mgf, spectrum_index
    )
    queries, candidates, library, panel_report = build_panel(
        records, excluded_identities, excluded_formulas, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.output_dir / "queries.csv.gz"
    candidate_path = args.output_dir / "candidate_references.csv.gz"
    library_path = args.output_dir / "reference_library.csv.gz"
    atomic_csv_gzip(query_path, queries)
    atomic_csv_gzip(candidate_path, candidates)
    atomic_csv_gzip(library_path, library)
    report = {
        "status": "bioaware_b44_massbank_panel_sealed",
        "formal": True,
        "outcomes_computed": False,
        "embedding_values_read": False,
        "model_fitted": False,
        "protocol": "one held spectrum per identity; globally disjoint reference rows; exact formula+ion mode+adduct candidate identities; B42 identity/formula excluded",
        "panel": panel_report,
        "spectrum_payload_audit": spectrum_audit,
        "metadata_mapping_audit": metadata_audit,
        "overlap": {
            "b42_query_identity": 0,
            "b42_candidate_identity": 0,
            "b42_truth_formula": 0,
        },
        "gates": {
            "queries_ge_500": bool(panel_report["queries"] >= 500),
            "truth_identities_ge_500": bool(panel_report["truth_identities"] >= 500),
            "truth_formulas_ge_300": bool(panel_report["truth_formulas"] >= 300),
            "every_query_has_isomer_competitor": bool(panel_report["candidate_identities_per_query"]["minimum"] >= 2),
            "identity_and_formula_disjoint_from_b42": True,
        },
        "pass_to_one_time_evaluation": False,
        "provenance": {
            "massbank_hdf5": sha256(args.massbank_hdf5),
            "massbank_mgf": sha256(args.massbank_mgf),
            "massbank_metadata": sha256(args.massbank_metadata),
            "b42_report": sha256(b42_report),
            "b42_candidate_features": sha256(b42_features),
            "queries": sha256(query_path),
            "candidate_references": sha256(candidate_path),
            "reference_library": sha256(library_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Sealed metadata-only external panel. No DreaMS or BioAware performance has been observed.",
    }
    report["pass_to_one_time_evaluation"] = bool(all(report["gates"].values()))
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_one_time_evaluation"]:
        raise RuntimeError(f"B44 panel gate failed: {report['gates']}")


if __name__ == "__main__":
    main()
