#!/usr/bin/env python3
"""Build a balanced orthogonal-validation packet for robust FullMSG actions.

Selection is deterministic and evaluation-only.  It includes favorable,
unfavorable and context-dependent response effects across library-evidence
strata.  Spectra and response effects are exported for manual or experimental
adjudication; no selected pair is promoted to a confirmed identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "MSV000095331_FULLMSG_ORTHOGONAL_VALIDATION_PACKET_COMPLETE"
LIBRARY_SOURCES = {
    "MSV000095331": Path(
        "data/reference/msv000095331_drug_microbiome_20261023/library_matches.tsv"
    ),
    "MSV000094899": Path(
        "data/reference/chemprop2_m1_20261011/library_matches.tsv"
    ),
    "MSV000093571": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_45drug/"
        "DB_result/669306e4edf8414c8999391c98463abc.tsv"
    ),
}
FEATURE_METADATA_SOURCES = {
    "MSV000095331": {
        "path": Path(
            "data/reference/msv000095331_drug_microbiome_20261023/quantification.csv"
        ),
        "sep": ",", "id": "row ID", "mz": "row m/z", "rt": "row retention time",
    },
    "MSV000094899": {
        "path": Path("data/reference/chemprop2_m1_20261011/quantification.csv"),
        "sep": ",", "id": "row ID", "mz": "row m/z", "rt": "row retention time",
    },
    "MSV000093571": {
        "path": Path(
            "data/reference/chemprop_cross_design_20261011/chemprop_45drug/"
            "clusterinfo_summary/2621b62320ea46e4a8d9c6f48f8267dc.tsv"
        ),
        "sep": "\t", "id": "cluster index", "mz": "parent mass", "rt": "RTMean",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_entity_id(entity_id: str) -> tuple[str, str]:
    study, raw = str(entity_id).split("::", maxsplit=1)
    return study, raw


def load_mgf_entities(path: Path, required_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    block: list[str] = []
    in_block = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.strip() == "BEGIN IONS":
                block = ["BEGIN IONS"]
                in_block = True
                continue
            if not in_block:
                continue
            block.append(line)
            if line.strip() != "END IONS":
                continue
            fields: dict[str, str] = {}
            peaks: list[tuple[float, float]] = []
            for item in block[1:-1]:
                if "=" in item:
                    key, value = item.split("=", maxsplit=1)
                    fields[key.strip().upper()] = value.strip()
                    continue
                pieces = item.split()
                if len(pieces) >= 2:
                    try:
                        peaks.append((float(pieces[0]), float(pieces[1])))
                    except ValueError:
                        pass
            scan = fields.get("SCANS", fields.get("FEATURE_ID", ""))
            if scan in required_ids:
                if scan in found:
                    raise RuntimeError(f"duplicate SCANS={scan} in {path}")
                found[scan] = {"fields": fields, "peaks": peaks}
            block = []
            in_block = False
    missing = required_ids - set(found)
    if missing:
        raise RuntimeError(f"missing spectra in {path}: {sorted(missing)[:10]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=Path(
            "data/validation/msv000095331_metric_robust_entity_followup_v2_20261025"
        ),
    )
    parser.add_argument(
        "--identity-registry",
        type=Path,
        default=Path(
            "data/validation/msv000095331_complete_identity_handoff_v3_20261023/"
            "study_entity_registry.tsv.gz"
        ),
    )
    parser.add_argument(
        "--common-identity-root",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_v9_common_identity_shards_20261023"
        ),
    )
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.per_stratum < 1:
        raise ValueError("--per-stratum must be positive")

    pair_path = args.robustness_dir / "unique_identity_pair_followup_registry.tsv"
    edge_path = args.robustness_dir / "changed_edge_metric_robustness.tsv.gz"
    pairs = pd.read_csv(pair_path, sep="\t", dtype=str)
    pairs["maximum_absolute_response_contribution"] = pd.to_numeric(
        pairs["maximum_absolute_response_contribution"], errors="raise"
    )
    pairs["fullmsg_metric_action_support_count"] = pd.to_numeric(
        pairs["fullmsg_metric_action_support_count"], errors="raise"
    ).astype(int)
    robust = pairs.loc[pairs["fullmsg_metric_action_support_count"].eq(4)].copy()
    robust["validation_stratum"] = (
        robust["identity_evidence_class"] + "__" + robust["response_contribution_class"]
    )
    selected = robust.sort_values(
        "maximum_absolute_response_contribution", ascending=False, kind="mergesort"
    ).groupby("validation_stratum", sort=True).head(args.per_stratum).copy()
    selected = selected.sort_values(
        ["validation_stratum", "maximum_absolute_response_contribution"],
        ascending=[True, False], kind="mergesort",
    )

    registry = pd.read_csv(args.identity_registry, sep="\t", dtype=str)
    registry = registry.set_index(["study_id", "raw_entity_id"], drop=False)
    entity_roles: dict[tuple[str, str], set[str]] = {}
    for row in selected.itertuples():
        for role, entity in (("query", row.query_id), ("candidate", row.candidate_id)):
            key = split_entity_id(entity)
            entity_roles.setdefault(key, set()).add(role)

    mgf_requirements: dict[Path, set[str]] = {}
    for key in entity_roles:
        if key not in registry.index:
            raise RuntimeError(f"selected entity absent from registry: {key}")
        mgf_path = Path(str(registry.loc[key, "mgf_path"]))
        mgf_requirements.setdefault(mgf_path, set()).add(key[1])

    spectra: dict[tuple[str, str], dict] = {}
    mgf_hashes: dict[str, str] = {}
    for mgf_path, raw_ids in mgf_requirements.items():
        loaded = load_mgf_entities(mgf_path, raw_ids)
        mgf_hashes[str(mgf_path)] = sha256(mgf_path)
        studies = {
            study for study, raw in entity_roles
            if raw in raw_ids and Path(str(registry.loc[(study, raw), "mgf_path"])) == mgf_path
        }
        for study in studies:
            for raw in raw_ids:
                key = (study, raw)
                if key in entity_roles:
                    spectra[key] = loaded[raw]

    feature_rows: list[dict] = []
    feature_source_hashes: dict[str, str] = {}
    for study in sorted({study for study, _ in entity_roles}):
        spec = FEATURE_METADATA_SOURCES[study]
        source = Path(spec["path"])
        feature_source_hashes[study] = sha256(source)
        needed = {raw for current_study, raw in entity_roles if current_study == study}
        usecols = [spec["id"], spec["mz"], spec["rt"]]
        table = pd.read_csv(
            source, sep=spec["sep"], usecols=usecols, dtype=str, low_memory=False
        )
        table = table.loc[table[spec["id"]].isin(needed)].copy()
        if set(table[spec["id"]]) != needed:
            missing = needed - set(table[spec["id"]])
            raise RuntimeError(f"missing feature metadata for {study}: {sorted(missing)}")
        for values in table.to_dict("records"):
            feature_rows.append({
                "study_id": study,
                "raw_entity_id": str(values[spec["id"]]),
                "feature_mz": str(values[spec["mz"]]),
                "feature_retention_time": str(values[spec["rt"]]),
                "feature_metadata_source": str(source),
            })
    feature_metadata = pd.DataFrame(feature_rows)

    metadata_rows: list[dict] = []
    peak_rows: list[dict] = []
    mgf_blocks: list[str] = []
    for output_scan, key in enumerate(sorted(spectra), start=1):
        study, raw = key
        record = spectra[key]
        fields = record["fields"]
        peaks = record["peaks"]
        intensities = np.asarray([intensity for _, intensity in peaks], dtype=float)
        peak_mzs = np.asarray([mz for mz, _ in peaks], dtype=float)
        base_index = int(np.argmax(intensities)) if len(intensities) else -1
        mgf_path = Path(str(registry.loc[key, "mgf_path"]))
        metadata_rows.append({
            "study_id": study,
            "raw_entity_id": raw,
            "entity_id": f"{study}::{raw}",
            "packet_roles": ";".join(sorted(entity_roles[key])),
            "identity_status": str(registry.loc[key, "identity_status"]),
            "source_mgf": str(mgf_path),
            "source_mgf_sha256": mgf_hashes[str(mgf_path)],
            "original_scans": fields.get("SCANS", ""),
            "precursor_mz": fields.get("PEPMASS", "").split()[0],
            "charge": fields.get("CHARGE", ""),
            "retention_time_seconds": fields.get("RTINSECONDS", ""),
            "peak_count": len(peaks),
            "total_fragment_intensity": float(intensities.sum()) if len(peaks) else 0.0,
            "base_peak_mz": float(peak_mzs[base_index]) if base_index >= 0 else np.nan,
            "base_peak_intensity": (
                float(intensities[base_index]) if base_index >= 0 else np.nan
            ),
        })
        for rank, (mz, intensity) in enumerate(
            sorted(peaks, key=lambda item: item[1], reverse=True), start=1
        ):
            peak_rows.append({
                "study_id": study,
                "raw_entity_id": raw,
                "fragment_intensity_rank": rank,
                "fragment_mz": mz,
                "fragment_intensity": intensity,
            })
        mgf_blocks.extend([
            "BEGIN IONS",
            f"TITLE={study}::{raw}",
            f"SCANS={output_scan}",
            f"SOURCE_STUDY={study}",
            f"SOURCE_RAW_ENTITY_ID={raw}",
            f"SOURCE_MGF={mgf_path}",
            f"SOURCE_SCANS={fields.get('SCANS', '')}",
            f"PEPMASS={fields.get('PEPMASS', '')}",
            f"CHARGE={fields.get('CHARGE', '')}",
        ])
        if fields.get("RTINSECONDS", ""):
            mgf_blocks.append(f"RTINSECONDS={fields['RTINSECONDS']}")
        mgf_blocks.extend([f"{mz:.8g}\t{intensity:.8g}" for mz, intensity in peaks])
        mgf_blocks.extend(["END IONS", ""])

    edges = pd.read_csv(edge_path, sep="\t", dtype={"candidate_pair_id": str})
    selected_effects = edges.loc[
        edges["candidate_pair_id"].isin(set(selected["candidate_pair_id"]))
    ].copy()

    library_columns = [
        "SpectrumID", "Compound_Name", "MQScore", "SharedPeaks", "MZErrorPPM",
        "Adduct", "Precursor_MZ", "molecular_formula", "InChIKey", "Smiles",
        "LibraryQualityString", "library_usi",
    ]
    library_rows: list[pd.DataFrame] = []
    for study, source in LIBRARY_SOURCES.items():
        needed = {raw for current_study, raw in entity_roles if current_study == study}
        if not needed:
            continue
        table = pd.read_csv(source, sep="\t", dtype=str, low_memory=False)
        if "#Scan#" not in table.columns:
            raise RuntimeError(f"library table lacks #Scan#: {source}")
        block = table.loc[table["#Scan#"].isin(needed)].copy()
        if block.empty:
            continue
        output = pd.DataFrame({
            "study_id": study,
            "raw_entity_id": block["#Scan#"].astype(str),
            "library_source": str(source),
        })
        for column in library_columns:
            output[column] = block[column].astype(str) if column in block.columns else ""
        library_rows.append(output)
    library_evidence = (
        pd.concat(library_rows, ignore_index=True)
        if library_rows else pd.DataFrame(
            columns=["study_id", "raw_entity_id", "library_source", *library_columns]
        )
    )

    common_rows: list[pd.DataFrame] = []
    common_paths: dict[str, Path] = {}
    for study in sorted({study for study, _ in entity_roles if study != "MSV000095331"}):
        source = args.common_identity_root / study / "identity_partition.tsv.gz"
        common_paths[study] = source
        needed = {raw for current_study, raw in entity_roles if current_study == study}
        table = pd.read_csv(source, sep="\t", dtype=str)
        block = table.loc[table["raw_entity_id"].isin(needed)].copy()
        common_rows.append(block)
    common_identity = (
        pd.concat(common_rows, ignore_index=True)
        if common_rows else pd.DataFrame(columns=["study_id", "raw_entity_id"])
    )

    native_compare = library_evidence[[
        "study_id", "raw_entity_id", "Compound_Name", "InChIKey", "MQScore",
        "SharedPeaks", "MZErrorPPM", "library_source",
    ]].copy()
    native_compare = native_compare.sort_values(
        "MQScore", ascending=False, kind="mergesort"
    ).drop_duplicates(["study_id", "raw_entity_id"])
    native_compare = native_compare.rename(columns={
        "Compound_Name": "native_library_name",
        "InChIKey": "native_inchikey",
        "MQScore": "native_score",
        "SharedPeaks": "native_shared_peaks",
        "MZErrorPPM": "native_mz_error_ppm",
        "library_source": "native_library_source",
    })
    common_columns = [
        "study_id", "raw_entity_id", "identity_status", "library_name",
        "library_inchikey", "wse", "matched_peaks", "library_quality",
        "library_spectrum_id",
    ]
    common_compare = common_identity.reindex(columns=common_columns).rename(columns={
        "identity_status": "common_identity_status",
        "library_name": "common_library_name",
        "library_inchikey": "common_inchikey",
        "wse": "common_wse",
        "matched_peaks": "common_matched_peaks",
        "library_quality": "common_library_quality",
        "library_spectrum_id": "common_library_spectrum_id",
    })
    comparison_keys = pd.DataFrame([
        {"study_id": study, "raw_entity_id": raw}
        for study, raw in sorted(entity_roles)
    ])
    identity_comparison = comparison_keys.merge(
        native_compare, on=["study_id", "raw_entity_id"], how="left",
        validate="one_to_one",
    ).merge(
        common_compare, on=["study_id", "raw_entity_id"], how="left",
        validate="one_to_one",
    )
    native_present = identity_comparison["native_library_name"].fillna("").ne("")
    common_candidate_present = identity_comparison["common_library_name"].fillna("").ne("")
    common_present = identity_comparison["common_identity_status"].fillna("").eq(
        "library_supported"
    )
    native_planar = identity_comparison["native_inchikey"].fillna("").str[:14]
    common_planar = identity_comparison["common_inchikey"].fillna("").str[:14]
    both_keys = native_planar.ne("") & common_planar.ne("")
    identity_comparison["cross_library_evidence_relation"] = np.select(
        [
            both_keys & native_planar.eq(common_planar),
            both_keys & native_planar.ne(common_planar),
            native_present & common_present,
            native_present & ~common_present,
            ~native_present & common_present,
            ~native_present & ~common_present & common_candidate_present,
        ],
        [
            "concordant_planar_inchikey",
            "conflicting_planar_inchikey",
            "both_present_key_unavailable",
            "native_only",
            "common_only",
            "common_below_threshold_candidate_only",
        ],
        default="no_library_evidence",
    )
    native_planar_for_group = identity_comparison["native_inchikey"].fillna("").str[:14]
    multiplicity = identity_comparison.loc[native_planar_for_group.ne("")].copy()
    multiplicity["native_planar_inchikey"] = native_planar_for_group.loc[
        native_planar_for_group.ne("")
    ]
    multiplicity["selected_feature_count_for_key"] = multiplicity.groupby(
        ["study_id", "native_planar_inchikey"]
    )["raw_entity_id"].transform("size")
    multiplicity = multiplicity.loc[
        multiplicity["selected_feature_count_for_key"].gt(1)
    ].merge(
        feature_metadata,
        on=["study_id", "raw_entity_id"],
        how="left",
        validate="one_to_one",
    )

    args.output_dir.mkdir(parents=True)
    outputs = {
        "selected_pairs": args.output_dir / "selected_identity_pairs.tsv",
        "response_contexts": args.output_dir / "selected_pair_response_contexts.tsv",
        "spectrum_metadata": args.output_dir / "selected_spectrum_metadata.tsv",
        "spectrum_peaks": args.output_dir / "selected_spectrum_peaks.tsv.gz",
        "spectra_mgf": args.output_dir / "selected_spectra.mgf",
        "native_library_evidence": (
            args.output_dir / "selected_source_native_library_evidence.tsv"
        ),
        "common_identity_evidence": (
            args.output_dir / "selected_common_identity_evidence.tsv"
        ),
        "identity_evidence_comparison": (
            args.output_dir / "selected_identity_evidence_comparison.tsv"
        ),
        "feature_metadata": args.output_dir / "selected_feature_metadata.tsv",
        "within_study_library_multiplicity": (
            args.output_dir / "selected_within_study_library_multiplicity.tsv"
        ),
    }
    selected.to_csv(outputs["selected_pairs"], sep="\t", index=False)
    selected_effects.to_csv(outputs["response_contexts"], sep="\t", index=False)
    spectrum_metadata = pd.DataFrame(metadata_rows).merge(
        feature_metadata,
        on=["study_id", "raw_entity_id"],
        how="left",
        validate="one_to_one",
    )
    spectrum_metadata.to_csv(outputs["spectrum_metadata"], sep="\t", index=False)
    pd.DataFrame(peak_rows).to_csv(
        outputs["spectrum_peaks"],
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    outputs["spectra_mgf"].write_bytes("\n".join(mgf_blocks).encode("utf-8"))
    library_evidence.to_csv(outputs["native_library_evidence"], sep="\t", index=False)
    common_identity.to_csv(outputs["common_identity_evidence"], sep="\t", index=False)
    identity_comparison.to_csv(
        outputs["identity_evidence_comparison"], sep="\t", index=False
    )
    feature_metadata.to_csv(outputs["feature_metadata"], sep="\t", index=False)
    multiplicity.to_csv(
        outputs["within_study_library_multiplicity"], sep="\t", index=False
    )

    report = {
        "status": STATUS,
        "dataset_role": "development_consumed_not_external_validation",
        "selection_rule": (
            "all-four-readout action support, then top absolute response contribution "
            "within identity-evidence x response-contribution strata"
        ),
        "per_stratum": args.per_stratum,
        "denominators": {
            "robust_unique_identity_pairs_eligible": int(len(robust)),
            "selected_identity_pairs": int(len(selected)),
            "selection_strata": int(selected["validation_stratum"].nunique()),
            "selected_unique_spectra": int(len(metadata_rows)),
            "selected_response_context_instances": int(len(selected_effects)),
            "library_evidence_rows": int(len(library_evidence)),
            "common_identity_evidence_rows": int(len(common_identity)),
            "common_qualified_library_support_rows": int(
                common_identity["identity_status"].eq("library_supported").sum()
            ),
            "cross_library_conflicts": int(
                identity_comparison["cross_library_evidence_relation"].eq(
                    "conflicting_planar_inchikey"
                ).sum()
            ),
            "within_study_selected_library_key_multiplicity_groups": int(
                multiplicity[["study_id", "native_planar_inchikey"]]
                .drop_duplicates().shape[0]
            ),
        },
        "stratum_counts": {
            str(key): int(value)
            for key, value in selected.groupby("validation_stratum").size().items()
        },
        "inputs": {
            "unique_pair_registry": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "changed_edge_robustness": {"path": str(edge_path), "sha256": sha256(edge_path)},
            "identity_registry": {
                "path": str(args.identity_registry), "sha256": sha256(args.identity_registry)
            },
            "source_mgf": mgf_hashes,
            "library_sources": {
                study: {"path": str(path), "sha256": sha256(path)}
                for study, path in LIBRARY_SOURCES.items()
                if any(current_study == study for current_study, _ in entity_roles)
            },
            "common_identity_sources": {
                study: {"path": str(path), "sha256": sha256(path)}
                for study, path in common_paths.items()
            },
            "feature_metadata_sources": {
                study: {
                    "path": str(FEATURE_METADATA_SOURCES[study]["path"]),
                    "sha256": digest,
                }
                for study, digest in feature_source_hashes.items()
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "Selection is post-response and may prioritize validation but cannot train a model.",
            "Four readouts share frozen embeddings and are not independent confirmations.",
            "Library support on one or both ends does not confirm the cross-study identity edge.",
            "Conflicting library candidates are preserved rather than collapsed to one name.",
            "Helpful, harmful and context-dependent response cases are all represented.",
            "Only orthogonal identity evidence may adjudicate these hypotheses.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
