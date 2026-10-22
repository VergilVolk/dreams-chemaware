#!/usr/bin/env python3
"""Build design-aware response objects from curated public GNPS FBMN tasks.

The manifest is deliberately small and explicit.  It freezes biological-unit
definitions and within-study contrasts before any cross-study retrieval score
is computed.  Public FBMN tables are author-preprocessed measurements, not a
uniform reprocessing of raw data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tasks") not in sys.path:
    sys.path.insert(0, str(ROOT / "tasks"))

from estimate_fbmn_response_object import (  # noqa: E402
    canonical_sample_column,
    collapse_units,
    estimate,
    normalize,
)


RESULT_URLS = (
    "https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile",
    "https://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile",
    "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile",
    "https://proteomics.ucsd.edu/ProteoSAFe/DownloadResultFile",
)
ARTIFACTS = {
    "quantification": "quantification_table_reformatted/",
    "metadata": "metadata_merged/",
    "mgf": "spectra_reformatted/",
    "library": "DB_result/",
}
PLANAR_INCHIKEY = re.compile(r"^[A-Z]{14}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_artifact(task_id: str, remote: str, output: Path) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    errors: list[str] = []
    for url in RESULT_URLS:
        print(f"DOWNLOAD_START {task_id} {remote} {url}", flush=True)
        try:
            with requests.get(
                url,
                params={"task": task_id, "file": remote, "block": "main"},
                stream=True,
                timeout=(20, 90),
            ) as response:
                response.raise_for_status()
                iterator = response.iter_content(1 << 20)
                first = next(iterator, b"")
                if not first or b"<html" in first[:500].lower():
                    raise RuntimeError("empty or HTML response")
                with partial.open("wb") as handle:
                    handle.write(first)
                    for block in iterator:
                        handle.write(block)
            partial.replace(output)
            print(f"DOWNLOAD_DONE {task_id} {remote} {output.stat().st_size}", flush=True)
            return
        except (requests.RequestException, OSError, RuntimeError) as error:
            errors.append(f"{url}:{type(error).__name__}")
            if partial.exists():
                partial.unlink()
    raise RuntimeError(f"all GNPS download hosts failed for {task_id}/{remote}: {errors}")


def read_delimited(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def mgf_feature_ids(path: Path) -> tuple[set[str], int, int]:
    feature_ids: set[str] = set()
    blocks = 0
    blocks_without_id = 0
    current: str | None = None
    has_positive_peak = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                current = None
                has_positive_peak = False
            elif line.startswith("FEATURE_ID="):
                current = line.split("=", 1)[1].strip()
            elif current is None and line.startswith("SCANS="):
                current = line.split("=", 1)[1].strip()
            elif line == "END IONS":
                blocks += 1
                if current and has_positive_peak:
                    feature_ids.add(current)
                elif not current:
                    blocks_without_id += 1
            elif line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        mz, intensity = float(fields[0]), float(fields[1])
                    except ValueError:
                        continue
                    has_positive_peak |= (
                        np.isfinite(mz) and np.isfinite(intensity) and mz >= 0 and intensity > 0
                    )
    return feature_ids, blocks, blocks_without_id


def select_level(metadata: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    column = str(rule["column"])
    if column not in metadata:
        raise ValueError(f"metadata lacks level column {column}")
    accepted = {str(value) for value in rule["values"]}
    return metadata[column].fillna("").astype(str).str.strip().isin(accepted).to_numpy()


def build_design(
    metadata: pd.DataFrame,
    source: dict[str, Any],
    contrast: dict[str, Any],
) -> pd.DataFrame:
    filename_column = str(source.get("filename_column", "filename"))
    if filename_column not in metadata:
        raise ValueError(f"metadata lacks filename column {filename_column}")
    eligible = np.ones(len(metadata), dtype=bool)
    for rule in contrast.get("filters", []):
        column = str(rule["column"])
        if column not in metadata:
            raise ValueError(f"metadata lacks filter column {column}")
        eligible &= metadata[column].fillna("").astype(str).str.strip().isin(
            {str(value) for value in rule["values"]}
        ).to_numpy()

    assigned = np.zeros(len(metadata), dtype=int)
    level = np.full(len(metadata), "", dtype=object)
    for output_level, rule in contrast["levels"].items():
        chosen = eligible & select_level(metadata, rule)
        assigned += chosen.astype(int)
        level[chosen] = str(output_level)
    if np.any(assigned > 1):
        raise ValueError(f"overlapping levels in {contrast['object_id']}")
    selected = eligible & (assigned == 1)
    design = pd.DataFrame({
        "filename": metadata.loc[selected, filename_column].astype(str).str.strip().to_numpy(),
        "level": level[selected],
    })
    unit_column = str(source["biological_unit_column"])
    if unit_column == "__filename__":
        design["biological_unit_id"] = design["filename"]
    else:
        if unit_column not in metadata:
            raise ValueError(f"metadata lacks biological unit column {unit_column}")
        design["biological_unit_id"] = (
            metadata.loc[selected, unit_column].astype(str).str.strip().to_numpy()
        )
    if design.filename.eq("").any() or design.biological_unit_id.eq("").any():
        raise ValueError("empty filename or biological unit ID")
    if design.filename.duplicated().any():
        raise ValueError("metadata design filenames are not unique")
    crossing = design.groupby("biological_unit_id")["level"].nunique()
    if crossing.gt(1).any():
        raise ValueError("biological units cross contrast levels")
    observed = design.drop_duplicates(["biological_unit_id", "level"]).groupby("level").size().to_dict()
    expected = {str(key): int(value) for key, value in contrast["expected_units_per_level"].items()}
    if observed != expected:
        raise ValueError(
            f"unit-count drift in {contrast['object_id']}: observed={observed}, expected={expected}"
        )
    return design[["filename", "biological_unit_id", "level"]]


def quantify_response(
    quant: pd.DataFrame,
    design: pd.DataFrame,
    contrast: dict[str, Any],
    mgf_ids: set[str],
    identity: pd.DataFrame,
    minimum_prevalence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lookup: dict[str, str] = {}
    for column in quant.columns:
        canonical = canonical_sample_column(column)
        if canonical in lookup:
            raise ValueError(f"duplicate canonical sample column: {canonical}")
        lookup[canonical] = column
    missing = sorted(set(design.filename) - set(lookup))
    if missing:
        raise ValueError(f"design files absent from quantification table: {missing[:5]}")
    sample_columns = [lookup[value] for value in design.filename]
    values = quant[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    normalized, offsets = normalize(values)
    unit_values, units = collapse_units(normalized, design)
    coefficients = {str(key): float(value) for key, value in contrast["level_coefficients"].items()}
    if not np.isclose(sum(coefficients.values()), 0.0):
        raise ValueError("contrast coefficients must sum to zero")
    minimum_units = int(contrast.get("minimum_units_per_level", 3))
    effects = estimate(
        unit_values,
        units.level.to_numpy(str),
        coefficients,
        minimum_units,
        minimum_prevalence,
    )
    feature_id_column = "row ID" if "row ID" in quant else quant.columns[0]
    entity_ids = quant[feature_id_column].astype(str)
    effects.insert(0, "entity_id", entity_ids)
    effects["ms2_available"] = entity_ids.isin(mgf_ids)
    identity_lookup = identity.set_index("entity_id")["identity_status"]
    if not set(entity_ids).issubset(set(identity_lookup.index)):
        raise ValueError("quantified entities are absent from the identity partition")
    effects["identity_status"] = entity_ids.map(identity_lookup)
    effects["analysis_eligible"] = effects.qc_pass & effects.ms2_available
    normalization = pd.DataFrame({
        "filename": design.filename,
        "log2_median_offset": offsets,
    })
    counts = units.groupby("level").size().astype(int).to_dict()
    summary = {
        "quantified_features": int(len(effects)),
        "biological_units": int(len(units)),
        "units_per_level": counts,
        "qc_eligible_features": int(effects.qc_pass.sum()),
        "qc_eligible_features_with_ms2": int(effects.analysis_eligible.sum()),
        "qc_eligible_features_by_identity": {
            str(key): int(value)
            for key, value in effects.loc[effects.qc_pass, "identity_status"].value_counts().items()
        },
    }
    return effects, normalization, summary


def build_identity_partition(quant: pd.DataFrame, library_path: Path) -> pd.DataFrame:
    feature_id_column = "row ID" if "row ID" in quant else quant.columns[0]
    entity_ids = quant[feature_id_column].astype(str)
    if entity_ids.duplicated().any():
        raise ValueError("quantification feature IDs are not unique")
    library = pd.read_csv(library_path, sep="\t", dtype=str)
    required = {"#Scan#", "Compound_Name", "LibraryQualityString", "InChIKey-Planar"}
    if missing := required - set(library):
        raise ValueError(f"library result lacks {sorted(missing)}")
    if library["#Scan#"].astype(str).duplicated().any():
        raise ValueError("GNPS library result contains duplicate feature hits")
    quality = library.LibraryQualityString.fillna("").astype(str)
    planar = library["InChIKey-Planar"].fillna("").astype(str)
    direct = (
        quality.isin({"Gold", "Silver", "Bronze"})
        & planar.map(lambda value: bool(PLANAR_INCHIKEY.fullmatch(value)))
        & ~library.Compound_Name.fillna("").astype(str).str.startswith("Suspect related")
    )
    supported = library.loc[direct, [
        "#Scan#", "Compound_Name", "LibraryQualityString", "MQScore",
        "SharedPeaks", "InChIKey-Planar", "Adduct", "IonMode",
    ]].rename(columns={"#Scan#": "entity_id"})
    supported["entity_id"] = supported.entity_id.astype(str)
    partition = pd.DataFrame({"entity_id": entity_ids})
    for source, target in (("row m/z", "precursor_mz"), ("row retention time", "retention_time")):
        if source in quant:
            partition[target] = quant[source]
    partition = partition.merge(supported, on="entity_id", how="left", validate="one_to_one")
    partition["identity_status"] = "unresolved_entity"
    partition.loc[
        partition.LibraryQualityString.notna(), "identity_status"
    ] = "library_supported"
    partition["identity_partition_rule"] = (
        "direct_GNPS_Gold_Silver_Bronze_with_planar_InChIKey"
    )
    return partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-prevalence", type=float, default=0.20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "public_fbmn_response_panel_manifest_v1":
        raise ValueError("unsupported manifest schema")

    source_reports: list[dict[str, Any]] = []
    contrast_reports: list[dict[str, Any]] = []
    source_unit_sets: dict[str, set[str]] = {}
    for source in manifest["sources"]:
        source_id = str(source["source_id"])
        task_id = str(source["task_id"])
        source_dir = args.source_root / task_id
        paths = {
            "quantification": source_dir / "quantification.csv",
            "metadata": source_dir / "metadata.tsv",
            "mgf": source_dir / "representative.mgf",
            "library": source_dir / "db_result.tsv",
        }
        for name, remote in ARTIFACTS.items():
            download_artifact(task_id, remote, paths[name])
        metadata = read_delimited(paths["metadata"])
        metadata = metadata.dropna(axis=1, how="all")
        quant = pd.read_csv(paths["quantification"], low_memory=False)
        mgf_ids, mgf_blocks, blocks_without_id = mgf_feature_ids(paths["mgf"])
        identity = build_identity_partition(quant, paths["library"])
        expected_rows = int(source["expected_metadata_rows"])
        if len(metadata) != expected_rows:
            raise ValueError(f"metadata-row drift for {source_id}: {len(metadata)} != {expected_rows}")
        source_output = args.output_dir / source_id
        source_output.mkdir(parents=True, exist_ok=True)
        identity_path = source_output / "identity_partition.tsv.gz"
        identity.to_csv(
            identity_path, sep="\t", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        identity_counts = identity.identity_status.value_counts().astype(int).to_dict()
        source_record = {
            "source_id": source_id,
            "task_id": task_id,
            "accession": source["accession"],
            "pmid": source["pmid"],
            "scientific_role": source["scientific_role"],
            "metadata_rows": int(len(metadata)),
            "quantified_features": int(len(quant)),
            "representative_mgf_blocks": mgf_blocks,
            "representative_mgf_nonempty_unique_feature_ids": len(mgf_ids),
            "mgf_blocks_without_feature_id": blocks_without_id,
            "identity_partition": identity_counts,
            "identity_partition_sha256": sha256(identity_path),
            "artifacts": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in paths.items()
            },
        }
        source_reports.append(source_record)
        source_unit_sets[source_id] = set()
        for contrast in source["contrasts"]:
            design = build_design(metadata, source, contrast)
            source_unit_sets[source_id].update(design.biological_unit_id.astype(str))
            effects, normalization, summary = quantify_response(
                quant, design, contrast, mgf_ids, identity, args.minimum_prevalence
            )
            contrast_dir = args.output_dir / source_id / str(contrast["object_id"])
            contrast_dir.mkdir(parents=True)
            design_path = contrast_dir / "design.tsv"
            response_path = contrast_dir / "response_object.tsv.gz"
            normalization_path = contrast_dir / "sample_normalization.tsv"
            design.to_csv(design_path, sep="\t", index=False)
            effects.to_csv(
                response_path, sep="\t", index=False,
                compression={"method": "gzip", "mtime": 0},
            )
            normalization.to_csv(normalization_path, sep="\t", index=False)
            record = {
                "source_id": source_id,
                "study_id": source["accession"],
                "pmid": source["pmid"],
                "object_id": contrast["object_id"],
                "scientific_role": source["scientific_role"],
                "estimand": contrast["estimand"],
                "causal_claim_allowed": bool(contrast["causal_claim_allowed"]),
                "level_coefficients": contrast["level_coefficients"],
                "denominators": summary,
                "provenance": {
                    "design_sha256": sha256(design_path),
                    "response_object_sha256": sha256(response_path),
                    "sample_normalization_sha256": sha256(normalization_path),
                },
            }
            (contrast_dir / "report.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            contrast_reports.append(record)
        source_record["unique_biological_units_in_any_contrast"] = len(
            source_unit_sets[source_id]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "public_fbmn_systematic_response_panel_v1",
        "status": "DEVELOPMENT_CONSUMED_RESPONSE_OBJECTS_BUILT_NO_CROSS_STUDY_RESULT",
        "denominators": {
            "independent_sources": len(source_reports),
            "response_objects": len(contrast_reports),
            "metadata_files_across_sources": int(sum(row["metadata_rows"] for row in source_reports)),
            "biological_units_summed_over_response_objects": int(
                sum(row["denominators"]["biological_units"] for row in contrast_reports)
            ),
            "unique_biological_units_within_sources": int(
                sum(len(values) for values in source_unit_sets.values())
            ),
            "eligible_response_rows": int(
                sum(row["denominators"]["qc_eligible_features"] for row in contrast_reports)
            ),
            "eligible_response_rows_with_ms2": int(
                sum(row["denominators"]["qc_eligible_features_with_ms2"] for row in contrast_reports)
            ),
        },
        "sources": source_reports,
        "contrasts": contrast_reports,
        "measurement_contract": {
            "normalization": "positive-only log2 intensity followed by within-contrast sample-median centering",
            "technical_files": "median within a manifest-frozen biological_unit_id before effect estimation",
            "missingness": "zero and nonpositive values remain missing; no zero-as-abundance and no imputation",
            "effect": "linear contrast of biological-unit means with independent-group Welch standard error",
            "identity": "strict phenotype-blind GNPS Gold/Silver/Bronze direct-hit control with planar InChIKey; every other feature is unresolved_entity",
        },
        "leakage_status": "within-study labels used only after task, biological-unit, contrast and count contracts were frozen; no cross-study retrieval score read",
        "provenance": {"manifest_sha256": sha256(args.manifest)},
        "claim_boundary": [
            "Every source and result is development/consumed, not external validation.",
            "Public FBMN feature tables reflect author preprocessing and are not uniform raw-data reprocessing.",
            "Unresolved means lacking a direct strict GNPS library-supported control hit, not a unique unknown metabolite.",
            "Response objects are research-level effect estimates, not validated cross-study biological relations.",
            "Observational context contrasts are descriptive and cannot be promoted to intervention effects.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
