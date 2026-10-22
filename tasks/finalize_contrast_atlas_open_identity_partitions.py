#!/usr/bin/env python3
"""Combine source-isolated common-library partitions without opening effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_STATUS = "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITION_FROZEN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--partitions-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_path = args.collection / "source_manifest.tsv"
    entity_path = args.collection / "response_entities.tsv.gz"
    sources = pd.read_csv(source_path, sep="\t", usecols=["study_id"], dtype=str)
    required = pd.read_csv(
        entity_path,
        sep="\t",
        usecols=["study_id", "entity_id", "identity_status"],
        dtype=str,
    ).drop_duplicates()
    if required.duplicated(["study_id", "entity_id"]).any():
        raise RuntimeError("source-native identity status changes across contrasts")
    required = required.rename(columns={"identity_status": "source_native_identity_status"})
    tables: list[pd.DataFrame] = []
    registry: list[dict[str, object]] = []
    invariant_parameters: dict[str, object] | None = None
    library_manifest_sha = ""
    library_mgf_sha = ""
    for study in sources["study_id"].astype(str):
        root = args.partitions_root / study
        report_path = root / "report.json"
        table_path = root / "identity_partition.tsv.gz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != SOURCE_STATUS or report.get("study_id") != study:
            raise RuntimeError(f"inadmissible common identity partition: {study}")
        parameters = report.get("parameters")
        if invariant_parameters is None:
            invariant_parameters = parameters
            library_manifest_sha = str(report["provenance"]["library_manifest_sha256"])
            library_mgf_sha = str(report["provenance"]["library_mgf_sha256"])
        elif (
            parameters != invariant_parameters
            or str(report["provenance"]["library_manifest_sha256"]) != library_manifest_sha
            or str(report["provenance"]["library_mgf_sha256"]) != library_mgf_sha
        ):
            raise RuntimeError(f"common identity policy drift: {study}")
        if sha256(table_path) != report["provenance"]["identity_partition_sha256"]:
            raise RuntimeError(f"identity partition checksum drift: {study}")
        table = pd.read_csv(
            table_path,
            sep="\t",
            usecols=["study_id", "entity_id", "identity_status", "common_identity_policy"],
            dtype=str,
        )
        if not table["study_id"].eq(study).all() or table["entity_id"].duplicated().any():
            raise RuntimeError(f"identity partition key drift: {study}")
        tables.append(table)
        registry.append({
            "study_id": study,
            "report_sha256": sha256(report_path),
            "partition_sha256": sha256(table_path),
            "response_entities": len(table),
        })
    combined = pd.concat(tables, ignore_index=True)
    if combined.duplicated(["study_id", "entity_id"]).any():
        raise RuntimeError("combined partition contains duplicate entity keys")
    merged = required.merge(
        combined, on=["study_id", "entity_id"], how="left", validate="one_to_one",
    )
    if merged["identity_status"].isna().any() or len(merged) != len(required):
        raise RuntimeError("combined partition does not cover the frozen response registry")
    if not set(merged["identity_status"]).issubset({"library_supported", "unresolved_entity"}):
        raise RuntimeError("combined partition contains invalid identity status")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "identity_partition.tsv.gz"
    merged[["study_id", "entity_id", "identity_status", "common_identity_policy"]].to_csv(
        output, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    counts = merged["identity_status"].value_counts().astype(int).to_dict()
    merged["identity_status_discordant"] = (
        merged["source_native_identity_status"] != merged["identity_status"]
    )
    comparison = (
        merged.groupby("study_id", sort=False)
        .agg(
            response_entities=("entity_id", "size"),
            source_native_library_supported=(
                "source_native_identity_status", lambda values: int((values == "library_supported").sum())
            ),
            common_library_supported=(
                "identity_status", lambda values: int((values == "library_supported").sum())
            ),
            identity_status_discordant=("identity_status_discordant", "sum"),
        )
        .reset_index()
    )
    comparison_path = args.output_dir / "source_native_vs_common_identity.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)
    report = {
        "schema": "contrast_atlas_open_common_identity_partitions_v1",
        "status": "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITIONS_COMPLETE",
        "denominators": {
            "independent_studies": int(merged["study_id"].nunique()),
            "response_entities": len(merged),
            "identity_status": counts,
            "source_native_library_supported": int(
                merged["source_native_identity_status"].eq("library_supported").sum()
            ),
            "common_library_supported": int(
                merged["identity_status"].eq("library_supported").sum()
            ),
            "identity_status_discordant": int(merged["identity_status_discordant"].sum()),
        },
        "parameters": invariant_parameters,
        "leakage_status": "mechanical phenotype-blind assembly; no effect or retrieval truth read",
        "provenance": {
            "source_manifest_sha256": sha256(source_path),
            "response_entities_sha256": sha256(entity_path),
            "library_manifest_sha256": library_manifest_sha,
            "library_mgf_sha256": library_mgf_sha,
            "combined_partition_sha256": sha256(output),
            "source_native_vs_common_identity_sha256": sha256(comparison_path),
            "sources": registry,
        },
        "claim_boundary": (
            "A common library partition permits a policy-consistent known control. "
            "It does not by itself create measurement-matched known controls or prove unknown increment."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
