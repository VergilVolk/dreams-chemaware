#!/usr/bin/env python3
"""Prepare truth-blind spectrum and identity inputs for the v10 Atlas axis.

The historical fifteen-source common identity partition is preserved exactly.
The new source has not undergone that common GNPS Gold/Silver search, so all of
its entities enter the *common* partition conservatively as unresolved.  Its
source-native library labels remain available in the source package and are
not promoted into the common known-only world.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


NEW_STUDY = "MSV000080918"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gzip_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-dir",
        default=(
            "data/validation/"
            "contrast_atlas_open_response_collection_v10_frozenbase_msv000080918_v7_20261031"
        ),
    )
    parser.add_argument(
        "--base-source-report",
        default="data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023/report.json",
    )
    parser.add_argument(
        "--base-identity-partition",
        default=(
            "data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/"
            "identity_partition.tsv.gz"
        ),
    )
    parser.add_argument(
        "--new-source-dir",
        default="data/validation/contrast_atlas_msv000080918_atlas_source_v5_20261031",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation/contrast_atlas_v10_identity_axis_inputs_v1_20261031",
    )
    parser.add_argument(
        "--spectrum-asset-root",
        default="",
        help="Optional checkout containing historical repo-relative MGF assets.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    collection = (root / args.collection_dir).resolve()
    base_source_report_path = (root / args.base_source_report).resolve()
    base_identity_path = (root / args.base_identity_partition).resolve()
    new_source = (root / args.new_source_dir).resolve()
    output = (root / args.output_dir).resolve()
    spectrum_asset_root = Path(args.spectrum_asset_root).resolve() if args.spectrum_asset_root else None
    if output.exists():
        raise FileExistsError(output)

    collection_report_path = collection / "report.json"
    collection_report = json.loads(collection_report_path.read_text(encoding="utf-8"))
    response_path = collection / "response_entities.tsv.gz"
    if sha256(response_path) != collection_report["output_sha256"]["response_entities.tsv.gz"]:
        raise ValueError("response collection hash mismatch")

    new_report_path = new_source / "report.json"
    new_report = json.loads(new_report_path.read_text(encoding="utf-8"))
    new_mgf = new_source / "representative_spectra.mgf"
    new_identity = new_source / "source_native_identity.tsv"
    if sha256(new_mgf) != new_report["outputs_sha256"]["representative_spectra.mgf"]:
        raise ValueError("new-source MGF hash mismatch")
    if sha256(new_identity) != new_report["outputs_sha256"]["source_native_identity.tsv"]:
        raise ValueError("new-source identity hash mismatch")

    response_entities: dict[str, set[str]] = {}
    response_rows = 0
    with gzip.open(response_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            response_rows += 1
            response_entities.setdefault(row["study_id"], set()).add(row["entity_id"])

    denominators = collection_report["denominators"]
    if response_rows != int(denominators["eligible_response_rows"]):
        raise ValueError("response row denominator drifted")
    if sum(map(len, response_entities.values())) != int(denominators["unique_study_entities"]):
        raise ValueError("response entity denominator drifted")
    if NEW_STUDY not in response_entities:
        raise ValueError(f"new study absent from collection: {NEW_STUDY}")

    base_source_report = json.loads(base_source_report_path.read_text(encoding="utf-8"))
    base_sources = list(base_source_report["sources"])
    base_studies = {str(row["study_id"]) for row in base_sources}
    if NEW_STUDY in base_studies:
        raise ValueError("new study already exists in the base source report")
    if base_studies | {NEW_STUDY} != set(response_entities):
        raise ValueError("combined spectrum sources differ from the response collection")
    for row in base_sources:
        relative_mgf = Path(str(row["mgf_path"]).replace("\\", "/"))
        mgf = root / relative_mgf
        if not mgf.exists() and spectrum_asset_root is not None:
            mgf = spectrum_asset_root / relative_mgf
        if sha256(mgf) != row["mgf_sha256"]:
            raise ValueError(f"base MGF hash mismatch: {row['study_id']}")

    base_partition: list[dict[str, str]] = []
    with gzip.open(base_identity_path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != [
            "study_id", "entity_id", "identity_status", "common_identity_policy"
        ]:
            raise ValueError("unexpected base identity-partition schema")
        base_partition.extend(reader)
    base_keys = {(row["study_id"], row["entity_id"]) for row in base_partition}
    expected_base_keys = {
        (study_id, entity_id)
        for study_id, entities in response_entities.items()
        if study_id != NEW_STUDY
        for entity_id in entities
    }
    if base_keys != expected_base_keys:
        raise ValueError("base common identity partition differs from frozen response entities")

    native: dict[str, str] = {}
    with new_identity.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            global_id = f"{NEW_STUDY}::{row['entity_id']}"
            if global_id in native:
                raise ValueError(f"duplicate new source identity: {global_id}")
            native[global_id] = row["identity_status"]
    if set(native) != response_entities[NEW_STUDY]:
        raise ValueError("new source native identities differ from response entities")

    combined_partition = [dict(row) for row in base_partition]
    combined_partition.extend(
        {
            "study_id": NEW_STUDY,
            "entity_id": entity_id,
            "identity_status": "unresolved_entity",
            "common_identity_policy": "pending_common_search_conservative_unresolved_v1",
        }
        for entity_id in sorted(native)
    )

    output.mkdir(parents=True)
    identity_output = output / "identity_partition.tsv.gz"
    write_gzip_tsv(
        identity_output,
        combined_partition,
        ["study_id", "entity_id", "identity_status", "common_identity_policy"],
    )

    source_manifest = {
        "schema": "contrast_atlas_spectrum_source_manifest_v1",
        "status": "DEVELOPMENT_CONSUMED_SPECTRUM_SOURCE_MANIFEST_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "response_rows": response_rows,
            "study_entities": sum(map(len, response_entities.values())),
            "studies": len(response_entities),
        },
        "sources": [
            *base_sources,
            {
                "study_id": NEW_STUDY,
                "mgf_path": str(new_mgf.relative_to(root)).replace("/", "\\"),
                "mgf_sha256": sha256(new_mgf),
                "mgf_spectra_total": len(native),
                "response_spectra_nonempty": len(native),
            },
        ],
        "identity_policy": {
            "base_sources": "preserved common GNPS Gold/Silver partition",
            "new_source": (
                "all entities remain unresolved in the common partition until the same common "
                "search protocol is run; source-native labels are retained only in the source package"
            ),
            "new_source_native_library_supported": sum(
                value == "library_supported" for value in native.values()
            ),
        },
        "provenance": {
            "collection_report_sha256": sha256(collection_report_path),
            "response_entities_sha256": sha256(response_path),
            "base_source_report_sha256": sha256(base_source_report_path),
            "base_identity_partition_sha256": sha256(base_identity_path),
            "new_source_report_sha256": sha256(new_report_path),
            "new_source_mgf_sha256": sha256(new_mgf),
            "new_source_native_identity_sha256": sha256(new_identity),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "This manifest only supplies spectra and common identity-world labels for a model-independent candidate axis.",
            "The new source's native library labels are not promoted into the common known-only world.",
            "No candidate score, response value, relation truth, or retrieval result is used.",
        ],
    }
    source_output = output / "spectrum_source_manifest.json"
    source_output.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = {
        "schema": "contrast_atlas_v10_identity_axis_inputs_v1",
        "status": "DEVELOPMENT_CONSUMED_IDENTITY_AXIS_INPUTS_COMPLETE",
        "denominators": {
            "studies": len(response_entities),
            "response_rows": response_rows,
            "study_entities": sum(map(len, response_entities.values())),
            "common_library_supported": sum(
                row["identity_status"] == "library_supported" for row in combined_partition
            ),
            "common_unresolved": sum(
                row["identity_status"] == "unresolved_entity" for row in combined_partition
            ),
            "new_source_entities_conservatively_unresolved": len(native),
        },
        "outputs_sha256": {
            "identity_partition.tsv.gz": sha256(identity_output),
            "spectrum_source_manifest.json": sha256(source_output),
        },
        "provenance": source_manifest["provenance"],
        "claim_boundary": source_manifest["claim_boundary"],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (identity_output, source_output, report_path)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
