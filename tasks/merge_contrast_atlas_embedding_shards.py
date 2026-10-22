#!/usr/bin/env python3
"""Merge phenotype-blind per-study embedding shards without opening effects.

The merger reads only the frozen response entity registry, the source manifest,
and encoder provenance.  It deliberately does not load beta, standard errors,
contrast labels, registered relations, or retrieval results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "contrast_atlas_open_response_frozen_embeddings_v1"
STATUS = "PHENOTYPE_BLIND_FROZEN_OPEN_RESPONSE_EMBEDDINGS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_required_entities(collection: Path) -> tuple[pd.DataFrame, Path]:
    entities_path = collection / "response_entities.tsv.gz"
    if not entities_path.is_file():
        raise FileNotFoundError(entities_path)
    entities = pd.read_csv(
        entities_path,
        sep="\t",
        usecols=["study_id", "entity_id"],
        dtype=str,
    ).drop_duplicates()
    if entities["entity_id"].duplicated().any():
        raise RuntimeError("response collection contains non-unique global entity IDs")
    return entities, entities_path


def validate_archive(
    archive_path: Path,
    expected_sha256: str,
    expected_entities: set[str],
    expected_dimensions: int,
) -> int:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if sha256(archive_path) != expected_sha256:
        raise RuntimeError(f"embedding archive checksum drift: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as packed:
        if set(packed.files) != {"entity_id", "embeddings"}:
            raise RuntimeError(f"invalid embedding archive fields: {archive_path}")
        ids = [str(value).strip() for value in packed["entity_id"]]
        embeddings = np.asarray(packed["embeddings"])
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected_entities):
        raise RuntimeError(f"embedding/entity registry mismatch: {archive_path}")
    if embeddings.shape != (len(ids), expected_dimensions):
        raise RuntimeError(f"embedding shape mismatch: {archive_path} {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"non-finite embeddings: {archive_path}")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if len(norms) and float(np.max(np.abs(norms - 1.0))) > 2e-5:
        raise RuntimeError(f"non-unit embeddings: {archive_path}")
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    entities, entities_path = load_required_entities(args.collection)
    source_path = args.collection / "source_manifest.tsv"
    collection_report_path = args.collection / "report.json"
    for path in (source_path, collection_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = pd.read_csv(source_path, sep="\t", usecols=["study_id"], dtype=str)
    studies = list(sources["study_id"].astype(str))
    if len(studies) != len(set(studies)):
        raise RuntimeError("source manifest contains duplicate study IDs")
    if set(studies) != set(entities["study_id"].astype(str)):
        raise RuntimeError("source and response-entity study inventories differ")

    invariant_report_fields: dict[str, object] | None = None
    invariant_provenance: dict[str, object] | None = None
    manifest_rows: list[dict[str, str]] = []
    source_reports: list[dict[str, object]] = []
    shard_reports: dict[str, str] = {}
    total_requested = 0
    total_encoded = 0
    dimensions: int | None = None
    model_kind = ""

    for study in studies:
        shard = args.shards_root / study
        report_path = shard / "report.json"
        manifest_path = shard / "embedding_manifest.tsv"
        if not report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete embedding shard: {study}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA or report.get("status") != STATUS:
            raise RuntimeError(f"inadmissible embedding shard: {study}")
        if int(report.get("denominators", {}).get("independent_studies", -1)) != 1:
            raise RuntimeError(f"embedding shard is not source-isolated: {study}")
        report_sources = report.get("sources")
        if not isinstance(report_sources, list) or len(report_sources) != 1:
            raise RuntimeError(f"invalid source report inventory: {study}")
        source_report = report_sources[0]
        if str(source_report.get("study_id")) != study:
            raise RuntimeError(f"embedding shard study mismatch: {study}")

        comparable_fields = {
            key: report.get(key)
            for key in ("data_role", "model_kind", "checkpoint_kind", "leakage_status", "preprocessing")
        }
        comparable_provenance = {
            key: report.get("provenance", {}).get(key)
            for key in (
                "contract_sha256", "contract_lineage_sha256",
                "response_entities_sha256", "raw_checkpoint_sha256",
                "encoder_checkpoint_sha256", "encoder_model_state_sha256",
                "implementation_sha256",
            )
        }
        if invariant_report_fields is None:
            invariant_report_fields = comparable_fields
            invariant_provenance = comparable_provenance
            model_kind = str(report.get("model_kind"))
        elif comparable_fields != invariant_report_fields or comparable_provenance != invariant_provenance:
            raise RuntimeError(f"cross-shard encoder or collection provenance drift: {study}")
        if comparable_provenance["response_entities_sha256"] != sha256(entities_path):
            raise RuntimeError(f"shard was encoded from a different response registry: {study}")

        shard_manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
        required_columns = {"study_id", "embedding_path", "method_id"}
        if set(shard_manifest.columns) != required_columns or len(shard_manifest) != 1:
            raise RuntimeError(f"invalid embedding shard manifest: {study}")
        row = shard_manifest.iloc[0]
        if str(row.study_id) != study or str(row.method_id) != model_kind:
            raise RuntimeError(f"embedding shard manifest identity drift: {study}")
        archive_path = Path(str(row.embedding_path))
        if not archive_path.is_absolute():
            archive_path = Path.cwd() / archive_path
        current_dimensions = int(source_report["embedding_dimensions"])
        if dimensions is None:
            dimensions = current_dimensions
        elif current_dimensions != dimensions:
            raise RuntimeError(f"embedding dimensions differ across shards: {study}")
        required_entities = set(
            entities.loc[entities["study_id"].eq(study), "entity_id"].astype(str)
        )
        observed_n = validate_archive(
            archive_path,
            str(source_report["embedding_npz_sha256"]),
            required_entities,
            current_dimensions,
        )
        if observed_n != int(source_report["response_entities_encoded"]):
            raise RuntimeError(f"embedding shard denominator drift: {study}")
        requested_n = int(source_report.get("response_entities_requested", observed_n))
        ineligible_n = int(source_report.get("response_entities_ineligible", 0))
        if requested_n != len(required_entities) or requested_n - observed_n != ineligible_n:
            raise RuntimeError(f"embedding eligibility denominator drift: {study}")
        eligibility_name = str(source_report.get("spectral_eligibility", ""))
        eligibility_path = shard / eligibility_name
        if not eligibility_name or not eligibility_path.is_file():
            raise RuntimeError(f"embedding shard lacks spectral eligibility ledger: {study}")
        if sha256(eligibility_path) != str(source_report.get("spectral_eligibility_sha256", "")):
            raise RuntimeError(f"spectral eligibility checksum drift: {study}")
        eligibility = pd.read_csv(eligibility_path, sep="\t", dtype={"entity_id": str})
        if set(eligibility["entity_id"].astype(str)) != required_entities:
            raise RuntimeError(f"spectral eligibility/entity registry mismatch: {study}")
        total_requested += requested_n
        total_encoded += observed_n
        manifest_rows.append({
            "study_id": study,
            "embedding_path": str(row.embedding_path),
            "method_id": model_kind,
        })
        source_reports.append(source_report)
        shard_reports[study] = sha256(report_path)

    if dimensions is None or invariant_report_fields is None or invariant_provenance is None:
        raise RuntimeError("no embedding shards were found")
    if total_requested != len(entities):
        raise RuntimeError("merged embedding request denominator differs from response registry")

    args.output_dir.mkdir(parents=True)
    output_manifest = args.output_dir / "embedding_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(output_manifest, sep="\t", index=False)
    merged_report = {
        "schema": "contrast_atlas_open_response_merged_embedding_manifest_v1",
        "status": "PHENOTYPE_BLIND_FROZEN_OPEN_RESPONSE_EMBEDDING_SHARDS_MERGED",
        **invariant_report_fields,
        "denominators": {
            "independent_studies": len(studies),
            "response_entities_requested": total_requested,
            "response_entities_encoded": total_encoded,
            "response_entities_ineligible": total_requested - total_encoded,
            "embedding_dimensions": dimensions,
        },
        "sources": source_reports,
        "provenance": {
            **invariant_provenance,
            "source_manifest_sha256": sha256(source_path),
            "collection_report_sha256": sha256(collection_report_path),
            "shard_report_sha256": shard_reports,
            "shard_report_registry_sha256": canonical_sha256(shard_reports),
            "embedding_manifest_sha256": sha256(output_manifest),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Manifest merge only; no effect, contrast, registered relation, retrieval score, "
            "or biological truth was loaded."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(merged_report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        f"{sha256(output_manifest)}  {output_manifest.name}\n"
        f"{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(merged_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
