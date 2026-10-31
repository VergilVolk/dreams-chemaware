#!/usr/bin/env python3
"""Encode only frozen v10 Atlas entities with one registered checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from encode_crossstudy_mgfs_checkpoint import embed_records
from encode_pan_redu_panel_spectra import load_mgf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-artifact", type=Path, required=True)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(args.output)
    for path in (
        args.axis_dir / "report.json",
        args.axis_dir / "observed_ion_registry.tsv.gz",
        args.source_manifest,
        args.checkpoint,
        args.raw_checkpoint,
        args.canonical_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    axis_report_path = args.axis_dir / "report.json"
    registry_path = args.axis_dir / "observed_ion_registry.tsv.gz"
    axis_report = json.loads(axis_report_path.read_text(encoding="utf-8"))
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if axis_report.get("status") != "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN":
        raise RuntimeError("refusing a non-frozen Atlas identity axis")
    if int(axis_report["denominators"].get("studies", -1)) != 16 or int(axis_report["denominators"].get("entities", -1)) != 55_127:
        raise RuntimeError("v10 Atlas denominator drifted")
    if sources.get("status") != "DEVELOPMENT_CONSUMED_SPECTRUM_SOURCE_MANIFEST_COMPLETE":
        raise RuntimeError("source manifest is not complete")

    registry = pd.read_csv(
        registry_path, sep="\t", dtype={"study_id": str, "raw_entity_id": str, "entity_id": str},
    )
    if len(registry) != 55_127 or registry["entity_id"].duplicated().any():
        raise RuntimeError("frozen entity registry drifted")
    source_by_study = {str(row["study_id"]): row for row in sources["sources"]}
    if set(source_by_study) != set(registry["study_id"].astype(str)):
        raise RuntimeError("entity registry and source manifest study sets differ")

    entity_ids: list[str] = []
    records: list[dict[str, object]] = []
    encoded_by_study: dict[str, int] = {}
    abstained_by_study: dict[str, int] = {}
    for study in sorted(source_by_study):
        source = source_by_study[study]
        path = Path(str(source["mgf_path"]).replace("\\", "/"))
        if sha256(path) != str(source["mgf_sha256"]):
            raise RuntimeError(f"source MGF hash drifted: {study}")
        required_frame = registry.loc[registry["study_id"].astype(str).eq(study)]
        required = dict(zip(required_frame["raw_entity_id"].astype(str), required_frame["entity_id"].astype(str), strict=True))
        all_ids, all_records = load_mgf(path, minimum_peaks=0, skip_ineligible=False)
        found: set[str] = set()
        eligible = 0
        abstained = 0
        for raw_id, record in zip(all_ids, all_records, strict=True):
            entity_id = required.get(str(raw_id))
            if entity_id is None:
                continue
            if entity_id in found:
                raise RuntimeError(f"duplicate representative spectrum: {entity_id}")
            found.add(entity_id)
            if np.asarray(record["peaks"]).shape[-1] < 3:
                abstained += 1
                continue
            entity_ids.append(entity_id)
            records.append(record)
            eligible += 1
        missing = set(required.values()) - found
        if missing:
            raise RuntimeError(f"{study} misses {len(missing)} frozen representative spectra")
        encoded_by_study[study] = eligible
        abstained_by_study[study] = abstained

    if len(entity_ids) != len(set(entity_ids)):
        raise RuntimeError("encoded entity identifiers are not unique")
    embeddings, checkpoint_kind = embed_records(
        records,
        args.checkpoint,
        args.raw_checkpoint,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    if embeddings.shape != (len(entity_ids), 1024):
        raise RuntimeError(f"embedding matrix shape drifted: {embeddings.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        entity_id=np.asarray(entity_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )
    report = {
        "schema": "contrast_atlas_v10_frozen_encoder_cache_v1",
        "status": "CONTRAST_ATLAS_V10_FROZEN_ENCODER_EMBEDDINGS_COMPLETE",
        "encoder_name": args.encoder_name,
        "checkpoint_kind": checkpoint_kind,
        "labels_or_responses_accessed": False,
        "embedding_dimension": 1024,
        "denominators": {
            "frozen_entities": 55_127,
            "encoded_entities": len(entity_ids),
            "abstained_lt3_peaks": int(sum(abstained_by_study.values())),
            "encoded_by_study": encoded_by_study,
            "abstained_by_study": abstained_by_study,
        },
        "inputs": {
            "axis_report": {"path": str(axis_report_path), "sha256": sha256(axis_report_path)},
            "entity_registry": {"path": str(registry_path), "sha256": sha256(registry_path)},
            "source_manifest": {"path": str(args.source_manifest), "sha256": sha256(args.source_manifest)},
            "application_checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
            "canonical_artifact": {"path": str(args.canonical_artifact), "sha256": sha256(args.canonical_artifact)},
            "raw_architecture": {"path": str(args.raw_checkpoint), "sha256": sha256(args.raw_checkpoint)},
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "claim_boundary": "Spectrum-only frozen inference; no response, phenotype, relation truth or identity truth was read.",
    }
    sidecar_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
