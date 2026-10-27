#!/usr/bin/env python3
"""Encode only fixed Contrast-Atlas edge endpoints missing from frozen caches.

The edge handoff is the sole entity request list. Existing embedding caches are
used only to skip rows that have already been encoded; no phenotype, response,
registered relation, or ranking output is read. The output can therefore be
merged with the existing caches without changing the frozen candidate graph.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from encode_crossstudy_mgfs_checkpoint import (
    MINIMUM_FRAGMENT_PEAKS,
    embed_records,
    load_eligible_mgf,
    sha256,
)


def requested_entities(edge_path: Path) -> dict[str, set[str]]:
    edges = pd.read_csv(
        edge_path,
        sep="\t",
        dtype=str,
        usecols=[
            "left_study_id",
            "right_study_id",
            "left_entity_id",
            "right_entity_id",
        ],
    )
    requested: dict[str, set[str]] = defaultdict(set)
    for side in ("left", "right"):
        for study, entity in zip(
            edges[f"{side}_study_id"], edges[f"{side}_entity_id"], strict=True
        ):
            study = str(study)
            entity = str(entity)
            prefix = f"{study}::"
            if not entity.startswith(prefix):
                raise RuntimeError(
                    f"edge entity {entity!r} is not qualified by study {study!r}"
                )
            requested[study].add(entity)
    return dict(requested)


def cached_entities(paths: list[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as body:
            if "entity_id" not in body or "embeddings" not in body:
                raise RuntimeError(f"invalid embedding cache: {path}")
            ids = np.asarray(body["entity_id"]).astype("U")
            embeddings = np.asarray(body["embeddings"])
        if embeddings.ndim != 2 or len(ids) != embeddings.shape[0]:
            raise RuntimeError(f"misaligned embedding cache: {path}")
        overlap = output.intersection(ids.tolist())
        if overlap:
            raise RuntimeError(
                f"existing caches overlap for {len(overlap)} entity IDs: {path}"
            )
        output.update(ids.tolist())
    return output


def source_paths(report_path: Path) -> dict[str, Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sources = report.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("alignment report lacks a sources list")
    output: dict[str, Path] = {}
    for source in sources:
        study = str(source["study_id"])
        path = Path(source["mgf_path"])
        if study in output:
            raise RuntimeError(f"duplicate study in alignment report: {study}")
        if not path.is_file():
            raise FileNotFoundError(path)
        output[study] = path
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-handoff", required=True, type=Path)
    parser.add_argument("--alignment-report", required=True, type=Path)
    parser.add_argument("--existing-embedding", action="append", default=[], type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--raw-checkpoint", required=True, type=Path)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    report_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError(args.output)
    for path in [
        args.edge_handoff,
        args.alignment_report,
        args.checkpoint,
        args.raw_checkpoint,
        *args.existing_embedding,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    requested = requested_entities(args.edge_handoff)
    paths = source_paths(args.alignment_report)
    if set(requested) - set(paths):
        raise RuntimeError(
            f"alignment report lacks requested studies: {sorted(set(requested) - set(paths))}"
        )
    cached = cached_entities(args.existing_embedding)

    global_ids: list[str] = []
    records: list[dict[str, object]] = []
    study_summary: dict[str, dict[str, int]] = {}
    unresolved: list[str] = []
    for study in sorted(requested):
        target = requested[study]
        already = target.intersection(cached)
        remaining = target - already
        if not remaining:
            study_summary[study] = {
                "requested": len(target),
                "already_cached": len(already),
                "encoded": 0,
                "unavailable_or_lt3_peaks": 0,
            }
            continue

        eligible_ids, eligible_records, _absent = load_eligible_mgf(paths[study])
        by_global = {
            f"{study}::{raw_id}": record
            for raw_id, record in zip(eligible_ids, eligible_records, strict=True)
        }
        selected = sorted(remaining.intersection(by_global))
        missing = sorted(remaining - set(selected))
        global_ids.extend(selected)
        records.extend(by_global[entity] for entity in selected)
        unresolved.extend(missing)
        study_summary[study] = {
            "requested": len(target),
            "already_cached": len(already),
            "encoded": len(selected),
            "unavailable_or_lt3_peaks": len(missing),
        }

    if len(global_ids) != len(set(global_ids)):
        raise RuntimeError("selected study-qualified entity IDs are not unique")
    if not records:
        raise RuntimeError("all requested endpoints were already cached or unavailable")

    embeddings, checkpoint_kind = embed_records(
        records,
        args.checkpoint,
        args.raw_checkpoint,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    if embeddings.shape[0] != len(global_ids):
        raise RuntimeError("embedding rows do not match selected endpoint registry")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        entity_id=np.asarray(global_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )
    report = {
        "status": "FIXED_EDGE_ENDPOINT_EMBEDDINGS_COMPLETE",
        "encoder_name": args.encoder_name,
        "checkpoint_kind": checkpoint_kind,
        "labels_responses_or_relations_accessed": False,
        "selection_contract": (
            "only endpoint IDs from the frozen edge handoff; existing caches skip "
            "work but cannot add an entity"
        ),
        "minimum_fragment_peaks": MINIMUM_FRAGMENT_PEAKS,
        "denominators": {
            "requested_unique_endpoints": int(sum(len(values) for values in requested.values())),
            "already_cached_requested_endpoints": int(
                sum(len(values.intersection(cached)) for values in requested.values())
            ),
            "newly_encoded_endpoints": len(global_ids),
            "unavailable_or_lt3_peak_endpoints": len(unresolved),
        },
        "studies": study_summary,
        "unavailable_entity_ids": unresolved,
        "inputs": {
            "edge_handoff": {"path": str(args.edge_handoff), "sha256": sha256(args.edge_handoff)},
            "alignment_report": {
                "path": str(args.alignment_report),
                "sha256": sha256(args.alignment_report),
            },
            "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
            "raw_checkpoint": {
                "path": str(args.raw_checkpoint),
                "sha256": sha256(args.raw_checkpoint),
            },
            "existing_embeddings": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.existing_embedding
            ],
            "mgfs_read": {
                study: {"path": str(paths[study]), "sha256": sha256(paths[study])}
                for study, values in study_summary.items()
                if values["requested"] > values["already_cached"]
            },
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
