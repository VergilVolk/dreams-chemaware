#!/usr/bin/env python3
"""Align entity embedding banks to a frozen CALMS candidate-evidence axis.

This adapter is deliberately score- and truth-blind.  Each query/candidate
entity contributes its single consensus embedding as reference slot zero;
unused Top-3 slots remain masked instead of duplicating a spectrum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ENCODER_NAMES = (
    "noise_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
REPRESENTATION_SCHEMA = "calms_aligned_representation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--embedding-bank must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if name not in ENCODER_NAMES or name in output or not raw_path:
            raise ValueError(f"invalid or duplicate embedding bank: {name!r}")
        output[name] = Path(raw_path)
    if set(output) != set(ENCODER_NAMES):
        raise RuntimeError("exactly four named embedding banks are required")
    return output


def load_bank(path: Path, name: str) -> tuple[np.ndarray, np.ndarray, str]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as body:
        if "entity_id" not in body.files or "embeddings" not in body.files:
            raise RuntimeError(f"{name} bank must contain entity_id and embeddings")
        entity_ids = np.asarray(body["entity_id"]).astype("U")
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if (
        embeddings.ndim != 2
        or embeddings.shape != (len(entity_ids), 1024)
        or len(np.unique(entity_ids)) != len(entity_ids)
        or not np.all(np.isfinite(embeddings))
    ):
        raise RuntimeError(f"invalid embedding bank: {name}")
    return entity_ids, embeddings, digest


def build(
    evidence_path: Path,
    evidence_manifest_path: Path,
    pair_table_path: Path,
    bank_paths: dict[str, Path],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    evidence_sha = sha256_file(evidence_path)
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    if evidence_manifest.get("evidence_sha256") != evidence_sha:
        raise RuntimeError("candidate-evidence manifest hash mismatch")
    upstream = evidence_manifest.get("upstream_artifacts")
    if not isinstance(upstream, dict):
        raise RuntimeError("candidate-evidence manifest lacks upstream registry")

    with np.load(evidence_path, allow_pickle=False) as body:
        required = {"candidate_pair_ids", "candidate_query_ids", "candidate_ids"}
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"candidate evidence lacks axis arrays: {missing}")
        pair_ids = np.asarray(body["candidate_pair_ids"]).astype("U")
        candidate_query_ids = np.asarray(body["candidate_query_ids"]).astype("U")
        candidate_ids = np.asarray(body["candidate_ids"]).astype("U")
    if len(pair_ids) == 0 or len(np.unique(pair_ids)) != len(pair_ids):
        raise RuntimeError("candidate_pair_ids must be non-empty and unique")

    pairs = pd.read_csv(
        pair_table_path,
        sep="\t",
        usecols=["candidate_pair_id", "query_id", "candidate_study_id", "candidate_id"],
        dtype={
            "candidate_pair_id": "string",
            "query_id": "string",
            "candidate_study_id": "string",
            "candidate_id": "string",
        },
    )
    if pairs["candidate_pair_id"].duplicated().any():
        raise RuntimeError("pair table has duplicate candidate_pair_id")
    indexed = pairs.set_index("candidate_pair_id", drop=False)
    if set(indexed.index.astype(str)) != set(pair_ids.tolist()):
        raise RuntimeError("pair table and candidate-evidence pair axes differ")
    pairs = indexed.loc[pair_ids.tolist()].reset_index(drop=True)
    expected_query_ids = (
        pairs["query_id"].astype(str) + "|" + pairs["candidate_study_id"].astype(str)
    ).to_numpy(dtype="U")
    if not np.array_equal(expected_query_ids, candidate_query_ids):
        raise RuntimeError("pair table and candidate-evidence query axes differ")
    if not np.array_equal(pairs["candidate_id"].astype(str).to_numpy(dtype="U"), candidate_ids):
        raise RuntimeError("pair table candidate identity order drifted")
    query_entities = pairs["query_id"].astype(str).to_numpy(dtype="U")

    n_candidates = len(pair_ids)
    query_positions = np.full((n_candidates, len(ENCODER_NAMES)), -1, dtype=np.int64)
    reference_positions = np.full((n_candidates, len(ENCODER_NAMES), 3), -1, dtype=np.int64)
    reference_mask = np.zeros((n_candidates, len(ENCODER_NAMES), 3), dtype=bool)
    bank_registry: dict[str, dict[str, str]] = {}
    entity_rows: dict[str, dict[str, int]] = {}

    for encoder_index, name in enumerate(ENCODER_NAMES):
        entity_ids, _, bank_sha = load_bank(bank_paths[name], name)
        row_by_entity = {entity_id: row for row, entity_id in enumerate(entity_ids.tolist())}
        entity_rows[name] = row_by_entity
        query_positions[:, encoder_index] = np.fromiter(
            (row_by_entity.get(value, -1) for value in query_entities), dtype=np.int64,
            count=n_candidates,
        )
        reference_positions[:, encoder_index, 0] = np.fromiter(
            (row_by_entity.get(value, -1) for value in candidate_ids), dtype=np.int64,
            count=n_candidates,
        )

        identity = upstream.get(name)
        if not isinstance(identity, dict):
            raise RuntimeError(f"candidate-evidence manifest lacks encoder identity: {name}")
        application_cache = identity.get("application_cache")
        if isinstance(application_cache, dict) and application_cache.get("sha256") != bank_sha:
            raise RuntimeError(f"embedding bank differs from evidence source: {name}")
        checkpoint_sha = identity.get("canonical_checkpoint_sha256", identity.get("sha256"))
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            raise RuntimeError(f"candidate-evidence manifest lacks checkpoint identity: {name}")
        bank_registry[name] = {
            "sha256": bank_sha,
            "encoder_checkpoint_sha256": checkpoint_sha.lower(),
        }

    # The frozen residual requires all four representations.  If any encoder
    # lacks a candidate entity, all four reference slots remain masked for that
    # row; the release then retains its parent score instead of fabricating a
    # zero vector or duplicating another reference.
    complete_candidate = np.all(reference_positions[:, :, 0] >= 0, axis=1)
    reference_mask[complete_candidate, :, 0] = True
    complete_query = np.all(query_positions >= 0, axis=1)

    arrays = {
        "candidate_pair_ids": pair_ids,
        "query_positions": query_positions,
        "reference_positions": reference_positions,
        "reference_mask": reference_mask,
    }
    manifest: dict[str, object] = {
        "schema": REPRESENTATION_SCHEMA,
        "representation_sha256": None,
        "candidate_evidence_sha256": evidence_sha,
        "embedding_banks": bank_registry,
        "axis": {
            "candidate_pairs": n_candidates,
            "reference_slots_available_per_encoder": 1,
            "unused_reference_slots_masked": 2,
            "rows_with_all_query_embeddings": int(complete_query.sum()),
            "rows_with_all_candidate_embeddings": int(complete_candidate.sum()),
            "rows_eligible_for_all_four_representations": int(
                np.sum(complete_query & complete_candidate)
            ),
        },
        "claim_boundary": (
            "Frozen candidate axis with one consensus spectrum per entity. "
            "No labels, biological responses, or duplicated pseudo-references enter scoring."
        ),
    }
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--embedding-bank", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    banks = named_paths(args.embedding_bank)
    for path in (args.evidence, args.evidence_manifest, args.pair_table, *banks.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    arrays, manifest = build(
        args.evidence, args.evidence_manifest, args.pair_table, banks,
    )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "candidate_references.npz"
    np.savez_compressed(output, **arrays)
    manifest["representation_sha256"] = sha256_file(output)
    manifest_path = args.output_dir / "candidate_references.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "CALMS_SINGLE_REFERENCE_REPRESENTATION_COMPLETE",
        "output": str(output),
        "manifest": str(manifest_path),
        "candidate_pairs": int(len(arrays["candidate_pair_ids"])),
    }, indent=2))


if __name__ == "__main__":
    main()
