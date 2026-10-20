#!/usr/bin/env python3
"""Build a phenotype-blind, precursor-compatible cross-source identity graph.

This graph is deliberately distinct from a chemical-relation graph.  For every
query ion and target source it considers only candidates inside a frozen ppm
window, then selects the highest frozen-embedding cosine if that cosine passes
the minimum.  It reads no beta, SE, contrast name, biological label, or truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_contrast_atlas_identity_relation_boundary import read_precursors


WORLDS = ("all", "known", "unresolved")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_mask(status: np.ndarray, world: str) -> np.ndarray:
    if world == "all":
        return np.ones(len(status), dtype=bool)
    if world == "known":
        return status == "library_supported"
    if world == "unresolved":
        return status == "unresolved_entity"
    raise ValueError(world)


def directed_mass_top1(
    query_ids: np.ndarray,
    query_mz: np.ndarray,
    query_vectors: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_mz: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    ppm: float,
    minimum_cosine: float,
) -> list[tuple[str, str, float, float, float]]:
    order = np.argsort(candidate_mz, kind="stable")
    candidate_ids = candidate_ids[order]
    candidate_mz = candidate_mz[order]
    candidate_vectors = candidate_vectors[order]
    rows: list[tuple[str, str, float, float, float]] = []
    for entity_id, mz, vector in zip(query_ids, query_mz, query_vectors):
        tolerance = float(mz) * ppm * 1e-6
        begin = int(np.searchsorted(candidate_mz, mz - tolerance, side="left"))
        end = int(np.searchsorted(candidate_mz, mz + tolerance, side="right"))
        if begin == end:
            continue
        similarities = candidate_vectors[begin:end] @ vector
        best_local = int(np.argmax(similarities))
        best_score = float(similarities[best_local])
        if best_score < minimum_cosine:
            continue
        target_index = begin + best_local
        target_mz = float(candidate_mz[target_index])
        symmetric_ppm = abs(float(mz) - target_mz) / ((float(mz) + target_mz) / 2.0) * 1e6
        rows.append((str(entity_id), str(candidate_ids[target_index]), best_score, float(mz), symmetric_ppm))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--minimum-cosine", type=float, default=0.70)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.ppm <= 0 or not 0 <= args.minimum_cosine <= 1:
        raise ValueError("invalid ppm or cosine threshold")

    # Identity status defines controls, not chemistry or biological truth.
    entities_path = args.collection / "response_entities.tsv.gz"
    registry = pd.read_csv(
        entities_path, sep="\t", usecols=["study_id", "entity_id", "identity_status"], dtype=str,
    ).drop_duplicates()
    if registry.duplicated(["study_id", "entity_id"]).any():
        raise ValueError("duplicate entity registry rows")
    if not set(registry.identity_status).issubset({"library_supported", "unresolved_entity"}):
        raise ValueError("unsupported identity partition")

    contract = json.loads(args.collection_contract.read_text(encoding="utf-8"))
    source_by_study = {str(source["study_id"]): source for source in contract["sources"]}
    jobs = pd.read_csv(args.embedding_manifest, sep="\t", dtype=str)
    if set(jobs.study_id) != set(source_by_study) or set(registry.study_id) != set(source_by_study):
        raise ValueError("contract, registry, and embedding studies differ")
    if jobs.method_id.nunique() != 1:
        raise ValueError("embedding manifest must contain one method")

    sources: dict[str, dict[str, object]] = {}
    provenance: dict[str, str] = {
        "collection_contract_sha256": sha256(args.collection_contract),
        "response_entities_sha256": sha256(entities_path),
        "embedding_manifest_sha256": sha256(args.embedding_manifest),
    }
    for job in jobs.itertuples(index=False):
        study = str(job.study_id)
        embedding_path = Path(job.embedding_path)
        archive = np.load(embedding_path, allow_pickle=False)
        ids = archive["entity_id"].astype(str)
        vectors = archive["embeddings"].astype(np.float32)
        if len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError(f"invalid embeddings for {study}")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(vectors).all() or float(np.max(np.abs(norms - 1))) > 2e-5:
            raise ValueError(f"non-unit embeddings for {study}")
        status_map = registry.loc[registry.study_id.eq(study)].set_index("entity_id").identity_status
        if set(ids) != set(status_map.index):
            raise ValueError(f"embedding and registry entities differ for {study}")
        mgf_path = Path(source_by_study[study]["mgf_path"])
        precursor_map = read_precursors(mgf_path, study)
        missing = set(ids) - set(precursor_map)
        if missing:
            raise ValueError(f"{study} lacks precursor mass for {len(missing)} entities")
        sources[study] = {
            "ids": ids,
            "vectors": vectors,
            "mz": np.asarray([precursor_map[value] for value in ids], dtype=float),
            "status": np.asarray([status_map[value] for value in ids], dtype=str),
        }
        provenance[f"embedding_{study}_sha256"] = sha256(embedding_path)
        provenance[f"mgf_{study}_sha256"] = sha256(mgf_path)

    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    studies = sorted(sources)
    for left_index, left_study in enumerate(studies):
        for right_study in studies[left_index + 1:]:
            left = sources[left_study]
            right = sources[right_study]
            for world in WORLDS:
                left_mask = world_mask(left["status"], world)
                right_mask = world_mask(right["status"], world)
                forward = directed_mass_top1(
                    left["ids"][left_mask], left["mz"][left_mask], left["vectors"][left_mask],
                    right["ids"][right_mask], right["mz"][right_mask], right["vectors"][right_mask],
                    ppm=args.ppm, minimum_cosine=args.minimum_cosine,
                )
                reverse = directed_mass_top1(
                    right["ids"][right_mask], right["mz"][right_mask], right["vectors"][right_mask],
                    left["ids"][left_mask], left["mz"][left_mask], left["vectors"][left_mask],
                    ppm=args.ppm, minimum_cosine=args.minimum_cosine,
                )
                coverage_rows.append({
                    "left_study_id": left_study,
                    "right_study_id": right_study,
                    "evidence_world": world,
                    "left_universe_entities": int(left_mask.sum()),
                    "right_universe_entities": int(right_mask.sum()),
                    "left_to_right_edges": len(forward),
                    "right_to_left_edges": len(reverse),
                })
                for direction, values in (("left_to_right", forward), ("right_to_left", reverse)):
                    for query_id, candidate_id, cosine, query_mz, delta_ppm in values:
                        if direction == "left_to_right":
                            left_id, right_id = query_id, candidate_id
                        else:
                            left_id, right_id = candidate_id, query_id
                        rows.append({
                            "left_study_id": left_study,
                            "right_study_id": right_study,
                            "evidence_world": world,
                            "direction": direction,
                            "left_entity_id": left_id,
                            "right_entity_id": right_id,
                            "chemical_similarity": cosine,
                            "query_precursor_mz": query_mz,
                            "precursor_delta_ppm": delta_ppm,
                        })

    edges = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    args.output_dir.mkdir(parents=True)
    edge_path = args.output_dir / "source_pair_frozen_edges.tsv.gz"
    coverage_path = args.output_dir / "source_pair_frozen_graph_summary.tsv"
    edges.to_csv(edge_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    coverage.to_csv(coverage_path, sep="\t", index=False)
    report = {
        "schema": "contrast_atlas_mass_compatible_identity_graph_v1",
        "status": "DEVELOPMENT_CONSUMED_MASS_COMPATIBLE_IDENTITY_GRAPH_FROZEN",
        "data_role": "development_consumed",
        "method_id": str(jobs.method_id.iloc[0]),
        "parameters": {
            "precursor_tolerance_ppm": args.ppm,
            "minimum_cosine": args.minimum_cosine,
            "neighbors_per_query_target_source": 1,
            "selection": "maximum frozen-embedding cosine among precursor-compatible candidates",
        },
        "leakage_status": (
            "graph reads only entity IDs, frozen identity partitions, precursor masses, and frozen embeddings; "
            "no beta, SE, contrast, phenotype, relation label, or strict identity truth"
        ),
        "denominators": {
            "studies": len(studies),
            "entities": int(len(registry)),
            "source_pair_world_graphs": int(len(coverage)),
            "directional_edges": int(len(edges)),
            "all_world_directional_edges": int(edges.evidence_world.eq("all").sum()),
        },
        "claim_boundary": [
            "The 10 ppm default is frozen for this post-audit development build, not an external validation choice.",
            "Precursor compatibility and spectral similarity do not by themselves prove identity.",
            "Known and unresolved worlds are controls; unresolved does not mean a unique novel metabolite.",
            "This graph contains no biological-response result."
        ],
        "provenance": {
            **provenance,
            "edges_sha256": sha256(edge_path),
            "graph_summary_sha256": sha256(coverage_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
