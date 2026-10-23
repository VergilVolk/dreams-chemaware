#!/usr/bin/env python3
"""Structure-agnostic local response alignment from sparse MS/MS fragments.

For every pair of source studies, top-1 fragment-cosine edges are frozen in
both directions before beta, standard errors, contrast definitions, or the
registered relation table are opened.  Study-level response similarity is then
computed from locally aligned entity response weights rather than a global
mean spectrum.  The complete entity universe remains in the denominator;
missing edges contribute zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from build_contrast_atlas_open_response_collection import load_contract
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    WORLDS,
    load_study_spectra,
    normalise_entity_id,
    pad_columns,
    sha256,
)
from score_contrast_atlas_symmetric_nn_alignment import (
    deterministic_profile_reassignment,
    frozen_source_universes,
    graph_summary,
    relation_audit,
    response_profiles,
    score_all,
)


NEIGHBORS = 1
MINIMUM_COSINE = 0.70


def world_mask(status: np.ndarray, world: str) -> np.ndarray:
    if world == "all":
        return np.ones(len(status), dtype=bool)
    if world == "known":
        return status == "library_supported"
    if world == "unresolved":
        return status == "unresolved_entity"
    raise ValueError(world)


def freeze_sparse_graphs(
    studies: list[str],
    universes: dict[str, pd.DataFrame],
    spectra: dict[str, tuple[np.ndarray, sparse.csr_matrix]],
    chunk_rows: int,
    minimum_cosine: float,
) -> tuple[dict[tuple[str, str, str], dict[str, object]], pd.DataFrame]:
    """Freeze three identity-world top-1 graphs using one sparse product/pair."""
    graphs: dict[tuple[str, str, str], dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []
    for left_index, left_study in enumerate(sorted(studies)):
        left_ids, left_matrix = spectra[left_study]
        left_status_lookup = dict(
            zip(
                universes[left_study]["entity_id"].astype(str),
                universes[left_study]["identity_status"].astype(str),
            )
        )
        left_status = np.asarray([left_status_lookup[value] for value in left_ids])
        for right_study in sorted(studies)[left_index + 1 :]:
            right_ids, right_matrix = spectra[right_study]
            right_status_lookup = dict(
                zip(
                    universes[right_study]["entity_id"].astype(str),
                    universes[right_study]["identity_status"].astype(str),
                )
            )
            right_status = np.asarray([right_status_lookup[value] for value in right_ids])
            left_masks = {world: world_mask(left_status, world) for world in WORLDS}
            right_masks = {world: world_mask(right_status, world) for world in WORLDS}
            forward: dict[str, dict[str, tuple[str, float]]] = {
                world: {} for world in WORLDS
            }
            reverse_best_score = {
                world: np.full(len(right_ids), -np.inf, dtype=np.float32)
                for world in WORLDS
            }
            reverse_best_left = {
                world: np.full(len(right_ids), -1, dtype=np.int64)
                for world in WORLDS
            }

            for start in range(0, len(left_ids), chunk_rows):
                stop = min(start + chunk_rows, len(left_ids))
                block = (left_matrix[start:stop] @ right_matrix.T).toarray()
                for world in WORLDS:
                    local_left = np.flatnonzero(left_masks[world][start:stop])
                    selected_right = np.flatnonzero(right_masks[world])
                    if not len(local_left) or not len(selected_right):
                        continue
                    local = block[np.ix_(local_left, selected_right)]

                    best_right_position = np.argmax(local, axis=1)
                    best_forward_score = local[
                        np.arange(len(local_left)), best_right_position
                    ]
                    for row_position, score in enumerate(best_forward_score):
                        if float(score) + 1e-12 < minimum_cosine:
                            continue
                        left_row = start + int(local_left[row_position])
                        right_row = int(selected_right[best_right_position[row_position]])
                        forward[world][str(left_ids[left_row])] = (
                            str(right_ids[right_row]), float(score)
                        )

                    best_left_position = np.argmax(local, axis=0)
                    best_reverse_score = local[
                        best_left_position, np.arange(len(selected_right))
                    ]
                    best_left_global = start + local_left[best_left_position]
                    previous = reverse_best_score[world][selected_right]
                    improve = best_reverse_score > previous + 1e-12
                    if improve.any():
                        changed = selected_right[improve]
                        reverse_best_score[world][changed] = best_reverse_score[improve]
                        reverse_best_left[world][changed] = best_left_global[improve]

            for world in WORLDS:
                reverse: dict[str, tuple[str, float]] = {}
                for right_row in np.flatnonzero(right_masks[world]):
                    score = float(reverse_best_score[world][right_row])
                    left_row = int(reverse_best_left[world][right_row])
                    if left_row >= 0 and score + 1e-12 >= minimum_cosine:
                        reverse[str(right_ids[right_row])] = (
                            str(left_ids[left_row]), score
                        )
                graphs[(left_study, right_study, world)] = {
                    "left_ids": left_ids[left_masks[world]],
                    "right_ids": right_ids[right_masks[world]],
                    "left_to_right": forward[world],
                    "right_to_left": reverse,
                }
                for direction, mapping in (
                    ("left_to_right", forward[world]),
                    ("right_to_left", reverse),
                ):
                    for source_id, (target_id, cosine) in mapping.items():
                        if direction == "left_to_right":
                            left_entity, right_entity = source_id, target_id
                        else:
                            left_entity, right_entity = target_id, source_id
                        edge_rows.append({
                            "left_study_id": left_study,
                            "right_study_id": right_study,
                            "evidence_world": world,
                            "direction": direction,
                            "left_entity_id": left_entity,
                            "right_entity_id": right_entity,
                            "chemical_similarity": cosine,
                        })
    return graphs, pd.DataFrame(edge_rows)


def load_sparse_sources(
    contract: dict[str, object],
    registry: pd.DataFrame,
    studies: list[str],
    bin_width: float,
) -> tuple[dict[str, tuple[np.ndarray, sparse.csr_matrix]], list[dict[str, object]], int]:
    source_contract = {str(item["study_id"]): item for item in contract["sources"]}
    raw_sources: dict[str, tuple[np.ndarray, sparse.csr_matrix]] = {}
    reports: list[dict[str, object]] = []
    dimensions = 0
    for study_id in studies:
        block = registry.loc[registry["study_id"].eq(study_id)]
        raw_to_entity = dict(zip(block["raw_entity_id"], block["entity_id"]))
        if len(raw_to_entity) != len(block):
            raise RuntimeError(f"raw entity IDs are not unique for {study_id}")
        mgf_path = Path(source_contract[study_id]["mgf_path"])
        raw_ids, matrix, details = load_study_spectra(
            mgf_path, set(raw_to_entity), bin_width
        )
        entity_ids = np.asarray([raw_to_entity[value] for value in raw_ids], dtype=str)
        order = np.argsort(entity_ids, kind="stable")
        entity_ids = entity_ids[order]
        matrix = matrix[order].tocsr()
        raw_sources[study_id] = (entity_ids, matrix)
        dimensions = max(dimensions, matrix.shape[1])
        reports.append({
            "study_id": study_id,
            "mgf_path": str(mgf_path),
            "mgf_sha256": sha256(mgf_path),
            **details,
        })
    sources = {
        study: (ids, pad_columns(matrix, dimensions))
        for study, (ids, matrix) in raw_sources.items()
    }
    return sources, reports, dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width-da", type=float, default=0.1)
    parser.add_argument("--minimum-cosine", type=float, default=MINIMUM_COSINE)
    parser.add_argument("--chunk-rows", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.chunk_rows < 1 or args.bin_width_da <= 0:
        raise ValueError("invalid chunk or bin width")

    root = Path(__file__).resolve().parents[1]
    contract, contract_paths = load_contract(args.contract.resolve(), root)
    paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "sources": args.collection / "source_manifest.tsv",
        "relations": args.collection / "registered_relations.tsv",
    }
    for path in (*paths.values(), args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Graph construction opens only source/entity identifiers, spectra and the
    # phenotype-blind common identity partition.
    raw_registry = pd.read_csv(
        paths["entities"], sep="\t",
        usecols=["study_id", "raw_entity_id", "entity_id"],
        dtype=str, low_memory=False,
    ).drop_duplicates()
    raw_registry["raw_entity_id"] = raw_registry["raw_entity_id"].map(normalise_entity_id)
    identity_report_path = args.identity_partition.parent / "report.json"
    identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
    if identity_report.get("status") != "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITIONS_COMPLETE":
        raise RuntimeError("identity partition is not the complete common policy")
    if identity_report["provenance"]["combined_partition_sha256"] != sha256(args.identity_partition):
        raise RuntimeError("identity partition checksum drift")
    identity = pd.read_csv(
        args.identity_partition, sep="\t",
        usecols=["study_id", "entity_id", "identity_status"], dtype=str,
    )
    registry = raw_registry.merge(
        identity, on=["study_id", "entity_id"], how="left", validate="one_to_one"
    )
    if registry["identity_status"].isna().any():
        raise RuntimeError("common identity partition does not cover the registry")
    sources_frame = pd.read_csv(paths["sources"], sep="\t", dtype=str)
    studies = sources_frame["study_id"].astype(str).tolist()
    universes = frozen_source_universes(registry)
    spectra, source_reports, dimensions = load_sparse_sources(
        contract, registry, studies, args.bin_width_da
    )
    graphs, edges = freeze_sparse_graphs(
        studies, universes, spectra, args.chunk_rows, args.minimum_cosine
    )
    graph_table = graph_summary(graphs)

    # Only after every graph is frozen are response effects and relations read.
    entities = pd.read_csv(paths["entities"], sep="\t", dtype={"entity_id": str}, low_memory=False)
    contrasts = pd.read_csv(paths["contrasts"], sep="\t").reset_index(drop=True)
    observed_profiles = response_profiles(entities)
    reassigned_profiles, reassignment = deterministic_profile_reassignment(
        universes, observed_profiles, contrasts
    )
    observed_scores = score_all(contrasts, graphs, observed_profiles, "observed")
    sensitivity_scores = score_all(
        contrasts, graphs, reassigned_profiles,
        "deterministic_within_study_identity_stratum_profile_reassignment",
    )
    relations = pd.read_csv(paths["relations"], sep="\t", dtype=str)
    audits = pd.concat([
        relation_audit(observed_scores, relations),
        relation_audit(sensitivity_scores, relations),
    ], ignore_index=True)

    args.output_dir.mkdir(parents=True)
    outputs = {
        "scores": args.output_dir / "fixed_candidate_scores.tsv.gz",
        "sensitivity": args.output_dir / "deterministic_profile_reassignment_scores.tsv.gz",
        "mapping": args.output_dir / "deterministic_profile_reassignment.tsv.gz",
        "edges": args.output_dir / "source_pair_frozen_edges.tsv.gz",
        "graphs": args.output_dir / "source_pair_frozen_graph_summary.tsv",
        "relations": args.output_dir / "registered_relation_absolute_ranks.tsv",
    }
    compression = {"method": "gzip", "mtime": 0}
    observed_scores.to_csv(outputs["scores"], sep="\t", index=False, compression=compression)
    sensitivity_scores.to_csv(outputs["sensitivity"], sep="\t", index=False, compression=compression)
    reassignment.to_csv(outputs["mapping"], sep="\t", index=False, compression=compression)
    edges.to_csv(outputs["edges"], sep="\t", index=False, compression=compression)
    graph_table.to_csv(outputs["graphs"], sep="\t", index=False)
    audits.to_csv(outputs["relations"], sep="\t", index=False)

    observed_audit = audits.loc[audits["analysis_arm"].eq("observed")]
    report = {
        "schema": "contrast_atlas_sparse_local_response_alignment_v1",
        "status": "DEVELOPMENT_CONSUMED_SPARSE_LOCAL_ALIGNMENT_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "fragment_bin_width_da": args.bin_width_da,
            "minimum_cosine": args.minimum_cosine,
            "neighbors_each_direction": NEIGHBORS,
            "graph_freeze": "source-pair and identity-world graphs frozen before beta, SE, contrasts or relations are opened",
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "directional_denominator": "complete common-policy source entity universe; absent edges contribute zero",
            "score": "symmetric mean of locally aligned response-weight products weighted by fragment cosine",
        },
        "denominators": {
            "studies": len(studies),
            "contrasts": len(contrasts),
            "response_rows": len(entities),
            "fragment_dimensions": dimensions,
            "source_pair_world_graphs": len(graphs),
            "frozen_directional_edges": len(edges),
            "fixed_candidate_rows_per_arm": len(observed_scores),
            "registered_directional_relations": len(relations),
        },
        "registered_relation_descriptive_ranks": {
            world: {
                "relations": int(len(block)),
                "similar_top1": int(block["similar_response_absolute_rank"].eq(1).sum()),
                "unsigned_top1": int(block["unsigned_response_absolute_rank"].eq(1).sum()),
                "composition_top1": int(block["composition_absolute_rank"].eq(1).sum()),
                "median_similar_rank": float(block["similar_response_absolute_rank"].median()),
                "median_unsigned_rank": float(block["unsigned_response_absolute_rank"].median()),
            }
            for world, block in observed_audit.groupby("evidence_world")
        },
        "truth_boundary": {
            "registered_signed_truth_relations": int(
                (~relations["expected_signed_direction"].str.lower().eq("unspecified")).sum()
            ),
            "unregistered_pairs": "unadjudicated, never negative",
            "pass_fail": "not adjudicated because no registered relation supplies signed metabolite-level truth or a frozen pass gate",
        },
        "sources": source_reports,
        "claim_boundary": [
            f"This is a complete {len(studies)}-study development-consumed run, not external validation.",
            "Registered condition-family relevance is not metabolite-level signed truth.",
            f"Top-1 counts are descriptive across {int(relations['relation_direction'].eq('registered').sum())} canonical registered relations with heterogeneous evidence qualification.",
            "Unresolved means not supported by the common strict library policy, not a unique unknown molecule.",
            "Feature rows may retain adduct and isotope redundancy.",
        ],
        "provenance": {
            "contract_sha256": sha256(args.contract),
            "contract_lineage": {str(path.relative_to(root)): sha256(path) for path in contract_paths},
            "response_entities_sha256": sha256(paths["entities"]),
            "contrast_manifest_sha256": sha256(paths["contrasts"]),
            "registered_relations_sha256": sha256(paths["relations"]),
            "identity_partition_sha256": sha256(args.identity_partition),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
