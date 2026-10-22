#!/usr/bin/env python3
"""Source-pair-frozen symmetric nearest-neighbour response alignment baseline.

This local-alignment baseline consumes only precomputed unit embeddings.  For
each source pair and identity world, bidirectional top-1 edges are frozen before
beta, standard errors, contrast labels, or registered relations are opened.
Response scores subsequently average over the complete directional source
entity universe, so absent edges and absent contrast response estimates are
zero contributions rather than deleted observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ad_cross_study_spectral_edges import directed_topk
from run_contrast_atlas_sparse_fragment_measure_baseline import WORLDS, sha256
from score_contrast_atlas_precomputed_embedding_measures import (
    load_embedding_archive,
    resolve_embedding_jobs,
)


NEIGHBORS = 1
MINIMUM_COSINE = 0.70
RANK_SCORES = (
    "similar_response_score",
    "opposite_response_score",
    "unsigned_response_score",
    "composition_score",
)


def frozen_source_universes(entities: pd.DataFrame) -> dict[str, pd.DataFrame]:
    identity = entities[["study_id", "entity_id", "identity_status"]].drop_duplicates()
    disagreement = identity.groupby(["study_id", "entity_id"])["identity_status"].nunique()
    if (disagreement > 1).any():
        raise ValueError("identity status changes across contrasts for an entity")
    if identity.duplicated(["study_id", "entity_id"]).any():
        raise ValueError("duplicate study/entity rows in frozen source universe")
    allowed = {"library_supported", "unresolved_entity"}
    if not set(identity["identity_status"]).issubset(allowed):
        raise ValueError("unsupported identity status in collection")
    return {
        str(study): block.sort_values("entity_id").reset_index(drop=True)
        for study, block in identity.groupby("study_id", sort=False)
    }


def world_ids(universe: pd.DataFrame, world: str) -> np.ndarray:
    if world == "all":
        selected = universe
    elif world == "known":
        selected = universe.loc[universe["identity_status"].eq("library_supported")]
    elif world == "unresolved":
        selected = universe.loc[universe["identity_status"].eq("unresolved_entity")]
    else:
        raise ValueError(world)
    return selected["entity_id"].astype(str).to_numpy()


def freeze_graphs(
    studies: list[str],
    universes: dict[str, pd.DataFrame],
    embeddings: dict[str, tuple[list[str], np.ndarray]],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], pd.DataFrame]:
    graphs: dict[tuple[str, str, str], dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []
    canonical_studies = sorted(studies)
    for left_index, left_study in enumerate(canonical_studies):
        for right_study in canonical_studies[left_index + 1 :]:
            left_embedding_ids, left_embedding = embeddings[left_study]
            right_embedding_ids, right_embedding = embeddings[right_study]
            left_lookup = {entity: row for row, entity in enumerate(left_embedding_ids)}
            right_lookup = {entity: row for row, entity in enumerate(right_embedding_ids)}
            for world in WORLDS:
                left_ids = world_ids(universes[left_study], world)
                right_ids = world_ids(universes[right_study], world)
                left_embedded_ids = np.asarray(
                    [entity for entity in left_ids if entity in left_lookup], dtype=str,
                )
                right_embedded_ids = np.asarray(
                    [entity for entity in right_ids if entity in right_lookup], dtype=str,
                )
                if len(left_embedded_ids) and len(right_embedded_ids):
                    left_vectors = left_embedding[
                        [left_lookup[entity] for entity in left_embedded_ids]
                    ]
                    right_vectors = right_embedding[
                        [right_lookup[entity] for entity in right_embedded_ids]
                    ]
                    forward = directed_topk(
                        left_embedded_ids,
                        left_vectors,
                        right_embedded_ids,
                        right_vectors,
                        neighbors=NEIGHBORS,
                        minimum_similarity=MINIMUM_COSINE,
                        reverse=False,
                    )
                    reverse = directed_topk(
                        right_embedded_ids,
                        right_vectors,
                        left_embedded_ids,
                        left_vectors,
                        neighbors=NEIGHBORS,
                        minimum_similarity=MINIMUM_COSINE,
                        reverse=True,
                    )
                else:
                    forward = pd.DataFrame(
                        columns=["left_entity_id", "right_entity_id", "chemical_similarity"]
                    )
                    reverse = forward.copy()
                forward_map = {
                    str(row.left_entity_id): (
                        str(row.right_entity_id),
                        float(row.chemical_similarity),
                    )
                    for row in forward.itertuples(index=False)
                }
                reverse_map = {
                    str(row.right_entity_id): (
                        str(row.left_entity_id),
                        float(row.chemical_similarity),
                    )
                    for row in reverse.itertuples(index=False)
                }
                if len(forward_map) != len(forward) or len(reverse_map) != len(reverse):
                    raise ValueError("top-1 edge table contains duplicate directional sources")
                graphs[(left_study, right_study, world)] = {
                    "left_ids": left_ids,
                    "right_ids": right_ids,
                    "left_to_right": forward_map,
                    "right_to_left": reverse_map,
                }
                for direction, frame in (
                    ("left_to_right", forward),
                    ("right_to_left", reverse),
                ):
                    for row in frame.itertuples(index=False):
                        edge_rows.append(
                            {
                                "left_study_id": left_study,
                                "right_study_id": right_study,
                                "evidence_world": world,
                                "direction": direction,
                                "left_entity_id": str(row.left_entity_id),
                                "right_entity_id": str(row.right_entity_id),
                                "chemical_similarity": float(row.chemical_similarity),
                            }
                        )
    return graphs, pd.DataFrame(edge_rows)


def response_profiles(entities: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    beta = entities["beta"].to_numpy(float)
    se = entities["se"].to_numpy(float)
    if not np.isfinite(np.column_stack([beta, se])).all() or (se < 0).any():
        raise ValueError("invalid beta or standard error")
    denominator = np.sqrt(beta * beta + se * se)
    values = np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)
    frame = entities[["study_id", "contrast_id", "entity_id"]].copy()
    frame["response_weight"] = values
    if frame.duplicated(["study_id", "contrast_id", "entity_id"]).any():
        raise ValueError("duplicate response entity within a contrast")
    return {
        (str(study), str(contrast)): dict(
            zip(block["entity_id"].astype(str), block["response_weight"].astype(float))
        )
        for (study, contrast), block in frame.groupby(
            ["study_id", "contrast_id"], sort=False
        )
    }


def deterministic_profile_reassignment(
    universes: dict[str, pd.DataFrame],
    observed: dict[tuple[str, str], dict[str, float]],
    contrasts: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, float]], pd.DataFrame]:
    mapping_rows: list[dict[str, str]] = []
    recipient_to_donor: dict[tuple[str, str], str] = {}
    for study, universe in universes.items():
        for identity, block in universe.groupby("identity_status", sort=True):
            entity_ids = sorted(block["entity_id"].astype(str))
            donors = entity_ids[1:] + entity_ids[:1] if len(entity_ids) > 1 else entity_ids
            for recipient, donor in zip(entity_ids, donors):
                recipient_to_donor[(study, recipient)] = donor
                mapping_rows.append(
                    {
                        "study_id": study,
                        "identity_status": str(identity),
                        "recipient_entity_id": recipient,
                        "donor_profile_entity_id": donor,
                        "fixed_point": str(recipient == donor),
                    }
                )
    reassigned: dict[tuple[str, str], dict[str, float]] = {}
    for row in contrasts.itertuples(index=False):
        key = (str(row.study_id), str(row.contrast_id))
        source = observed[key]
        reassigned[key] = {
            recipient: float(source.get(donor, 0.0))
            for (study, recipient), donor in recipient_to_donor.items()
            if study == key[0]
        }
    return reassigned, pd.DataFrame(mapping_rows)


def directional_components(
    source_ids: np.ndarray,
    edges: dict[str, tuple[str, float]],
    source_response: dict[str, float],
    target_response: dict[str, float],
) -> dict[str, float]:
    denominator = len(source_ids)
    if denominator == 0:
        return {
            "similar": float("nan"),
            "opposite": float("nan"),
            "unsigned": float("nan"),
            "composition": float("nan"),
            "edge_coverage": float("nan"),
            "response_pair_coverage": float("nan"),
        }
    similar = opposite = composition = 0.0
    response_pairs = 0
    for entity in source_ids:
        edge = edges.get(str(entity))
        if edge is None:
            continue
        target, cosine = edge
        composition += cosine
        left = float(source_response.get(str(entity), 0.0))
        right = float(target_response.get(str(target), 0.0))
        if left != 0 and right != 0:
            response_pairs += 1
        product = left * right * cosine
        similar += max(product, 0.0)
        opposite += max(-product, 0.0)
    return {
        "similar": similar / denominator,
        "opposite": opposite / denominator,
        "unsigned": (similar + opposite) / denominator,
        "composition": composition / denominator,
        "edge_coverage": len(edges) / denominator,
        "response_pair_coverage": response_pairs / denominator,
    }


def graph_for_orientation(
    graphs: dict[tuple[str, str, str], dict[str, object]],
    query_study: str,
    candidate_study: str,
    world: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[str, float]], dict[str, tuple[str, float]]]:
    left, right = sorted((query_study, candidate_study))
    graph = graphs[(left, right, world)]
    if query_study == left:
        return (
            graph["left_ids"],
            graph["right_ids"],
            graph["left_to_right"],
            graph["right_to_left"],
        )
    return (
        graph["right_ids"],
        graph["left_ids"],
        graph["right_to_left"],
        graph["left_to_right"],
    )


def add_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    for score in RANK_SCORES:
        rank = score.replace("_score", "_absolute_rank")
        scores[rank] = np.nan
        groups = scores.groupby(
            ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
        ).groups
        for indices in groups.values():
            values = scores.loc[indices, score]
            present = values.notna()
            scores.loc[values.index[present], rank] = values.loc[present].rank(
                ascending=False, method="min"
            )
    return scores


def score_all(
    contrasts: pd.DataFrame,
    graphs: dict[tuple[str, str, str], dict[str, object]],
    profiles: dict[tuple[str, str], dict[str, float]],
    analysis_arm: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query in contrasts.itertuples(index=False):
        for candidate in contrasts.itertuples(index=False):
            if str(query.source_family) == str(candidate.source_family):
                continue
            query_key = (str(query.study_id), str(query.contrast_id))
            candidate_key = (str(candidate.study_id), str(candidate.contrast_id))
            for world in WORLDS:
                qids, cids, q_to_c, c_to_q = graph_for_orientation(
                    graphs, query_key[0], candidate_key[0], world
                )
                forward = directional_components(
                    qids, q_to_c, profiles[query_key], profiles[candidate_key]
                )
                reverse = directional_components(
                    cids, c_to_q, profiles[candidate_key], profiles[query_key]
                )
                available = len(qids) > 0 and len(cids) > 0
                row: dict[str, object] = {
                    "analysis_arm": analysis_arm,
                    "query_study_id": query_key[0],
                    "query_source_family": str(query.source_family),
                    "query_contrast_id": query_key[1],
                    "candidate_study_id": candidate_key[0],
                    "candidate_source_family": str(candidate.source_family),
                    "candidate_contrast_id": candidate_key[1],
                    "evidence_world": world,
                    "query_universe_entities": len(qids),
                    "candidate_universe_entities": len(cids),
                    "query_to_candidate_edge_coverage": forward["edge_coverage"],
                    "candidate_to_query_edge_coverage": reverse["edge_coverage"],
                    "query_to_candidate_response_pair_coverage": forward[
                        "response_pair_coverage"
                    ],
                    "candidate_to_query_response_pair_coverage": reverse[
                        "response_pair_coverage"
                    ],
                    "score_available": available,
                    "adjudication_status": "unadjudicated",
                }
                if available:
                    similar = 0.5 * (forward["similar"] + reverse["similar"])
                    opposite = 0.5 * (forward["opposite"] + reverse["opposite"])
                    row.update(
                        {
                            "similar_response_score": similar,
                            "opposite_response_score": opposite,
                            "signed_response_score": similar - opposite,
                            "unsigned_response_score": 0.5
                            * (forward["unsigned"] + reverse["unsigned"]),
                            "composition_score": 0.5
                            * (forward["composition"] + reverse["composition"]),
                        }
                    )
                else:
                    for column in (*RANK_SCORES, "signed_response_score"):
                        row[column] = np.nan
                rows.append(row)
    return add_ranks(pd.DataFrame(rows))


def relation_audit(scores: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for relation in relations.to_dict("records"):
        mask = (
            scores["query_study_id"].eq(str(relation["left_study_id"]))
            & scores["query_contrast_id"].eq(str(relation["left_contrast_id"]))
            & scores["candidate_study_id"].eq(str(relation["right_study_id"]))
            & scores["candidate_contrast_id"].eq(str(relation["right_contrast_id"]))
        )
        matched = scores.loc[mask]
        if len(matched) != len(WORLDS):
            raise ValueError("registered relation does not match exactly three worlds")
        for row in matched.to_dict("records"):
            result = dict(relation)
            result.update(
                {
                    "analysis_arm": row["analysis_arm"],
                    "evidence_world": row["evidence_world"],
                    "candidate_universe_size": int(
                        len(
                            scores.loc[
                                scores["query_study_id"].eq(row["query_study_id"])
                                & scores["query_contrast_id"].eq(row["query_contrast_id"])
                                & scores["evidence_world"].eq(row["evidence_world"])
                            ]
                        )
                    ),
                    "adjudication_status": (
                        "DESCRIPTIVE_DESIGN_RELEVANCE_ONLY_SIGNED_DIRECTION_UNSPECIFIED"
                    ),
                    "pass_fail": "NOT_ADJUDICATED",
                }
            )
            for column in (*RANK_SCORES, "signed_response_score"):
                result[column] = row[column]
            for score in RANK_SCORES:
                rank = score.replace("_score", "_absolute_rank")
                result[rank] = row[rank]
            output.append(result)
    return pd.DataFrame(output)


def graph_summary(
    graphs: dict[tuple[str, str, str], dict[str, object]]
) -> pd.DataFrame:
    rows = []
    for (left, right, world), graph in graphs.items():
        left_n, right_n = len(graph["left_ids"]), len(graph["right_ids"])
        forward_n = len(graph["left_to_right"])
        reverse_n = len(graph["right_to_left"])
        rows.append(
            {
                "left_study_id": left,
                "right_study_id": right,
                "evidence_world": world,
                "left_universe_entities": left_n,
                "right_universe_entities": right_n,
                "left_to_right_edges": forward_n,
                "right_to_left_edges": reverse_n,
                "left_to_right_coverage": forward_n / left_n if left_n else np.nan,
                "right_to_left_coverage": reverse_n / right_n if right_n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embedding-manifest", type=Path)
    source.add_argument("--embeddings-root", type=Path)
    parser.add_argument("--method-id")
    parser.add_argument(
        "--identity-partition",
        type=Path,
        help=(
            "Optional complete phenotype-blind common-policy partition. When supplied, "
            "it replaces source-native identity labels before any graph is built."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--unit-norm-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "sources": args.collection / "source_manifest.tsv",
        "relations": args.collection / "registered_relations.tsv",
        "report": args.collection / "report.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # The graph phase reads only the phenotype-blind entity registry and source
    # IDs.  Effect and contrast columns are opened only after every graph is
    # frozen.
    registry = pd.read_csv(
        paths["entities"],
        sep="\t",
        usecols=["study_id", "entity_id", "identity_status"],
        dtype=str,
    ).drop_duplicates()
    identity_partition_report_path: Path | None = None
    identity_partition_provenance: dict[str, str] = {}
    if args.identity_partition is not None:
        identity_partition_report_path = args.identity_partition.parent / "report.json"
        if not args.identity_partition.is_file() or not identity_partition_report_path.is_file():
            raise FileNotFoundError(args.identity_partition)
        identity_report = json.loads(identity_partition_report_path.read_text(encoding="utf-8"))
        if identity_report.get("status") != "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITIONS_COMPLETE":
            raise RuntimeError("identity partition is not a complete common-policy artifact")
        expected_partition_sha = identity_report.get("provenance", {}).get(
            "combined_partition_sha256"
        )
        if expected_partition_sha != sha256(args.identity_partition):
            raise RuntimeError("common identity partition checksum drift")
        identity = pd.read_csv(
            args.identity_partition,
            sep="\t",
            usecols=["study_id", "entity_id", "identity_status"],
            dtype=str,
        )
        if identity.duplicated(["study_id", "entity_id"]).any():
            raise RuntimeError("common identity partition contains duplicate entity keys")
        base_keys = registry[["study_id", "entity_id"]].drop_duplicates()
        if len(identity) != len(base_keys) or set(map(tuple, identity[["study_id", "entity_id"]].to_numpy())) != set(
            map(tuple, base_keys[["study_id", "entity_id"]].to_numpy())
        ):
            raise RuntimeError("common identity partition does not match collection entities")
        registry = base_keys.merge(
            identity, on=["study_id", "entity_id"], how="left", validate="one_to_one"
        )
        identity_partition_provenance = {
            str(args.identity_partition): sha256(args.identity_partition),
            str(identity_partition_report_path): sha256(identity_partition_report_path),
        }
    sources = pd.read_csv(paths["sources"], sep="\t", dtype={"study_id": str})
    studies = list(sources["study_id"].astype(str).drop_duplicates())
    jobs = resolve_embedding_jobs(args, studies)
    method_id = str(jobs["method_id"].iloc[0])
    universes = frozen_source_universes(registry)

    embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    embedding_reports: list[dict[str, object]] = []
    dimensions: set[int] = set()
    for job in jobs.to_dict("records"):
        study = str(job["study_id"])
        required = set(universes[study]["entity_id"].astype(str))
        ids, vectors, report = load_embedding_archive(
            Path(job["embedding_path"]), required, args.unit_norm_tolerance
        )
        dimensions.add(vectors.shape[1])
        embeddings[study] = (ids, vectors)
        embedding_reports.append({"study_id": study, "method_id": method_id, **report})
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions differ across studies")

    # Freeze all source-pair graphs before opening beta/SE, contrast definitions,
    # or registered truth.
    graphs, edges = freeze_graphs(studies, universes, embeddings)
    summary = graph_summary(graphs)

    entities = pd.read_csv(paths["entities"], sep="\t", dtype={"entity_id": str})
    contrasts = pd.read_csv(paths["contrasts"], sep="\t").reset_index(drop=True)
    if set(contrasts["study_id"].astype(str)) != set(studies):
        raise ValueError("contrast and source registries contain different studies")
    observed_profiles = response_profiles(entities)
    reassigned_profiles, reassignment = deterministic_profile_reassignment(
        universes, observed_profiles, contrasts
    )
    observed_scores = score_all(contrasts, graphs, observed_profiles, "observed")
    sensitivity_scores = score_all(
        contrasts,
        graphs,
        reassigned_profiles,
        "deterministic_within_study_identity_stratum_profile_reassignment",
    )

    args.output_dir.mkdir(parents=True)
    output_paths = {
        "scores": args.output_dir / "fixed_candidate_scores.tsv.gz",
        "sensitivity": args.output_dir
        / "deterministic_profile_reassignment_scores.tsv.gz",
        "mapping": args.output_dir / "deterministic_profile_reassignment.tsv.gz",
        "edges": args.output_dir / "source_pair_frozen_edges.tsv.gz",
        "graph_summary": args.output_dir / "source_pair_frozen_graph_summary.tsv",
        "relations": args.output_dir / "registered_relation_absolute_ranks.tsv",
    }
    compression = {"method": "gzip", "mtime": 0}
    observed_scores.to_csv(output_paths["scores"], sep="\t", index=False, compression=compression)
    sensitivity_scores.to_csv(
        output_paths["sensitivity"], sep="\t", index=False, compression=compression
    )
    reassignment.to_csv(output_paths["mapping"], sep="\t", index=False, compression=compression)
    edges.to_csv(output_paths["edges"], sep="\t", index=False, compression=compression)
    summary.to_csv(output_paths["graph_summary"], sep="\t", index=False)

    # Registered relations are opened only after both score arms are complete.
    relations = pd.read_csv(paths["relations"], sep="\t", dtype=str)
    audits = pd.concat(
        [relation_audit(observed_scores, relations), relation_audit(sensitivity_scores, relations)],
        ignore_index=True,
    )
    audits.to_csv(output_paths["relations"], sep="\t", index=False)

    report = {
        "schema": "contrast_atlas_symmetric_nn_response_alignment_v1",
        "status": "DEVELOPMENT_CONSUMED_LOCAL_ALIGNMENT_BASELINE_COMPLETE",
        "data_role": "development_consumed",
        "method_id": method_id,
        "parameters": {
            "neighbors_each_direction": NEIGHBORS,
            "minimum_cosine": MINIMUM_COSINE,
            "parameter_selection": "fixed a priori; not tuned on response or truth",
            "graph_freeze": "per source pair and all/known/unresolved world before beta, SE, contrast definitions, or truth are loaded",
            "directional_denominator": "complete frozen source entity universe in that identity world",
            "zero_contribution_rule": "no edge or no contrast response contributes zero and remains in denominator",
            "symmetric_score": "mean of query-to-candidate and candidate-to-query directional means",
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "similar": "mean(max(weight_product,0)*cosine)",
            "opposite": "mean(max(-weight_product,0)*cosine)",
            "unsigned": "similar + opposite",
            "composition": "coverage-weighted mean edge cosine without response",
            "identity_policy": (
                "complete common GNPS Gold/Silver strict partition"
                if args.identity_partition is not None
                else "source-native development partitions"
            ),
        },
        "denominators": {
            "independent_studies": int(contrasts["study_id"].nunique()),
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": len(contrasts),
            "biological_units_across_sources": int(sources["biological_units"].sum()),
            "eligible_response_rows": len(entities),
            "source_pair_world_graphs": len(graphs),
            "frozen_directional_edges": len(edges),
            "fixed_candidate_rows_per_arm": len(observed_scores),
            "observed_score_available_rows": int(observed_scores["score_available"].sum()),
            "sensitivity_score_available_rows": int(sensitivity_scores["score_available"].sum()),
            "reassignment_entities": len(reassignment),
            "reassignment_fixed_points": int(reassignment["fixed_point"].eq("True").sum()),
            "registered_directional_relations": len(relations),
            "registered_signed_direction_truth_relations": int(
                (~relations["expected_signed_direction"].str.lower().eq("unspecified")).sum()
            ),
            "embedding_dimensions": next(iter(dimensions)),
        },
        "sensitivity_boundary": (
            "Deterministic cyclic response-profile reassignment within study and identity stratum; "
            "one entity mapping is shared across every contrast. This is a non-calibrated sensitivity, "
            "not a randomization null or p-value."
        ),
        "truth_boundary": {
            "registered_relations_opened": "after observed and reassigned fixed-candidate scores were written",
            "unregistered_pairs": "unadjudicated, never negative",
            "unspecified_signed_direction": "descriptive rank only; no PASS/FAIL",
        },
        "embedding_sources": embedding_reports,
        "input_sha256": {
            **{str(path): sha256(path) for path in paths.values()},
            **identity_partition_provenance,
        },
        "claim_boundary": [
            "This is a local spectral-alignment baseline, not the final Contrast Atlas method.",
            "No ion-family collapse or measurement/covariate matching is performed.",
            (
                "The common identity policy removes cross-source partition drift, but absence of "
                "ion-family collapse and measurement-matched known controls still prevents a "
                "formal unknown-information increment claim."
                if args.identity_partition is not None
                else "Source-native identity policies differ; known/all comparisons cannot support "
                "an unknown-information increment claim."
            ),
            "Unresolved means not library-supported, not a unique metabolite.",
            "All collection sources are development/consumed.",
            "The deterministic reassignment is a sensitivity stress, not a calibrated null.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checksum_paths = {**output_paths, "report": report_path}
    (args.output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in checksum_paths.values()
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
