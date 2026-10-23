#!/usr/bin/env python3
"""Calibrate local Contrast Atlas response alignment against profile reassignment.

The sparse chemical graph and the complete query/candidate universe are consumed
from a finished alignment run.  Registered biological relations are not opened
until observed and null-calibrated scores have been frozen.  The null reassigns
whole response profiles among entities within each study and common identity
stratum, preserving study design, marginal response distributions, missingness
patterns, and the phenotype-blind chemical graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_sparse_fragment_measure_baseline import sha256


METRICS = ("similar", "opposite", "unsigned")
WORLDS = ("all", "known", "unresolved")


def response_weight(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    denominator = np.sqrt(beta * beta + se * se)
    return np.divide(
        beta,
        denominator,
        out=np.zeros_like(beta, dtype=np.float64),
        where=denominator > 0,
    )


def build_profile_matrices(
    entities: pd.DataFrame,
    contrasts: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, np.ndarray],
]:
    """Return study matrices plus stable entity/contrast/status registries."""
    required = {"study_id", "contrast_id", "entity_id", "beta", "se"}
    if not required.issubset(entities.columns):
        raise ValueError(f"response table lacks {sorted(required - set(entities.columns))}")
    if entities.duplicated(["study_id", "contrast_id", "entity_id"]).any():
        raise ValueError("duplicate study/contrast/entity response rows")
    identity = identity[["study_id", "entity_id", "identity_status"]].copy()
    if identity.duplicated(["study_id", "entity_id"]).any():
        raise ValueError("duplicate common identity rows")
    allowed = {"library_supported", "unresolved_entity"}
    if not set(identity["identity_status"]).issubset(allowed):
        raise ValueError("unsupported common identity status")

    entity_ids: dict[str, list[str]] = {}
    contrast_ids: dict[str, list[str]] = {}
    statuses: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    for study in sorted(contrasts["study_id"].astype(str).unique()):
        study_identity = identity.loc[identity["study_id"].astype(str).eq(study)].copy()
        study_identity["entity_id"] = study_identity["entity_id"].astype(str)
        study_identity = study_identity.sort_values("entity_id").reset_index(drop=True)
        ids = study_identity["entity_id"].tolist()
        cids = contrasts.loc[
            contrasts["study_id"].astype(str).eq(study), "contrast_id"
        ].astype(str).tolist()
        if not ids or not cids:
            raise ValueError(f"empty entity or contrast registry for {study}")
        row = {value: index for index, value in enumerate(ids)}
        column = {value: index for index, value in enumerate(cids)}
        matrix = np.zeros((len(ids), len(cids)), dtype=np.float64)
        block = entities.loc[entities["study_id"].astype(str).eq(study)].copy()
        missing_entities = set(block["entity_id"].astype(str)) - set(ids)
        missing_contrasts = set(block["contrast_id"].astype(str)) - set(cids)
        if missing_entities or missing_contrasts:
            raise ValueError(f"response registry drift for {study}")
        weights = response_weight(block["beta"].to_numpy(float), block["se"].to_numpy(float))
        for entity, contrast, value in zip(
            block["entity_id"].astype(str),
            block["contrast_id"].astype(str),
            weights,
        ):
            matrix[row[entity], column[contrast]] = float(value)
        entity_ids[study] = ids
        contrast_ids[study] = cids
        statuses[study] = study_identity["identity_status"].astype(str).to_numpy()
        matrices[study] = matrix
    return matrices, entity_ids, contrast_ids, statuses


def permutation_indices(
    statuses: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    """Reassign whole entity profiles within study and identity stratum."""
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for study in sorted(statuses):
        status = statuses[study]
        donors = np.arange(len(status), dtype=np.int64)
        for stratum in sorted(set(status)):
            positions = np.flatnonzero(status == stratum)
            donors[positions] = rng.permutation(positions)
        output[study] = donors
    return output


def directional_score_matrices(
    source_profiles: np.ndarray,
    target_profiles: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    cosine: np.ndarray,
    denominator: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized positive, negative and absolute products for all contrasts."""
    shape = (source_profiles.shape[1], target_profiles.shape[1])
    if denominator == 0:
        nan = np.full(shape, np.nan, dtype=np.float64)
        return nan.copy(), nan.copy(), nan.copy()
    if not len(source_rows):
        zero = np.zeros(shape, dtype=np.float64)
        return zero.copy(), zero.copy(), zero.copy()
    left = source_profiles[source_rows]
    right = target_profiles[target_rows]
    weight = cosine[:, None]
    left_pos = np.clip(left, 0.0, None)
    left_neg = np.clip(-left, 0.0, None)
    right_pos = np.clip(right, 0.0, None)
    right_neg = np.clip(-right, 0.0, None)
    similar = left_pos.T @ (weight * right_pos) + left_neg.T @ (weight * right_neg)
    opposite = left_pos.T @ (weight * right_neg) + left_neg.T @ (weight * right_pos)
    similar /= denominator
    opposite /= denominator
    return similar, opposite, similar + opposite


def directional_response_energy_matrices(
    source_profiles: np.ndarray,
    target_profiles: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    cosine: np.ndarray,
    source_universe_rows: np.ndarray,
    target_universe_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Response-measure kernel normalized by complete-world response energy.

    The numerator is supported only by the phenotype-blind frozen chemical
    edges.  The denominator contains the complete response energy in the
    selected identity world, so missing chemical support is not silently
    discarded.  Whole-profile permutation preserves these denominators.
    """
    shape = (source_profiles.shape[1], target_profiles.shape[1])
    if not len(source_universe_rows) or not len(target_universe_rows):
        nan = np.full(shape, np.nan, dtype=np.float64)
        return nan.copy(), nan.copy(), nan.copy()
    source_energy = np.sum(
        source_profiles[source_universe_rows] * source_profiles[source_universe_rows],
        axis=0,
    )
    target_energy = np.sum(
        target_profiles[target_universe_rows] * target_profiles[target_universe_rows],
        axis=0,
    )
    denominator = np.sqrt(source_energy[:, None] * target_energy[None, :])
    if not len(source_rows):
        zero = np.zeros(shape, dtype=np.float64)
        return zero.copy(), zero.copy(), zero.copy()
    left = source_profiles[source_rows]
    right = target_profiles[target_rows]
    weight = cosine[:, None]
    left_pos = np.clip(left, 0.0, None)
    left_neg = np.clip(-left, 0.0, None)
    right_pos = np.clip(right, 0.0, None)
    right_neg = np.clip(-right, 0.0, None)
    similar_numerator = (
        left_pos.T @ (weight * right_pos) + left_neg.T @ (weight * right_neg)
    )
    opposite_numerator = (
        left_pos.T @ (weight * right_neg) + left_neg.T @ (weight * right_pos)
    )
    similar = np.divide(
        similar_numerator,
        denominator,
        out=np.zeros(shape, dtype=np.float64),
        where=denominator > 0,
    )
    opposite = np.divide(
        opposite_numerator,
        denominator,
        out=np.zeros(shape, dtype=np.float64),
        where=denominator > 0,
    )
    return similar, opposite, similar + opposite


def prepare_graphs(
    edges: pd.DataFrame,
    entity_ids: dict[str, list[str]],
    statuses: dict[str, np.ndarray],
    expected_keys: set[tuple[str, str, str]] | None = None,
) -> dict[tuple[str, str, str], dict[str, object]]:
    lookups = {
        study: {entity: index for index, entity in enumerate(ids)}
        for study, ids in entity_ids.items()
    }
    graphs: dict[tuple[str, str, str], dict[str, object]] = {}
    observed_keys = set(
        map(
            tuple,
            edges[["left_study_id", "right_study_id", "evidence_world"]]
            .astype(str)
            .drop_duplicates()
            .to_numpy(),
        )
    )
    keys = observed_keys if expected_keys is None else expected_keys
    if not observed_keys.issubset(keys):
        raise ValueError("edge table includes an unexpected graph key")
    for left, right, world in sorted(keys):
        block = edges.loc[
            edges["left_study_id"].astype(str).eq(left)
            & edges["right_study_id"].astype(str).eq(right)
            & edges["evidence_world"].astype(str).eq(world)
        ]
        if world == "all":
            left_universe_rows = np.arange(len(statuses[left]), dtype=np.int64)
            right_universe_rows = np.arange(len(statuses[right]), dtype=np.int64)
            left_denominator = len(statuses[left])
            right_denominator = len(statuses[right])
        elif world == "known":
            left_universe_rows = np.flatnonzero(statuses[left] == "library_supported")
            right_universe_rows = np.flatnonzero(statuses[right] == "library_supported")
            left_denominator = int((statuses[left] == "library_supported").sum())
            right_denominator = int((statuses[right] == "library_supported").sum())
        elif world == "unresolved":
            left_universe_rows = np.flatnonzero(statuses[left] == "unresolved_entity")
            right_universe_rows = np.flatnonzero(statuses[right] == "unresolved_entity")
            left_denominator = int((statuses[left] == "unresolved_entity").sum())
            right_denominator = int((statuses[right] == "unresolved_entity").sum())
        else:
            raise ValueError(world)
        forward = block.loc[block["direction"].eq("left_to_right")]
        reverse = block.loc[block["direction"].eq("right_to_left")]
        graphs[(left, right, world)] = {
            "forward_source": np.asarray(
                [lookups[left][value] for value in forward["left_entity_id"].astype(str)],
                dtype=np.int64,
            ),
            "forward_target": np.asarray(
                [lookups[right][value] for value in forward["right_entity_id"].astype(str)],
                dtype=np.int64,
            ),
            "forward_cosine": forward["chemical_similarity"].to_numpy(float),
            "forward_denominator": left_denominator,
            "left_universe_rows": left_universe_rows,
            "right_universe_rows": right_universe_rows,
            "reverse_source": np.asarray(
                [lookups[right][value] for value in reverse["right_entity_id"].astype(str)],
                dtype=np.int64,
            ),
            "reverse_target": np.asarray(
                [lookups[left][value] for value in reverse["left_entity_id"].astype(str)],
                dtype=np.int64,
            ),
            "reverse_cosine": reverse["chemical_similarity"].to_numpy(float),
            "reverse_denominator": right_denominator,
        }
    return graphs


def score_pair(
    left_profiles: np.ndarray,
    right_profiles: np.ndarray,
    graph: dict[str, object],
    score_mode: str = "entity_mean",
) -> dict[str, np.ndarray]:
    if score_mode == "entity_mean":
        forward = directional_score_matrices(
            left_profiles, right_profiles, graph["forward_source"],
            graph["forward_target"], graph["forward_cosine"],
            int(graph["forward_denominator"]),
        )
        reverse = directional_score_matrices(
            right_profiles, left_profiles, graph["reverse_source"],
            graph["reverse_target"], graph["reverse_cosine"],
            int(graph["reverse_denominator"]),
        )
    elif score_mode == "response_energy":
        forward = directional_response_energy_matrices(
            left_profiles, right_profiles, graph["forward_source"],
            graph["forward_target"], graph["forward_cosine"],
            graph["left_universe_rows"], graph["right_universe_rows"],
        )
        reverse = directional_response_energy_matrices(
            right_profiles, left_profiles, graph["reverse_source"],
            graph["reverse_target"], graph["reverse_cosine"],
            graph["right_universe_rows"], graph["left_universe_rows"],
        )
    else:
        raise ValueError(f"unsupported score mode: {score_mode}")
    return {
        metric: 0.5 * (forward[index] + reverse[index].T)
        for index, metric in enumerate(METRICS)
    }


def candidate_coordinates(
    scores: pd.DataFrame,
    contrast_ids: dict[str, list[str]],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    contrast_lookup = {
        study: {contrast: index for index, contrast in enumerate(values)}
        for study, values in contrast_ids.items()
    }
    output = {}
    canonical = scores.assign(
        pair=scores.apply(
            lambda row: tuple(sorted((str(row.query_study_id), str(row.candidate_study_id)))),
            axis=1,
        )
    )
    for (pair, world), block in canonical.groupby(["pair", "evidence_world"], sort=False):
        left, right = pair
        rows = block.index.to_numpy(np.int64)
        left_columns = []
        right_columns = []
        for row in block.itertuples(index=False):
            if str(row.query_study_id) == left:
                left_columns.append(contrast_lookup[left][str(row.query_contrast_id)])
                right_columns.append(contrast_lookup[right][str(row.candidate_contrast_id)])
            else:
                left_columns.append(contrast_lookup[left][str(row.candidate_contrast_id)])
                right_columns.append(contrast_lookup[right][str(row.query_contrast_id)])
        output[(left, right, str(world))] = (
            rows,
            np.asarray(left_columns, dtype=np.int64),
            np.asarray(right_columns, dtype=np.int64),
        )
    return output


def score_candidate_rows(
    profiles: dict[str, np.ndarray],
    graphs: dict[tuple[str, str, str], dict[str, object]],
    coordinates: dict[
        tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    row_count: int,
    score_mode: str = "entity_mean",
) -> dict[str, np.ndarray]:
    values = {
        metric: np.full(row_count, np.nan, dtype=np.float64) for metric in METRICS
    }
    for key, graph in graphs.items():
        left, right, _world = key
        pair_scores = score_pair(
            profiles[left], profiles[right], graph, score_mode=score_mode
        )
        rows, left_columns, right_columns = coordinates[key]
        for metric in METRICS:
            values[metric][rows] = pair_scores[metric][left_columns, right_columns]
    return values


def connected_relation_components(relations: pd.DataFrame) -> int:
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in relations.itertuples(index=False):
        union(
            (str(row.left_study_id), str(row.left_contrast_id)),
            (str(row.right_study_id), str(row.right_contrast_id)),
        )
    return len({find(node) for node in parent})


def add_calibrated_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    groups = frame.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
    ).groups
    for metric in METRICS:
        column = f"{metric}_response_null_z"
        rank = f"{metric}_response_null_z_absolute_rank"
        frame[rank] = np.nan
        for indices in groups.values():
            values = frame.loc[indices, column]
            present = values.notna()
            frame.loc[values.index[present], rank] = values.loc[present].rank(
                ascending=False, method="min"
            )
    return frame


def relation_audit(scores: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for relation in relations.to_dict("records"):
        mask = (
            scores["query_study_id"].eq(str(relation["left_study_id"]))
            & scores["query_contrast_id"].eq(str(relation["left_contrast_id"]))
            & scores["candidate_study_id"].eq(str(relation["right_study_id"]))
            & scores["candidate_contrast_id"].eq(str(relation["right_contrast_id"]))
        )
        matched = scores.loc[mask]
        if len(matched) != len(WORLDS):
            raise ValueError("registered relation does not match all three worlds")
        for row in matched.to_dict("records"):
            result = dict(relation)
            result.update({
                "candidate_universe_size": int(
                    len(scores.loc[
                        scores["query_study_id"].eq(row["query_study_id"])
                        & scores["query_contrast_id"].eq(row["query_contrast_id"])
                        & scores["evidence_world"].eq(row["evidence_world"])
                    ])
                ),
                "adjudication_status": (
                    "DESCRIPTIVE_DESIGN_RELEVANCE_ONLY_SIGNED_DIRECTION_UNSPECIFIED"
                ),
                "pass_fail": "NOT_ADJUDICATED",
            })
            keep = [
                column for column in scores.columns
                if column.endswith("_response_score")
                or "response_null" in column
                or column in {"evidence_world", "composition_score", "composition_absolute_rank"}
            ]
            result.update({column: row[column] for column in keep})
            rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--score-mode",
        choices=("entity_mean", "response_energy"),
        default="entity_mean",
        help="entity_mean reproduces the frozen v1 score; response_energy uses a complete-world response-measure kernel",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.permutations < 2:
        raise ValueError("at least two null reassignments are required")

    collection_paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "relations": args.collection / "registered_relations.tsv",
    }
    alignment_paths = {
        "scores": args.alignment_dir / "fixed_candidate_scores.tsv.gz",
        "edges": args.alignment_dir / "source_pair_frozen_edges.tsv.gz",
        "report": args.alignment_dir / "report.json",
    }
    for path in (*collection_paths.values(), *alignment_paths.values(), args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)
    alignment_report = json.loads(alignment_paths["report"].read_text(encoding="utf-8"))
    if alignment_report.get("status") != "DEVELOPMENT_CONSUMED_SPARSE_LOCAL_ALIGNMENT_COMPLETE":
        raise RuntimeError("input alignment is not complete")
    provenance = alignment_report.get("provenance", {})
    if provenance.get("scores") != sha256(alignment_paths["scores"]):
        raise RuntimeError("alignment score checksum drift")
    if provenance.get("edges") != sha256(alignment_paths["edges"]):
        raise RuntimeError("alignment edge checksum drift")
    if provenance.get("response_entities_sha256") != sha256(collection_paths["entities"]):
        raise RuntimeError("response table differs from alignment input")
    if provenance.get("contrast_manifest_sha256") != sha256(collection_paths["contrasts"]):
        raise RuntimeError("contrast manifest differs from alignment input")
    if provenance.get("identity_partition_sha256") != sha256(args.identity_partition):
        raise RuntimeError("identity partition differs from alignment input")

    # Relations are deliberately not read until all null scores are frozen.
    entities = pd.read_csv(collection_paths["entities"], sep="\t", low_memory=False)
    contrasts = pd.read_csv(collection_paths["contrasts"], sep="\t", dtype=str)
    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    observed = pd.read_csv(alignment_paths["scores"], sep="\t")
    edges = pd.read_csv(alignment_paths["edges"], sep="\t", dtype={
        "left_study_id": str,
        "right_study_id": str,
        "left_entity_id": str,
        "right_entity_id": str,
    })
    matrices, entity_ids, contrast_ids, statuses = build_profile_matrices(
        entities, contrasts, identity
    )
    coordinates = candidate_coordinates(observed, contrast_ids)
    graphs = prepare_graphs(edges, entity_ids, statuses, set(coordinates))
    if set(coordinates) != set(graphs):
        missing = set(coordinates).symmetric_difference(graphs)
        raise RuntimeError(f"candidate/graph key mismatch: {sorted(missing)}")

    generator = np.random.default_rng(args.seed)
    seeds = generator.integers(0, np.iinfo(np.int64).max, size=args.permutations, dtype=np.int64)
    n = len(observed)
    totals = {metric: np.zeros(n, dtype=np.float64) for metric in METRICS}
    squares = {metric: np.zeros(n, dtype=np.float64) for metric in METRICS}
    exceed = {metric: np.zeros(n, dtype=np.int64) for metric in METRICS}
    observed_values = {
        metric: observed[f"{metric}_response_score"].to_numpy(float)
        for metric in METRICS
    }
    recomputed_observed = score_candidate_rows(
        matrices, graphs, coordinates, row_count=n, score_mode=args.score_mode
    )
    if args.score_mode == "entity_mean":
        for metric in METRICS:
            if not np.allclose(
                recomputed_observed[metric], observed_values[metric], rtol=1e-10, atol=1e-12
            ):
                maximum = float(
                    np.max(np.abs(recomputed_observed[metric] - observed_values[metric]))
                )
                raise RuntimeError(
                    f"vectorized {metric} scorer does not reproduce frozen scores; max={maximum}"
                )
    else:
        observed_values = recomputed_observed
        for metric in METRICS:
            observed[f"{metric}_response_score"] = observed_values[metric]
        observed["signed_response_score"] = (
            observed_values["similar"] - observed_values["opposite"]
        )
        rank_groups = observed.groupby(
            ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
        ).groups
        for metric in METRICS:
            rank_column = f"{metric}_response_absolute_rank"
            observed[rank_column] = np.nan
            for indices in rank_groups.values():
                values = observed.loc[indices, f"{metric}_response_score"]
                present = values.notna()
                observed.loc[values.index[present], rank_column] = values.loc[present].rank(
                    ascending=False, method="min"
                )
    for seed in seeds:
        donors = permutation_indices(statuses, int(seed))
        permuted = {study: matrix[donors[study]] for study, matrix in matrices.items()}
        values = score_candidate_rows(
            permuted, graphs, coordinates, row_count=n, score_mode=args.score_mode
        )
        for metric in METRICS:
            if not np.isfinite(values[metric]).all():
                raise RuntimeError(f"non-finite null {metric} scores")
            totals[metric] += values[metric]
            squares[metric] += values[metric] * values[metric]
            exceed[metric] += values[metric] >= observed_values[metric] - 1e-15

    calibrated = observed.copy()
    for metric in METRICS:
        mean = totals[metric] / args.permutations
        variance = np.maximum(
            (squares[metric] - args.permutations * mean * mean) / (args.permutations - 1),
            0.0,
        )
        sd = np.sqrt(variance)
        difference = observed_values[metric] - mean
        z = np.divide(
            difference,
            sd,
            out=np.zeros_like(difference),
            where=sd > 1e-15,
        )
        z[(sd <= 1e-15) & (difference > 1e-15)] = np.inf
        z[(sd <= 1e-15) & (difference < -1e-15)] = -np.inf
        calibrated[f"{metric}_response_null_mean"] = mean
        calibrated[f"{metric}_response_null_sd"] = sd
        calibrated[f"{metric}_response_null_z"] = z
        calibrated[f"{metric}_response_null_empirical_p"] = (
            1.0 + exceed[metric]
        ) / (args.permutations + 1.0)
    calibrated = add_calibrated_ranks(calibrated)

    args.output_dir.mkdir(parents=True)
    outputs = {
        "scores": args.output_dir / "response_null_calibrated_candidate_scores.tsv.gz",
        "seeds": args.output_dir / "permutation_seeds.tsv",
        "relations": args.output_dir / "registered_relation_calibrated_ranks.tsv",
    }
    compression = {"method": "gzip", "mtime": 0}
    calibrated.to_csv(outputs["scores"], sep="\t", index=False, compression=compression)
    pd.DataFrame({"permutation_index": np.arange(args.permutations), "seed": seeds}).to_csv(
        outputs["seeds"], sep="\t", index=False
    )

    # Only now is the registered design-relevance table opened.
    relations = pd.read_csv(collection_paths["relations"], sep="\t", dtype=str)
    relation_components = connected_relation_components(relations)
    audit = relation_audit(calibrated, relations)
    audit.to_csv(outputs["relations"], sep="\t", index=False)
    canonical = audit.loc[audit["relation_direction"].eq("registered")]
    summary = {}
    for world, block in canonical.groupby("evidence_world"):
        summary[str(world)] = {
            "canonical_relations": int(len(block)),
            "similar_z_top1": int(block["similar_response_null_z_absolute_rank"].eq(1).sum()),
            "unsigned_z_top1": int(block["unsigned_response_null_z_absolute_rank"].eq(1).sum()),
            "median_similar_z_rank": float(block["similar_response_null_z_absolute_rank"].median()),
            "median_unsigned_z_rank": float(block["unsigned_response_null_z_absolute_rank"].median()),
            "empirical_p_at_minimum_resolution": int(
                block["unsigned_response_null_empirical_p"].le(1 / (args.permutations + 1)).sum()
            ),
        }
    stratum_summary = {}
    for (relation_class, world), block in canonical.groupby(
        ["relation_class", "evidence_world"], sort=True
    ):
        stratum_summary.setdefault(str(relation_class), {})[str(world)] = {
            "canonical_relations": int(len(block)),
            "similar_z_top1": int(block["similar_response_null_z_absolute_rank"].eq(1).sum()),
            "unsigned_z_top1": int(block["unsigned_response_null_z_absolute_rank"].eq(1).sum()),
            "median_similar_z_rank": float(block["similar_response_null_z_absolute_rank"].median()),
            "median_unsigned_z_rank": float(block["unsigned_response_null_z_absolute_rank"].median()),
            "empirical_p_at_minimum_resolution": int(
                block["unsigned_response_null_empirical_p"].le(1 / (args.permutations + 1)).sum()
            ),
        }
    report = {
        "schema": "contrast_atlas_sparse_local_response_null_calibration_v2",
        "status": "DEVELOPMENT_CONSUMED_RESPONSE_SPECIFICITY_CALIBRATION_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "permutations": args.permutations,
            "seed": args.seed,
            "reassignment_unit": "whole entity response profile",
            "reassignment_strata": "within study and common identity status",
            "preserved": [
                "frozen phenotype-blind chemical graph",
                "complete candidate universe",
                "study and contrast structure",
                "within-profile cross-contrast dependence",
                "identity-world entity counts",
            ],
            "calibrated_score": "candidate-specific z score against response-profile reassignment",
            "score_mode": args.score_mode,
            "score_normalization": (
                "complete identity-world entity count"
                if args.score_mode == "entity_mean"
                else "geometric mean of complete identity-world query and candidate response energy"
            ),
            "registered_relations_opened": "after every observed and null candidate score was frozen",
        },
        "denominators": {
            "studies": len(matrices),
            "contrasts": len(contrasts),
            "candidate_rows": len(calibrated),
            "permutations": args.permutations,
            "canonical_registered_relations": int(relations["relation_direction"].eq("registered").sum()),
            "registered_directional_relations": len(relations),
            "independent_biological_relation_components": relation_components,
        },
        "canonical_registered_relation_descriptive_ranks": summary,
        "canonical_registered_relation_strata_descriptive_ranks": stratum_summary,
        "decision_boundary": {
            "pass_fail": "NOT_ADJUDICATED",
            "reason": "heterogeneous consumed and post-hoc relation strata, no signed metabolite-level truth, and no pre-registered pass gate",
            "unregistered_pairs": "unadjudicated, never treated as negatives",
        },
        "provenance": {
            "alignment_report_sha256": sha256(alignment_paths["report"]),
            "alignment_scores_sha256": sha256(alignment_paths["scores"]),
            "alignment_edges_sha256": sha256(alignment_paths["edges"]),
            "response_entities_sha256": sha256(collection_paths["entities"]),
            "contrast_manifest_sha256": sha256(collection_paths["contrasts"]),
            "registered_relations_sha256": sha256(collection_paths["relations"]),
            "identity_partition_sha256": sha256(args.identity_partition),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "This calibration removes candidate-specific chemical-composition expectations; it does not create biological truth.",
            "Empirical p values are descriptive development statistics and are not multiplicity-corrected discovery claims.",
            f"Forward and reverse registered rows are not independent; the primary summary uses {int(len(canonical) / len(WORLDS))} canonical relations per evidence world.",
            f"The {relation_components} relation components do not imply equal experimental independence; twelve exact-drug components share one consumed ChemProp project.",
            "No external generalization claim is qualified.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
