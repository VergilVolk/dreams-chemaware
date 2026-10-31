#!/usr/bin/env python
"""Signed, energy-normalised Contrast Atlas response retrieval.

This experiment addresses a specific failure of the earlier open-response
operator: same- and opposite-direction evidence were accumulated as two
non-negative scores, and both scores increased with the amount of matched
chemical evidence.  Here every matched entity contributes one signed product,
the sum is normalised by endpoint response energy, and a query receives only
one signed answer.  A second, stricter score subtracts the mean response match
to the other contrasts in the same candidate study.  The latter is unavailable
when a candidate study contains only one contrast; no pseudo-control is made.

The chemical edge graph remains frozen and phenotype-blind.  Registered
relations are read only after the complete score ledger has been written and
hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_open_local_edge_enrichment import (
    add_effect_statistics,
    build_profile_reassignments,
    build_profiles,
)


METHODS = (
    "signed_cosine",
    "dual_fdr_signed_cosine",
    "candidate_source_centered_signed_cosine",
    "source_pair_double_centered_signed_cosine",
    "entity_profile_residual_signed_cosine",
)
WORLDS = ("all", "known", "unresolved")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_cosine_matrix(
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded signed response cosine and overlapping-edge counts."""
    left = np.nan_to_num(left_weight.astype(np.float64), nan=0.0)
    right = np.nan_to_num(right_weight.astype(np.float64), nan=0.0)
    weighted_left = similarity[:, None] * left
    numerator = weighted_left.T @ right
    left_energy = (similarity[:, None] * left * left).sum(axis=0)
    right_energy = (similarity[:, None] * right * right).sum(axis=0)
    denominator = np.sqrt(left_energy[:, None] * right_energy[None, :])
    score = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    support = (np.abs(left) > 0).astype(np.int32).T @ (
        np.abs(right) > 0
    ).astype(np.int32)
    return np.clip(score, -1.0, 1.0), support


def center_candidate_axis(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Remove the candidate-study mean while preserving the query axis."""
    if matrix.shape[axis] < 2:
        return np.full_like(matrix, np.nan)
    finite_count = np.isfinite(matrix).sum(axis=axis, keepdims=True)
    total = np.nansum(matrix, axis=axis, keepdims=True)
    mean = np.divide(
        total,
        finite_count,
        out=np.full_like(total, np.nan),
        where=finite_count > 0,
    )
    return matrix - mean


def double_center_source_pair(matrix: np.ndarray, query_axis: int, candidate_axis: int) -> np.ndarray:
    """Retain only the contrast-by-contrast interaction within a source pair."""
    if matrix.shape[query_axis] < 2 or matrix.shape[candidate_axis] < 2:
        return np.full_like(matrix, np.nan)
    query_mean = center_candidate_axis(matrix, query_axis)
    return center_candidate_axis(query_mean, candidate_axis)


def residualise_entity_profiles(
    matrix: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Remove each entity's cross-contrast mean within a multi-axis study."""
    if matrix.shape[1] < 2:
        return np.full_like(matrix, np.nan)
    values = np.where(observed, matrix, np.nan)
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    total = np.nansum(values, axis=1, keepdims=True)
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    )
    return values - mean


def endpoint_donors(
    rows: np.ndarray,
    status: np.ndarray,
    precursor_mz: np.ndarray,
    permutations: int,
    seed: int,
    key: str,
) -> np.ndarray:
    """Permute only observed graph endpoints within identity and m/z strata."""
    donors = np.tile(rows.astype(np.int64), (permutations, 1))
    mass_bin = np.floor(np.asarray(precursor_mz, dtype=float) / 100.0).astype(int)
    labels = np.asarray(
        [f"{identity}|{bin_id}" for identity, bin_id in zip(status, mass_bin)],
        dtype=object,
    )
    key_seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    for replicate in range(permutations):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, key_seed]))
        for label in sorted(set(labels)):
            positions = np.flatnonzero(labels == label)
            if len(positions) >= 2:
                donors[replicate, positions] = rows[rng.permutation(positions)]
    return donors


def load_inputs(
    collection: Path,
    identity_partition: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    effects = pd.read_csv(collection / "response_entities.tsv.gz", sep="\t", low_memory=False)
    contrasts = pd.read_csv(collection / "contrast_manifest.tsv", sep="\t")
    identity = pd.read_csv(identity_partition, sep="\t", dtype=str)
    for frame in (effects, contrasts, identity):
        frame["study_id"] = frame["study_id"].astype(str)
    effects["entity_id"] = effects["entity_id"].astype(str)
    identity["entity_id"] = identity["entity_id"].astype(str)
    effects = effects.drop(columns=["identity_status"], errors="ignore").merge(
        identity[["study_id", "entity_id", "identity_status"]],
        on=["study_id", "entity_id"],
        how="left",
        validate="many_to_one",
    )
    if effects["identity_status"].isna().any():
        raise RuntimeError("identity partition does not cover every response entity")
    effects = add_effect_statistics(effects, 0.05)
    profiles = build_profiles(effects, contrasts, 0.05)
    return effects, contrasts, profiles, {}


def score_pairs(
    profiles: dict[str, object],
    donors: dict[str, np.ndarray],
    edges: pd.DataFrame,
    null_mode: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, object]] = []
    null_by_row: dict[int, np.ndarray] = {}
    row_id = 0
    edge_blocks = {
        (str(left), str(right), str(world)): block.reset_index(drop=True)
        for (left, right, world), block in edges.groupby(
            ["left_study_id", "right_study_id", "evidence_world"], sort=False
        )
    }
    empty = edges.iloc[0:0].copy()
    for left_study, right_study in combinations(sorted(profiles), 2):
        left = profiles[str(left_study)]
        right = profiles[str(right_study)]
        left_residual = residualise_entity_profiles(
            left.weight, np.isfinite(left.beta) & np.isfinite(left.se)
        )
        right_residual = residualise_entity_profiles(
            right.weight, np.isfinite(right.beta) & np.isfinite(right.se)
        )
        left_qualified = np.where(np.isfinite(left.q) & (left.q <= 0.05), left.weight, 0.0)
        right_qualified = np.where(
            np.isfinite(right.q) & (right.q <= 0.05), right.weight, 0.0
        )
        left_lookup = {value: index for index, value in enumerate(left.entities)}
        right_lookup = {value: index for index, value in enumerate(right.entities)}
        for world in WORLDS:
            block = edge_blocks.get((left_study, right_study, world), empty)
            permutations = donors[str(left_study)].shape[0]
            if len(block):
                left_rows = np.asarray(
                    [left_lookup[str(value)] for value in block["left_entity_id"]],
                    dtype=np.int64,
                )
                right_rows = np.asarray(
                    [right_lookup[str(value)] for value in block["right_entity_id"]],
                    dtype=np.int64,
                )
                similarity = block["chemical_similarity"].to_numpy(float)
                observed, support = signed_cosine_matrix(
                    left.weight[left_rows], right.weight[right_rows], similarity
                )
                residual_observed, residual_support = signed_cosine_matrix(
                    left_residual[left_rows], right_residual[right_rows], similarity
                )
                qualified_observed, qualified_support = signed_cosine_matrix(
                    left_qualified[left_rows], right_qualified[right_rows], similarity
                )
                null = np.full(
                    (permutations, *observed.shape), np.nan, dtype=np.float32
                )
                residual_null = np.full_like(null, np.nan)
                qualified_null = np.full_like(null, np.nan)
                if null_mode == "edge_endpoints_identity_mz100":
                    left_endpoint_donors = endpoint_donors(
                        left_rows,
                        left.status[left_rows],
                        block["left_precursor_mz"].to_numpy(float),
                        permutations,
                        seed,
                        f"{left_study}|{right_study}|{world}|left",
                    )
                    right_endpoint_donors = endpoint_donors(
                        right_rows,
                        right.status[right_rows],
                        block["right_precursor_mz"].to_numpy(float),
                        permutations,
                        seed,
                        f"{left_study}|{right_study}|{world}|right",
                    )
                for replicate in range(permutations):
                    if null_mode == "edge_endpoints_identity_mz100":
                        left_perm = left_endpoint_donors[replicate]
                        right_perm = right_endpoint_donors[replicate]
                    else:
                        left_perm = donors[str(left_study)][replicate, left_rows]
                        right_perm = donors[str(right_study)][replicate, right_rows]
                    null[replicate], _ = signed_cosine_matrix(
                        left.weight[left_perm], right.weight[right_perm], similarity
                    )
                    residual_null[replicate], _ = signed_cosine_matrix(
                        left_residual[left_perm],
                        right_residual[right_perm],
                        similarity,
                    )
                    qualified_null[replicate], _ = signed_cosine_matrix(
                        left_qualified[left_perm],
                        right_qualified[right_perm],
                        similarity,
                    )
            else:
                observed = np.full(
                    (len(left.contrasts), len(right.contrasts)), np.nan, dtype=float
                )
                support = np.zeros_like(observed, dtype=np.int32)
                residual_observed = np.full_like(observed, np.nan)
                residual_support = np.zeros_like(observed, dtype=np.int32)
                qualified_observed = np.full_like(observed, np.nan)
                qualified_support = np.zeros_like(observed, dtype=np.int32)
                null = np.full(
                    (permutations, *observed.shape), np.nan, dtype=np.float32
                )
                residual_null = np.full_like(null, np.nan)
                qualified_null = np.full_like(null, np.nan)

            observed_left_to_right = {
                "signed_cosine": observed,
                "dual_fdr_signed_cosine": qualified_observed,
                "candidate_source_centered_signed_cosine": center_candidate_axis(observed, 1),
                "source_pair_double_centered_signed_cosine": double_center_source_pair(
                    observed, 0, 1
                ),
                "entity_profile_residual_signed_cosine": residual_observed,
            }
            null_left_to_right = {
                "signed_cosine": null,
                "dual_fdr_signed_cosine": qualified_null,
                "candidate_source_centered_signed_cosine": center_candidate_axis(null, 2),
                "source_pair_double_centered_signed_cosine": double_center_source_pair(
                    null, 1, 2
                ),
                "entity_profile_residual_signed_cosine": residual_null,
            }
            observed_right_to_left = {
                "signed_cosine": observed.T,
                "dual_fdr_signed_cosine": qualified_observed.T,
                "candidate_source_centered_signed_cosine": center_candidate_axis(observed.T, 1),
                "source_pair_double_centered_signed_cosine": double_center_source_pair(
                    observed.T, 0, 1
                ),
                "entity_profile_residual_signed_cosine": residual_observed.T,
            }
            null_right_to_left = {
                "signed_cosine": null.transpose(0, 2, 1),
                "dual_fdr_signed_cosine": qualified_null.transpose(0, 2, 1),
                "candidate_source_centered_signed_cosine": center_candidate_axis(
                    null.transpose(0, 2, 1), 2
                ),
                "source_pair_double_centered_signed_cosine": double_center_source_pair(
                    null.transpose(0, 2, 1), 1, 2
                ),
                "entity_profile_residual_signed_cosine": residual_null.transpose(0, 2, 1),
            }
            directions = (
            (
                str(left_study),
                str(right_study),
                left.contrasts,
                right.contrasts,
                observed_left_to_right,
                null_left_to_right,
                support,
                residual_support,
                qualified_support,
            ),
            (
                str(right_study),
                str(left_study),
                right.contrasts,
                left.contrasts,
                observed_right_to_left,
                null_right_to_left,
                support.T,
                residual_support.T,
                qualified_support.T,
            ),
            )
            for (
            query_study,
            candidate_study,
            query_contrasts,
            candidate_contrasts,
            observed_by_method,
            null_by_method,
            directed_support,
            directed_residual_support,
            directed_qualified_support,
        ) in directions:
                for method in METHODS:
                    method_observed = observed_by_method[method]
                    method_null = null_by_method[method]
                    for query_index, query_contrast in enumerate(query_contrasts):
                        for candidate_index, candidate_contrast in enumerate(candidate_contrasts):
                            value = float(method_observed[query_index, candidate_index])
                            null_values = method_null[:, query_index, candidate_index].astype(float)
                            available = bool(np.isfinite(value) and np.isfinite(null_values).all())
                            if available:
                                empirical_p = float(
                                    (1 + np.count_nonzero(np.abs(null_values) >= abs(value)))
                                    / (len(null_values) + 1)
                                )
                                null_mean = float(np.mean(null_values))
                                null_sd = float(np.std(null_values, ddof=1))
                            else:
                                empirical_p = 1.0
                                null_mean = np.nan
                                null_sd = np.nan
                            rows.append(
                            {
                                "row_id": row_id,
                                "query_study_id": query_study,
                                "query_contrast_id": str(query_contrast),
                                "candidate_study_id": candidate_study,
                                "candidate_contrast_id": str(candidate_contrast),
                                "evidence_world": str(world),
                                "response_method": method,
                                "signed_score": value,
                                "absolute_score": abs(value) if np.isfinite(value) else np.nan,
                                "response_face": (
                                    "similar" if value > 0 else "opposite" if value < 0 else "zero"
                                ),
                                "mutual_edges": int(len(block)),
                                "response_overlap_edges": int(
                                    (
                                        directed_residual_support
                                        if method == "entity_profile_residual_signed_cosine"
                                        else directed_qualified_support
                                        if method == "dual_fdr_signed_cosine"
                                        else directed_support
                                    )[query_index, candidate_index]
                                ),
                                "candidate_source_contrasts": int(len(candidate_contrasts)),
                                "score_available": available,
                                "candidate_empirical_two_sided_p": empirical_p,
                                "null_mean": null_mean,
                                "null_sd": null_sd,
                            }
                            )
                            null_by_row[row_id] = null_values
                            row_id += 1
    return pd.DataFrame(rows), null_by_row


def calibrate_open_queries(
    scores: pd.DataFrame,
    null_by_row: dict[int, np.ndarray],
) -> pd.DataFrame:
    scores = scores.copy()
    group_columns = ["query_study_id", "query_contrast_id", "evidence_world", "response_method"]
    scores["absolute_rank"] = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    scores["query_max_absT_p"] = 1.0
    scores["evaluable_candidate_contrasts"] = 0
    for _key, indices in scores.groupby(group_columns, sort=False).groups.items():
        usable = [index for index in indices if bool(scores.loc[index, "score_available"])]
        scores.loc[list(indices), "evaluable_candidate_contrasts"] = len(usable)
        if not usable:
            continue
        values = scores.loc[usable, "absolute_score"]
        scores.loc[usable, "absolute_rank"] = values.rank(
            method="min", ascending=False
        ).astype("Int64")
        null_stack = np.vstack(
            [np.abs(null_by_row[int(scores.loc[index, "row_id"])]) for index in usable]
        )
        max_null = np.nanmax(null_stack, axis=0)
        for index in usable:
            observed = float(scores.loc[index, "absolute_score"])
            scores.loc[index, "query_max_absT_p"] = float(
                (1 + np.count_nonzero(max_null >= observed)) / (len(max_null) + 1)
            )
    return scores


def make_top_hits(
    scores: pd.DataFrame,
    minimum_support: int,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["query_study_id", "query_contrast_id", "evidence_world", "response_method"]
    for key, block in scores.groupby(group_columns, sort=True):
        available = block.loc[block["score_available"]].sort_values(
            ["absolute_rank", "candidate_study_id", "candidate_contrast_id"]
        )
        if available.empty:
            rows.append(
                dict(
                    zip(group_columns, key),
                    result_class="abstain_no_contrast_specific_candidate",
                    candidate_study_id="",
                    candidate_contrast_id="",
                    response_face="",
                    signed_score=np.nan,
                    absolute_score=np.nan,
                    query_max_absT_p=1.0,
                    response_overlap_edges=0,
                    evaluable_candidate_contrasts=0,
                )
            )
            continue
        hit = available.iloc[0]
        if int(hit["response_overlap_edges"]) < minimum_support:
            result = "abstain_insufficient_evidence"
        elif float(hit["query_max_absT_p"]) <= alpha:
            result = f"{hit['response_face']}_response"
        else:
            result = "no_calibrated_match"
        record = {column: value for column, value in zip(group_columns, key)}
        record.update(
            {
                "result_class": result,
                "candidate_study_id": str(hit["candidate_study_id"]),
                "candidate_contrast_id": str(hit["candidate_contrast_id"]),
                "response_face": str(hit["response_face"]),
                "signed_score": float(hit["signed_score"]),
                "absolute_score": float(hit["absolute_score"]),
                "query_max_absT_p": float(hit["query_max_absT_p"]),
                "response_overlap_edges": int(hit["response_overlap_edges"]),
                "evaluable_candidate_contrasts": int(
                    hit["evaluable_candidate_contrasts"]
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def registered_ranks(scores: pd.DataFrame, relation_path: Path) -> pd.DataFrame:
    relations = pd.read_csv(relation_path, sep="\t", dtype=str)
    frames: list[pd.DataFrame] = []
    for world in WORLDS:
        for method in METHODS:
            block = scores.loc[
                scores["evidence_world"].eq(world)
                & scores["response_method"].eq(method)
            ]
            merged = relations.merge(
                block,
                left_on=[
                    "left_study_id",
                    "left_contrast_id",
                    "right_study_id",
                    "right_contrast_id",
                ],
                right_on=[
                    "query_study_id",
                    "query_contrast_id",
                    "candidate_study_id",
                    "candidate_contrast_id",
                ],
                how="left",
                validate="one_to_one",
            )
            frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--edge-handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--minimum-support", type=int, default=3)
    parser.add_argument("--query-alpha", type=float, default=0.05)
    parser.add_argument("--parent-project-map", type=Path)
    parser.add_argument(
        "--null-mode",
        choices=("all_entities_identity", "edge_endpoints_identity_mz100"),
        default="edge_endpoints_identity_mz100",
    )
    args = parser.parse_args()

    for path in (
        args.collection / "response_entities.tsv.gz",
        args.collection / "contrast_manifest.tsv",
        args.collection / "registered_relations.tsv",
        args.identity_partition,
        args.edge_handoff,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    effects, contrasts, profiles, _ = load_inputs(
        args.collection, args.identity_partition
    )
    donors = build_profile_reassignments(profiles, args.permutations, args.seed)
    edges = pd.read_csv(args.edge_handoff, sep="\t", low_memory=False)
    if set(edges["evidence_world"].unique()) != set(WORLDS):
        raise RuntimeError("edge handoff does not contain all/known/unresolved worlds")
    if edges.duplicated(
        [
            "left_study_id",
            "right_study_id",
            "evidence_world",
            "left_entity_id",
            "right_entity_id",
        ]
    ).any():
        raise RuntimeError("edge handoff contains duplicate world-specific entity pairs")

    scores, null_by_row = score_pairs(
        profiles, donors, edges, args.null_mode, args.seed
    )
    parent_project_provenance: dict[str, object] = {}
    candidate_universe = "source_disjoint"
    if args.parent_project_map is not None:
        mapping = pd.read_csv(args.parent_project_map, sep="\t", dtype=str)
        if mapping["study_id"].duplicated().any():
            raise RuntimeError("parent-project map contains duplicate study IDs")
        lookup = dict(zip(mapping["study_id"], mapping["parent_project_id"]))
        studies = set(contrasts["study_id"].astype(str))
        if set(lookup) != studies:
            raise RuntimeError(
                "parent-project map does not exactly cover the response collection: "
                f"missing={sorted(studies - set(lookup))}, extra={sorted(set(lookup) - studies)}"
            )
        scores["query_parent_project"] = scores["query_study_id"].map(lookup)
        scores["candidate_parent_project"] = scores["candidate_study_id"].map(lookup)
        scores = scores.loc[
            scores["query_parent_project"].ne(scores["candidate_parent_project"])
        ].reset_index(drop=True)
        candidate_universe = "parent_project_disjoint"
        parent_project_provenance = {
            "parent_project_map_sha256": sha256(args.parent_project_map),
            "parent_project_families": int(len(set(lookup.values()))),
        }
    scores = calibrate_open_queries(scores, null_by_row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0}
    score_path = args.output_dir / "signed_response_candidate_scores.tsv.gz"
    scores.to_csv(score_path, sep="\t", index=False, compression=compression)
    score_hash_before_truth = sha256(score_path)

    top = make_top_hits(scores, args.minimum_support, args.query_alpha)
    top_path = args.output_dir / "signed_response_top_hits.tsv"
    top.to_csv(top_path, sep="\t", index=False)
    registered = registered_ranks(
        scores, args.collection / "registered_relations.tsv"
    )
    registered_path = args.output_dir / "registered_relation_ranks.tsv"
    registered.to_csv(registered_path, sep="\t", index=False)

    call_summary = (
        top.groupby(["evidence_world", "response_method", "result_class"])
        .size()
        .rename("queries")
        .reset_index()
        .to_dict(orient="records")
    )
    relation_summary: list[dict[str, object]] = []
    primary = registered.loc[
        registered["eligible_primary_relevance"].astype(str).str.lower().eq("true")
    ]
    for (world, method), block in primary.groupby(
        ["evidence_world", "response_method"], sort=True
    ):
        ranks = pd.to_numeric(block["absolute_rank"], errors="coerce")
        relation_summary.append(
            {
                "evidence_world": str(world),
                "response_method": str(method),
                "eligible_primary_relations": int(len(block)),
                "evaluable_relations": int(ranks.notna().sum()),
                "top1": int(ranks.eq(1).sum()),
                "top5": int(ranks.le(5).sum()),
                "median_rank": None if not ranks.notna().any() else float(ranks.median()),
                "query_maxT_005": int(
                    pd.to_numeric(block["query_max_absT_p"], errors="coerce")
                    .le(args.query_alpha)
                    .sum()
                ),
            }
        )

    output_hashes = {
        score_path.name: sha256(score_path),
        top_path.name: sha256(top_path),
        registered_path.name: sha256(registered_path),
    }
    report = {
        "schema": "contrast_atlas_signed_specificity_retrieval_v1",
        "status": "DEVELOPMENT_CONSUMED_SIGNED_SPECIFICITY_RETRIEVAL_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "signed_cosine": "sum(similarity * query_weight * candidate_weight) divided by matched endpoint response energy",
            "candidate_source_centering": "subtract the query-specific mean score across all contrasts in the candidate study; unavailable for single-contrast candidate studies",
            "source_pair_double_centering": "subtract both source-pair contrast main effects and retain only the contrast-by-contrast interaction; unavailable unless both sources have at least two contrasts",
            "entity_profile_residualisation": "subtract each entity's mean response across all contrasts in its study before signed-cosine scoring; unavailable unless both sources have at least two contrasts",
            "response_weight": "beta / sqrt(beta^2 + SE^2)",
            "dual_fdr_arm": "signed cosine restricted to edges whose two endpoints each have within-contrast normal-approximation BH q <= 0.05; conservative development sensitivity analysis",
            "null": (
                "whole multi-contrast response profiles reassigned only among the actual source-pair graph endpoints within identity and fixed 100-Da precursor strata"
                if args.null_mode == "edge_endpoints_identity_mz100"
                else "whole multi-contrast entity response profiles reassigned within study and common identity stratum"
            ),
            "null_mode": args.null_mode,
            "open_calibration": f"single two-sided query-level max-abs-T across the complete {candidate_universe.replace('_', '-')} candidate universe",
            "candidate_universe": candidate_universe,
            "registered_truth_timing": "registered relations opened only after signed_response_candidate_scores.tsv.gz was written and hashed",
        },
        "denominators": {
            "study_source_ids": int(contrasts["study_id"].nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "response_rows": int(len(effects)),
            "world_specific_edge_rows": int(len(edges)),
            "directed_score_rows": int(len(scores)),
            "permutations": int(args.permutations),
            **parent_project_provenance,
        },
        "open_query_calls": call_summary,
        "registered_relation_summary": relation_summary,
        "claim_boundary": [
            "All sources and contrasts are development/consumed; this is not sealed external validation.",
            "A max-T call rejects only the implemented entity-profile reassignment null; it does not establish a biological mechanism.",
            "Candidate-source centering measures contrast specificity relative to other contrasts in that source and abstains when that comparison is impossible.",
            "Known and unresolved worlds must be interpreted separately; this run cannot establish an unresolved-information increment without a matched comparison.",
            "The edge handoff uses WSE; this run does not test Full23 or BioAware.",
        ],
        "provenance": {
            "response_entities_sha256": sha256(args.collection / "response_entities.tsv.gz"),
            "contrast_manifest_sha256": sha256(args.collection / "contrast_manifest.tsv"),
            "identity_partition_sha256": sha256(args.identity_partition),
            "edge_handoff_sha256": sha256(args.edge_handoff),
            **(
                {"parent_project_map_sha256": sha256(args.parent_project_map)}
                if args.parent_project_map is not None
                else {}
            ),
            "score_sha256_before_registered_relations": score_hash_before_truth,
            "outputs_sha256": output_hashes,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksums = args.output_dir / "checksums.sha256"
    checksums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(args.output_dir.iterdir()) if path.name != checksums.name),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
