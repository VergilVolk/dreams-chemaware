#!/usr/bin/env python3
"""Build a complete, traceable Contrast Atlas connection evidence ledger.

This is a deterministic materialisation step.  It does not rerun permutations,
select new connections, or read registered biological relations.  The primary
track expands the frozen complete-axis WSE reciprocal same-precursor edges into
every study-contrast pair to which each edge can contribute.  It then proves
that the row-level contributions reproduce the already-frozen same/opposite
response strengths.

The primary edges are *identity hypotheses*, not confirmed molecular identity.
The WSE value is a non-negative evidence weight, not a probability.  An optional
legacy different-mass fragment-cosine relation track is written to separate
files and is never admitted to the identity ledger.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_open_local_edge_enrichment import add_effect_statistics


COMMON_IDENTITY_COUNTS = {
    "library_supported": 979,
    "unresolved_entity": 53882,
}
WORLDS = ("all", "known", "unresolved")
IDENTITY_TRACK = "same_precursor_10ppm_identity_hypothesis"
RELATION_TRACK = "nonmass_chemical_relation_hypothesis"
EDGE_KEY = [
    "left_study_id",
    "right_study_id",
    "evidence_world",
    "left_entity_id",
    "right_entity_id",
]
CONNECTION_KEY = [
    "left_study_id",
    "left_contrast_id",
    "right_study_id",
    "right_contrast_id",
    "evidence_world",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    unknown = ~text.isin({"true", "false", "1", "0"})
    if unknown.any():
        raise RuntimeError(
            f"cannot parse Boolean values: {sorted(text.loc[unknown].unique())[:5]}"
        )
    return text.isin({"true", "1"})


def load_common_identity_partition(path: Path) -> pd.DataFrame:
    partition = pd.read_csv(path, sep="\t", dtype=str)
    required = {
        "study_id",
        "entity_id",
        "identity_status",
        "common_identity_policy",
    }
    if not required.issubset(partition.columns):
        raise RuntimeError(f"common identity partition lacks {sorted(required - set(partition))}")
    if partition.duplicated(["study_id", "entity_id"]).any():
        raise RuntimeError("common identity partition contains duplicate entity keys")
    counts = partition["identity_status"].value_counts().to_dict()
    if counts != COMMON_IDENTITY_COUNTS:
        raise RuntimeError(
            f"common identity denominator drift: {counts} != {COMMON_IDENTITY_COUNTS}"
        )
    if partition["common_identity_policy"].nunique() != 1:
        raise RuntimeError("more than one common identity policy is present")
    return partition.sort_values(["study_id", "entity_id"]).reset_index(drop=True)


def prepare_response_effects(
    response_path: Path,
    partition: pd.DataFrame,
    effect_fdr: float,
) -> pd.DataFrame:
    effects = pd.read_csv(response_path, sep="\t", low_memory=False)
    required = {"study_id", "entity_id", "contrast_id", "beta", "se"}
    if not required.issubset(effects.columns):
        raise RuntimeError(f"response table lacks {sorted(required - set(effects))}")
    effects["study_id"] = effects["study_id"].astype(str)
    effects["entity_id"] = effects["entity_id"].astype(str)
    effects["contrast_id"] = effects["contrast_id"].astype(str)
    if effects.duplicated(["study_id", "entity_id", "contrast_id"]).any():
        raise RuntimeError("response table contains duplicate entity/contrast rows")

    if "identity_status" in effects.columns:
        effects = effects.rename(columns={"identity_status": "collection_identity_status"})
    effects = effects.merge(
        partition.rename(
            columns={
                "identity_status": "common_identity_status",
                "common_identity_policy": "common_identity_policy",
            }
        ),
        on=["study_id", "entity_id"],
        how="left",
        validate="many_to_one",
    )
    if effects["common_identity_status"].isna().any():
        raise RuntimeError("common identity partition does not cover every response row")
    response_keys = set(map(tuple, effects[["study_id", "entity_id"]].drop_duplicates().to_numpy()))
    partition_keys = set(map(tuple, partition[["study_id", "entity_id"]].to_numpy()))
    if response_keys != partition_keys:
        raise RuntimeError(
            "response entity universe differs from the frozen common identity partition"
        )

    effects = add_effect_statistics(effects, effect_fdr)
    # The frozen retrieval stored profiles as float32.  Preserve the exact
    # numerical input used by the existing strength calculation explicitly.
    effects["retrieval_response_weight"] = (
        effects["response_weight"].to_numpy(float).astype(np.float32).astype(float)
    )
    beta = pd.to_numeric(effects["beta"], errors="coerce").to_numpy(float)
    effects["response_direction"] = np.where(
        beta > 0, "positive", np.where(beta < 0, "negative", "zero")
    )
    effects["identity_partition_role"] = "common_gnps_gold_silver_partition"
    return effects.sort_values(
        ["study_id", "entity_id", "contrast_id"], kind="stable"
    ).reset_index(drop=True)


def prepare_method_edges(
    edge_path: Path,
    partition: pd.DataFrame,
    *,
    axis: str,
) -> pd.DataFrame:
    edges = pd.read_csv(edge_path, sep="\t", low_memory=False)
    required = set(EDGE_KEY).union(
        {
            "chemical_similarity",
            "directional_support",
            "left_precursor_mz",
            "right_precursor_mz",
            "precursor_ppm_difference",
            "precursor_compatible_10ppm",
            "precursor_incompatible_over_10ppm",
        }
    )
    if not required.issubset(edges.columns):
        raise RuntimeError(f"edge table lacks {sorted(required - set(edges))}")
    for column in ("left_study_id", "right_study_id", "left_entity_id", "right_entity_id"):
        edges[column] = edges[column].astype(str)
    edges["precursor_compatible_10ppm"] = _normalise_bool(
        edges["precursor_compatible_10ppm"]
    )
    edges["precursor_incompatible_over_10ppm"] = _normalise_bool(
        edges["precursor_incompatible_over_10ppm"]
    )
    if axis == "identity":
        edges = edges.loc[edges["precursor_compatible_10ppm"]].copy()
        if edges["precursor_incompatible_over_10ppm"].any():
            raise RuntimeError("identity axis contains different-mass relation edges")
        method_name = "weighted_spectral_entropy_complete_axis"
        edge_claim = (
            "mutual_top1_same_precursor_10ppm_identity_hypothesis_not_confirmed_structure"
        )
        prefix = "wse_identity"
    elif axis == "legacy_relation":
        edges = edges.loc[edges["precursor_incompatible_over_10ppm"]].copy()
        if edges["precursor_compatible_10ppm"].any():
            raise RuntimeError("legacy relation axis contains same-precursor identity edges")
        method_name = "legacy_fragment_cosine_mutual_top1_relation_axis_exploratory"
        edge_claim = (
            "different_mass_chemical_relation_hypothesis_distinct_entities_never_identity"
        )
        prefix = "legacy_relation"
    else:
        raise ValueError(axis)

    if edges.duplicated(EDGE_KEY).any():
        raise RuntimeError(f"{axis} edge table contains duplicate canonical keys")
    if set(edges["evidence_world"].astype(str)) != set(WORLDS):
        raise RuntimeError(f"{axis} edges do not cover all frozen evidence worlds")
    score = pd.to_numeric(edges["chemical_similarity"], errors="coerce").to_numpy(float)
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise RuntimeError(f"{axis} evidence weights must be finite in [0,1]")
    if not pd.to_numeric(edges["directional_support"], errors="coerce").eq(2).all():
        raise RuntimeError(f"{axis} table contains non-reciprocal edges")

    status = partition.set_index(["study_id", "entity_id"])["identity_status"]
    try:
        edges["left_common_identity_status"] = [
            status.loc[(study, entity)]
            for study, entity in zip(edges["left_study_id"], edges["left_entity_id"])
        ]
        edges["right_common_identity_status"] = [
            status.loc[(study, entity)]
            for study, entity in zip(edges["right_study_id"], edges["right_entity_id"])
        ]
    except KeyError as exc:
        raise RuntimeError(f"edge endpoint is absent from common identity partition: {exc}") from exc

    known = edges["evidence_world"].eq("known")
    unresolved = edges["evidence_world"].eq("unresolved")
    if not (
        edges.loc[known, ["left_common_identity_status", "right_common_identity_status"]]
        .eq("library_supported")
        .all()
        .all()
    ):
        raise RuntimeError(f"{axis} known world violates the common identity partition")
    if not (
        edges.loc[
            unresolved,
            ["left_common_identity_status", "right_common_identity_status"],
        ]
        .eq("unresolved_entity")
        .all()
        .all()
    ):
        raise RuntimeError(f"{axis} unresolved world violates the common identity partition")

    edges = edges.sort_values(EDGE_KEY, kind="stable").reset_index(drop=True)
    edges.insert(0, "method_edge_id", [f"{prefix}:{index:06d}" for index in range(len(edges))])
    edges.insert(1, "method_name", method_name)
    edges.insert(2, "edge_claim_boundary", edge_claim)
    edges.insert(
        3,
        "score_semantics",
        "nonnegative_spectral_evidence_weight_not_probability",
    )
    edges = edges.rename(columns={"chemical_similarity": "method_evidence_weight"})
    return edges


def canonical_reference_connections(
    score_path: Path,
    evidence_track: str,
) -> pd.DataFrame:
    scores = pd.read_csv(score_path, sep="\t", low_memory=False)
    required = {
        "undirected_id",
        "query_study_id",
        "query_contrast_id",
        "candidate_study_id",
        "candidate_contrast_id",
        "evidence_world",
        "evidence_track",
        "same_direction_strength",
        "opposite_direction_strength",
    }
    if not required.issubset(scores.columns):
        raise RuntimeError(f"candidate score table lacks {sorted(required - set(scores))}")
    # The legacy 2026-10-27 fragment-cosine ledger predates the explicit
    # weighted-edge column.  Every retained legacy edge has strictly positive
    # cosine, so its weighted-edge count is exactly its mutual-edge count.
    if "weighted_evidence_edges" not in scores.columns:
        scores["weighted_evidence_edges"] = scores["mutual_edges"]
    scores = scores.loc[scores["evidence_track"].eq(evidence_track)].copy()
    if not len(scores):
        raise RuntimeError(f"candidate score table contains no {evidence_track} rows")

    query_first = scores["query_study_id"].astype(str) < scores["candidate_study_id"].astype(str)
    scores["left_study_id"] = np.where(
        query_first, scores["query_study_id"], scores["candidate_study_id"]
    )
    scores["left_contrast_id"] = np.where(
        query_first, scores["query_contrast_id"], scores["candidate_contrast_id"]
    )
    scores["right_study_id"] = np.where(
        query_first, scores["candidate_study_id"], scores["query_study_id"]
    )
    scores["right_contrast_id"] = np.where(
        query_first, scores["candidate_contrast_id"], scores["query_contrast_id"]
    )
    duplicated = scores.groupby(CONNECTION_KEY, sort=False)
    sizes = duplicated.size()
    if not sizes.eq(2).all():
        raise RuntimeError("each frozen connection must have exactly two directed score rows")
    for column in (
        "undirected_id",
        "same_direction_strength",
        "opposite_direction_strength",
    ):
        if not duplicated[column].nunique(dropna=False).eq(1).all():
            raise RuntimeError(f"directed score rows disagree on {column}")
    canonical = scores.loc[query_first].copy()
    if canonical.duplicated(CONNECTION_KEY).any():
        raise RuntimeError("canonical reference connection keys are not unique")
    keep = [
        "undirected_id",
        *CONNECTION_KEY,
        "mutual_edges",
        "weighted_evidence_edges",
        "same_direction_dual_fdr_edges",
        "opposite_direction_dual_fdr_edges",
        "same_direction_strength",
        "opposite_direction_strength",
    ]
    return canonical[keep].sort_values(CONNECTION_KEY, kind="stable").reset_index(drop=True)


def iter_edge_contribution_blocks(
    method_edges: pd.DataFrame,
    effects: pd.DataFrame,
    references: pd.DataFrame,
) -> Iterator[pd.DataFrame]:
    effect_columns = [
        "study_id",
        "entity_id",
        "contrast_id",
        "beta",
        "se",
        "effect_q",
        "response_qualified",
        "retrieval_response_weight",
        "common_identity_status",
    ]
    effect_by_study = {
        str(study): block[effect_columns].copy()
        for study, block in effects.groupby("study_id", sort=False)
    }
    contrasts_by_study = {
        study: pd.DataFrame(
            {"contrast_id": sorted(block["contrast_id"].astype(str).unique())}
        )
        for study, block in effect_by_study.items()
    }
    connection_ids = references.set_index(CONNECTION_KEY)["undirected_id"]
    for (left_study, right_study, world), edge_block in method_edges.groupby(
        ["left_study_id", "right_study_id", "evidence_world"], sort=True
    ):
        left = effect_by_study[str(left_study)].rename(
            columns={
                "entity_id": "left_entity_id",
                "contrast_id": "left_contrast_id",
                "beta": "left_beta",
                "se": "left_se",
                "effect_q": "left_effect_q",
                "response_qualified": "left_response_qualified",
                "retrieval_response_weight": "left_response_weight",
                "common_identity_status": "left_common_identity_status_effect",
            }
        ).drop(columns="study_id")
        right = effect_by_study[str(right_study)].rename(
            columns={
                "entity_id": "right_entity_id",
                "contrast_id": "right_contrast_id",
                "beta": "right_beta",
                "se": "right_se",
                "effect_q": "right_effect_q",
                "response_qualified": "right_response_qualified",
                "retrieval_response_weight": "right_response_weight",
                "common_identity_status": "right_common_identity_status_effect",
            }
        ).drop(columns="study_id")
        # Response tables need not be rectangular: an entity can be absent
        # from one contrast in a multi-contrast study.  The frozen retrieval
        # nevertheless counts every method edge in every contrast pair and
        # gives an unobserved endpoint zero response weight.  Materialise the
        # complete edge x left-contrast x right-contrast product before joining
        # effects so zero-contribution rows remain traceable.
        left_contrasts = contrasts_by_study[str(left_study)].rename(
            columns={"contrast_id": "left_contrast_id"}
        )
        right_contrasts = contrasts_by_study[str(right_study)].rename(
            columns={"contrast_id": "right_contrast_id"}
        )
        expanded = edge_block.merge(left_contrasts, how="cross").merge(
            left,
            on=["left_entity_id", "left_contrast_id"],
            how="left",
            validate="many_to_one",
        ).merge(right_contrasts, how="cross").merge(
            right,
            on=["right_entity_id", "right_contrast_id"],
            how="left",
            validate="many_to_one",
        )
        expanded["left_effect_observed"] = expanded["left_response_weight"].notna()
        expanded["right_effect_observed"] = expanded["right_response_weight"].notna()
        expanded[["left_response_weight", "right_response_weight"]] = expanded[
            ["left_response_weight", "right_response_weight"]
        ].fillna(0.0)
        for column in ("left_response_qualified", "right_response_qualified"):
            expanded[column] = (
                expanded[column].astype("boolean").fillna(False).astype(bool)
            )
        keys = pd.MultiIndex.from_frame(expanded[CONNECTION_KEY])
        try:
            expanded.insert(0, "undirected_connection_id", connection_ids.loc[keys].to_numpy())
        except KeyError as exc:
            raise RuntimeError(f"expanded edge has no frozen connection row: {exc}") from exc

        left_weight = expanded["left_response_weight"].to_numpy(float)
        right_weight = expanded["right_response_weight"].to_numpy(float)
        evidence_weight = expanded["method_evidence_weight"].to_numpy(float)
        same_sign = np.sign(left_weight) == np.sign(right_weight)
        nonzero = (left_weight != 0) & (right_weight != 0)
        magnitude = evidence_weight * np.abs(left_weight) * np.abs(right_weight)
        expanded["same_direction_contribution"] = np.where(
            nonzero & same_sign, magnitude, 0.0
        )
        expanded["opposite_direction_contribution"] = np.where(
            nonzero & ~same_sign, magnitude, 0.0
        )
        left_qualified = expanded["left_response_qualified"].astype(bool).to_numpy()
        right_qualified = expanded["right_response_qualified"].astype(bool).to_numpy()
        expanded["same_direction_dual_fdr_support"] = (
            left_qualified & right_qualified & nonzero & same_sign
        )
        expanded["opposite_direction_dual_fdr_support"] = (
            left_qualified & right_qualified & nonzero & ~same_sign
        )
        expanded["connection_role"] = (
            "edge_contribution_to_response_connection_no_effect_meta_analysis"
        )
        keep = [
            "undirected_connection_id",
            "method_edge_id",
            "method_name",
            "edge_claim_boundary",
            "score_semantics",
            *CONNECTION_KEY,
            "left_entity_id",
            "right_entity_id",
            "left_precursor_mz",
            "right_precursor_mz",
            "precursor_ppm_difference",
            "method_evidence_weight",
            "left_common_identity_status",
            "right_common_identity_status",
            "left_beta",
            "left_se",
            "left_effect_q",
            "left_effect_observed",
            "left_response_weight",
            "right_beta",
            "right_se",
            "right_effect_q",
            "right_effect_observed",
            "right_response_weight",
            "same_direction_dual_fdr_support",
            "opposite_direction_dual_fdr_support",
            "same_direction_contribution",
            "opposite_direction_contribution",
            "connection_role",
        ]
        yield expanded[keep].sort_values(
            ["undirected_connection_id", "method_edge_id"], kind="stable"
        ).reset_index(drop=True)


def summarise_and_validate_contributions(
    contribution_blocks: list[pd.DataFrame],
    references: pd.DataFrame,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if contribution_blocks:
        summaries = []
        for block in contribution_blocks:
            summaries.append(
                block.groupby(CONNECTION_KEY, sort=False).agg(
                    reconstructed_method_edges=("method_edge_id", "size"),
                    reconstructed_weighted_evidence_edges=(
                        "method_evidence_weight",
                        lambda values: int(np.count_nonzero(np.asarray(values, dtype=float) > 0)),
                    ),
                    reconstructed_same_direction_dual_fdr_edges=(
                        "same_direction_dual_fdr_support",
                        "sum",
                    ),
                    reconstructed_opposite_direction_dual_fdr_edges=(
                        "opposite_direction_dual_fdr_support",
                        "sum",
                    ),
                    reconstructed_same_direction_strength=(
                        "same_direction_contribution",
                        "sum",
                    ),
                    reconstructed_opposite_direction_strength=(
                        "opposite_direction_contribution",
                        "sum",
                    ),
                ).reset_index()
            )
        reconstructed = pd.concat(summaries, ignore_index=True)
        reconstructed = reconstructed.groupby(CONNECTION_KEY, sort=False).sum().reset_index()
    else:
        reconstructed = pd.DataFrame(columns=CONNECTION_KEY)
    merged = references.merge(
        reconstructed,
        on=CONNECTION_KEY,
        how="left",
        validate="one_to_one",
    )
    reconstructed_columns = [column for column in merged if column.startswith("reconstructed_")]
    merged[reconstructed_columns] = merged[reconstructed_columns].fillna(0)
    count_pairs = (
        ("mutual_edges", "reconstructed_method_edges"),
        ("weighted_evidence_edges", "reconstructed_weighted_evidence_edges"),
        ("same_direction_dual_fdr_edges", "reconstructed_same_direction_dual_fdr_edges"),
        (
            "opposite_direction_dual_fdr_edges",
            "reconstructed_opposite_direction_dual_fdr_edges",
        ),
    )
    for expected, observed in count_pairs:
        if not np.array_equal(
            pd.to_numeric(merged[expected]).to_numpy(int),
            pd.to_numeric(merged[observed]).to_numpy(int),
        ):
            raise RuntimeError(f"contribution count conservation failed for {expected}")

    diagnostics: dict[str, float] = {}
    for face in ("same", "opposite"):
        expected_column = f"{face}_direction_strength"
        observed_column = f"reconstructed_{face}_direction_strength"
        expected = pd.to_numeric(merged[expected_column]).to_numpy(float)
        observed = pd.to_numeric(merged[observed_column]).to_numpy(float)
        difference = observed - expected
        merged[f"{face}_strength_reconstruction_difference"] = difference
        diagnostics[f"max_abs_{face}_strength_difference"] = float(
            np.max(np.abs(difference), initial=0.0)
        )
        if not np.allclose(
            observed,
            expected,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            worst = int(np.argmax(np.abs(difference)))
            raise RuntimeError(
                f"{face} contribution conservation failed at row {worst}: "
                f"observed={observed[worst]}, expected={expected[worst]}, "
                f"difference={difference[worst]}"
            )
    merged["contribution_conservation_pass"] = True
    return merged, diagnostics


def write_gzip_blocks(path: Path, blocks: list[pd.DataFrame]) -> int:
    rows = 0
    with path.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                for index, block in enumerate(blocks):
                    block.to_csv(text, sep="\t", index=False, header=index == 0)
                    rows += len(block)
    return rows


def materialise_axis(
    *,
    edge_path: Path,
    score_path: Path,
    output_edge_path: Path,
    output_contribution_path: Path,
    output_connection_path: Path,
    partition: pd.DataFrame,
    effects: pd.DataFrame,
    axis: str,
    expected_edge_rows: int | None,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    track = IDENTITY_TRACK if axis == "identity" else RELATION_TRACK
    edges = prepare_method_edges(edge_path, partition, axis=axis)
    if expected_edge_rows is not None and len(edges) != expected_edge_rows:
        raise RuntimeError(
            f"{axis} edge denominator drift: {len(edges)} != {expected_edge_rows}"
        )
    references = canonical_reference_connections(score_path, track)
    blocks = list(iter_edge_contribution_blocks(edges, effects, references))
    connections, diagnostics = summarise_and_validate_contributions(
        blocks,
        references,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    compression = {"method": "gzip", "mtime": 0}
    edges.to_csv(output_edge_path, sep="\t", index=False, compression=compression)
    contribution_rows = write_gzip_blocks(output_contribution_path, blocks)
    connections.to_csv(
        output_connection_path, sep="\t", index=False, compression=compression
    )
    return {
        "axis": axis,
        "method_edges": int(len(edges)),
        "connection_rows_undirected": int(len(connections)),
        "connection_rows_directed_in_frozen_score_table": int(2 * len(connections)),
        "edge_contribution_rows": int(contribution_rows),
        "world_edge_counts": {
            str(key): int(value)
            for key, value in edges["evidence_world"].value_counts().sort_index().items()
        },
        "conservation": diagnostics,
        "outputs": {
            output_edge_path.name: sha256(output_edge_path),
            output_contribution_path.name: sha256(output_contribution_path),
            output_connection_path.name: sha256(output_connection_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"
        ),
    )
    parser.add_argument(
        "--identity-partition",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz"
        ),
    )
    parser.add_argument(
        "--identity-retrieval-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v3_20261029"
        ),
    )
    parser.add_argument(
        "--legacy-relation-dir",
        type=Path,
        help=(
            "Optional legacy fragment-cosine directory.  Its different-mass relation "
            "axis is emitted separately and never merged with identity evidence."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--effect-fdr", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-5)
    parser.add_argument("--relative-tolerance", type=float, default=2e-8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (0 < args.effect_fdr < 1):
        raise ValueError("--effect-fdr must lie in (0,1)")

    response_path = args.collection / "response_entities.tsv.gz"
    identity_edges = args.identity_retrieval_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz"
    identity_scores = args.identity_retrieval_dir / "open_local_edge_candidate_scores.tsv.gz"
    inputs = [response_path, args.identity_partition, identity_edges, identity_scores]
    if args.legacy_relation_dir is not None:
        inputs.extend(
            [
                args.legacy_relation_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
                args.legacy_relation_dir / "open_local_edge_candidate_scores.tsv.gz",
            ]
        )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    partition = load_common_identity_partition(args.identity_partition)
    effects = prepare_response_effects(response_path, partition, args.effect_fdr)
    if len(effects) != 238412:
        raise RuntimeError(f"response row denominator drift: {len(effects)} != 238412")

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    response_output = args.output_dir / "response_effects.tsv.gz"
    effects.to_csv(response_output, sep="\t", index=False, compression=compression)

    identity_summary = materialise_axis(
        edge_path=identity_edges,
        score_path=identity_scores,
        output_edge_path=args.output_dir / "method_entity_edges.tsv.gz",
        output_contribution_path=args.output_dir / "connection_edge_contributions.tsv.gz",
        output_connection_path=args.output_dir / "identity_connection_strengths.tsv.gz",
        partition=partition,
        effects=effects,
        axis="identity",
        expected_edge_rows=67465,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )

    relation_summary = None
    if args.legacy_relation_dir is not None:
        relation_summary = materialise_axis(
            edge_path=args.legacy_relation_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
            score_path=args.legacy_relation_dir / "open_local_edge_candidate_scores.tsv.gz",
            output_edge_path=args.output_dir / "legacy_relation_edges.tsv.gz",
            output_contribution_path=(
                args.output_dir / "legacy_relation_edge_contributions.tsv.gz"
            ),
            output_connection_path=(
                args.output_dir / "legacy_relation_connection_strengths.tsv.gz"
            ),
            partition=partition,
            effects=effects,
            axis="legacy_relation",
            expected_edge_rows=23989,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )

    report = {
        "schema": "contrast_atlas_connection_evidence_ledger_v1",
        "status": "DEVELOPMENT_CONSUMED_COMPLETE_TRACEABLE_LEDGER_BUILT",
        "data_role": "development_consumed",
        "operation": (
            "deterministic materialisation of frozen response effects, entity edges, "
            "and per-edge response contributions; no permutation or model rerun"
        ),
        "denominators": {
            "response_rows": int(len(effects)),
            "unique_study_entities": int(
                effects[["study_id", "entity_id"]].drop_duplicates().shape[0]
            ),
            "common_identity_partition": COMMON_IDENTITY_COUNTS,
        },
        "identity_axis": identity_summary,
        "legacy_relation_axis": relation_summary,
        "claim_boundaries": [
            "Identity-track rows are same-precursor reciprocal spectral identity hypotheses, not confirmed molecular structures.",
            "method_evidence_weight is a WSE or legacy fragment-cosine evidence weight, never a probability.",
            "Endpoint beta and SE values remain study-specific; the ledger never meta-analyses effects across distinct entities.",
            "The optional different-mass relation axis is exploratory, describes distinct entities, and is never merged into identity evidence.",
            "Contribution conservation reproduces a frozen score calculation; it does not validate a biological connection.",
            "Registered relations and outcome truth are not read by this materialisation step.",
        ],
        "provenance": {
            "inputs": {str(path): sha256(path) for path in inputs},
            "response_effects.tsv.gz": sha256(response_output),
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = sorted(
        path for path in args.output_dir.iterdir() if path.name != "checksums.sha256"
    )
    (args.output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
