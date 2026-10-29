#!/usr/bin/env python3
"""Build reciprocal-edge-opportunity-matched unresolved WSE controls.

The full complete-axis WSE graph contains far more unresolved than strict-known
entities and reciprocal edges.  This response-blind control holds fixed the
number and source-pair distribution of reciprocal edges that can enter the
study-response retrieval.  For each strict-known reciprocal edge, it selects
one unresolved reciprocal edge without replacement inside an exact stratum:

* unordered source pair;
* physicochemical compatibility grade; and
* the number of response contrasts available in the two sources.

Within a stratum, matching uses only ion, candidate-opportunity and frozen WSE
edge-quality covariates.  It never reads beta, SE, q values, phenotype labels,
registered biological relations, molecule names or structure identifiers.

This is an edge-opportunity control, not a complete measurement-matched design:
platform, peak count, TIC, retention time and chemical-class coverage remain
uncontrolled and must stay outside the claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_contrast_atlas_complete_identity_wse_alignment import (
    KNOWN,
    UNRESOLVED,
    sha256,
    world_mask,
)


KEY = ["left_study_id", "right_study_id", "left_entity_id", "right_entity_id"]
STRATA = [
    "left_study_id",
    "right_study_id",
    "physicochemical_compatibility_grade",
    "left_n_response_contrasts",
    "right_n_response_contrasts",
]
MATCH_COVARIATES = [
    "log_precursor_midpoint",
    "symmetric_ppm",
    "conservative_wse",
    "log1p_min_matched_fragments",
    "log1p_left_candidate_count",
    "log1p_right_candidate_count",
    "left_singleton",
    "right_singleton",
    "left_top1_top2_margin",
    "right_top1_top2_margin",
    "log1p_left_full_degree",
    "log1p_right_full_degree",
]


def stable_seed(base_seed: int, replicate: int, values: tuple[object, ...]) -> int:
    payload = "|".join([str(base_seed), str(replicate), *map(str, values)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (
        2**63 - 1
    )


def canonicalise_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy()
    forward = result["query_study_id"].astype(str) < result["candidate_study_id"].astype(str)
    result["left_study_id"] = result["query_study_id"].where(
        forward, result["candidate_study_id"]
    )
    result["right_study_id"] = result["candidate_study_id"].where(
        forward, result["query_study_id"]
    )
    result["left_entity_id"] = result["query_entity_id"].where(
        forward, result["candidate_entity_id"]
    )
    result["right_entity_id"] = result["candidate_entity_id"].where(
        forward, result["query_entity_id"]
    )
    return result


def directional_metrics(
    pairs: pd.DataFrame,
    score: np.ndarray,
    matched: np.ndarray,
    available: np.ndarray,
    world: str,
) -> pd.DataFrame:
    mask = world_mask(pairs, world) & np.asarray(available, dtype=bool)
    work = pairs.loc[mask, [
        "query_block_index",
        "query_study_id",
        "query_entity_id",
        "candidate_study_id",
        "candidate_entity_id",
        "query_precursor_mz",
        "candidate_precursor_mz",
        "symmetric_ppm",
        "physicochemical_compatibility_grade",
    ]].copy()
    work["wse_score"] = np.asarray(score, dtype=float)[mask]
    work["matched_fragments"] = np.asarray(matched, dtype=int)[mask]
    grouped = work.groupby("query_block_index", sort=False)["wse_score"]
    work["world_candidate_count"] = grouped.transform("size").astype(int)
    top_two = grouped.apply(
        lambda values: np.sort(values.to_numpy(float))[-2:]
    )
    margins: dict[object, float] = {}
    for block_index, values in top_two.items():
        if len(values) == 1:
            margins[block_index] = 0.0
        else:
            margins[block_index] = float(values[-1] - values[-2])
    work["top1_top2_margin"] = work["query_block_index"].map(margins).astype(float)
    work["singleton"] = work["world_candidate_count"].eq(1).astype(int)
    if work.duplicated([
        "query_study_id", "query_entity_id", "candidate_study_id", "candidate_entity_id"
    ]).any():
        raise RuntimeError("directed candidate axis contains duplicate endpoint pairs")
    return work


def mutual_edge_table(
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    contrast_counts: dict[str, int],
) -> pd.DataFrame:
    """Return one row per frozen reciprocal edge with response-blind covariates."""
    if edges.empty:
        return pd.DataFrame()
    counts = edges.groupby(KEY, sort=False)["direction"].nunique()
    mutual_keys = counts[counts.eq(2)].reset_index()[KEY]
    mutual_edges = edges.merge(mutual_keys, on=KEY, how="inner", validate="many_to_one")
    left_to_right = mutual_edges["direction"].eq("left_to_right")
    mutual_edges["query_study_id"] = mutual_edges["left_study_id"].where(
        left_to_right, mutual_edges["right_study_id"]
    )
    mutual_edges["query_entity_id"] = mutual_edges["left_entity_id"].where(
        left_to_right, mutual_edges["right_entity_id"]
    )
    mutual_edges["candidate_study_id"] = mutual_edges["right_study_id"].where(
        left_to_right, mutual_edges["left_study_id"]
    )
    mutual_edges["candidate_entity_id"] = mutual_edges["right_entity_id"].where(
        left_to_right, mutual_edges["left_entity_id"]
    )
    merged = mutual_edges.merge(
        metrics,
        on=[
            "query_study_id", "query_entity_id",
            "candidate_study_id", "candidate_entity_id",
        ],
        how="left",
        validate="one_to_one",
    )
    if merged["wse_score"].isna().any():
        raise RuntimeError("frozen WSE edge is absent from directed metric axis")
    if not np.allclose(
        pd.to_numeric(merged["chemical_similarity"], errors="raise"),
        pd.to_numeric(merged["wse_score"], errors="raise"),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("frozen edge score disagrees with aligned WSE score")

    direction_metrics = merged.pivot(
        index=KEY,
        columns="direction",
        values=[
            "world_candidate_count", "singleton", "top1_top2_margin",
            "wse_score", "matched_fragments",
        ],
    )
    direction_metrics.columns = [f"{direction}_{name}" for name, direction in direction_metrics.columns]
    direction_metrics = direction_metrics.reset_index()

    chemistry = (
        merged.groupby(KEY, sort=False)
        .agg(
            physicochemical_compatibility_grade=(
                "physicochemical_compatibility_grade", "first"
            ),
            left_precursor_mz=("query_precursor_mz", "first"),
            right_precursor_mz=("candidate_precursor_mz", "first"),
            symmetric_ppm=("symmetric_ppm", "max"),
        )
        .reset_index()
    )
    # The first directional row is not guaranteed to point left-to-right.
    precursor = canonicalise_pairs(merged[[
        "query_study_id", "query_entity_id", "candidate_study_id", "candidate_entity_id",
        "query_precursor_mz", "candidate_precursor_mz",
    ]]).copy()
    precursor["left_precursor_mz"] = precursor["query_precursor_mz"].where(
        precursor["query_study_id"].astype(str) < precursor["candidate_study_id"].astype(str),
        precursor["candidate_precursor_mz"],
    )
    precursor["right_precursor_mz"] = precursor["candidate_precursor_mz"].where(
        precursor["query_study_id"].astype(str) < precursor["candidate_study_id"].astype(str),
        precursor["query_precursor_mz"],
    )
    precursor = precursor.drop_duplicates(KEY)[KEY + ["left_precursor_mz", "right_precursor_mz"]]
    chemistry = chemistry.drop(columns=["left_precursor_mz", "right_precursor_mz"]).merge(
        precursor, on=KEY, validate="one_to_one"
    )
    table = chemistry.merge(direction_metrics, on=KEY, validate="one_to_one")

    for side in ("left", "right"):
        table[f"{side}_n_response_contrasts"] = table[f"{side}_study_id"].map(
            contrast_counts
        )
        if table[f"{side}_n_response_contrasts"].isna().any():
            raise RuntimeError("contrast count missing for a graph source")

    left_key = list(zip(table["left_study_id"], table["left_entity_id"]))
    right_key = list(zip(table["right_study_id"], table["right_entity_id"]))
    degree: dict[tuple[str, str], int] = {}
    for key in [*left_key, *right_key]:
        degree[(str(key[0]), str(key[1]))] = degree.get((str(key[0]), str(key[1])), 0) + 1
    table["left_full_degree"] = [degree[(str(a), str(b))] for a, b in left_key]
    table["right_full_degree"] = [degree[(str(a), str(b))] for a, b in right_key]

    table["precursor_midpoint"] = (
        pd.to_numeric(table["left_precursor_mz"], errors="raise")
        + pd.to_numeric(table["right_precursor_mz"], errors="raise")
    ) / 2.0
    table["log_precursor_midpoint"] = np.log(table["precursor_midpoint"].to_numpy(float))
    table["conservative_wse"] = table[[
        "left_to_right_wse_score", "right_to_left_wse_score"
    ]].min(axis=1)
    table["min_matched_fragments"] = table[[
        "left_to_right_matched_fragments", "right_to_left_matched_fragments"
    ]].min(axis=1)
    table["log1p_min_matched_fragments"] = np.log1p(
        table["min_matched_fragments"].to_numpy(float)
    )
    table["left_candidate_count"] = table["left_to_right_world_candidate_count"].astype(int)
    table["right_candidate_count"] = table["right_to_left_world_candidate_count"].astype(int)
    table["left_singleton"] = table["left_to_right_singleton"].astype(int)
    table["right_singleton"] = table["right_to_left_singleton"].astype(int)
    table["left_top1_top2_margin"] = table["left_to_right_top1_top2_margin"].astype(float)
    table["right_top1_top2_margin"] = table["right_to_left_top1_top2_margin"].astype(float)
    table["log1p_left_candidate_count"] = np.log1p(table["left_candidate_count"])
    table["log1p_right_candidate_count"] = np.log1p(table["right_candidate_count"])
    table["log1p_left_full_degree"] = np.log1p(table["left_full_degree"])
    table["log1p_right_full_degree"] = np.log1p(table["right_full_degree"])
    table["edge_id"] = [
        hashlib.sha256("|".join(map(str, row)).encode("utf-8")).hexdigest()
        for row in table[KEY].itertuples(index=False, name=None)
    ]
    return table


def robust_scale(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[MATCH_COVARIATES].to_numpy(float)
    centre = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = q75 - q25
    standard = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
    return centre, scale


def match_edges(
    known: pd.DataFrame,
    unresolved: pd.DataFrame,
    base_seed: int,
    replicate: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    pair_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    for stratum, known_block in known.groupby(STRATA, sort=True, dropna=False):
        values = stratum if isinstance(stratum, tuple) else (stratum,)
        mask = np.ones(len(unresolved), dtype=bool)
        for column, value in zip(STRATA, values):
            mask &= unresolved[column].eq(value).to_numpy()
        pool = unresolved.loc[mask].reset_index(drop=True)
        reference = known_block.reset_index(drop=True)
        if len(pool) < len(reference):
            raise RuntimeError(
                f"unresolved edge shortage in stratum {values}: {len(pool)} < {len(reference)}"
            )
        combined = pd.concat([reference, pool], ignore_index=True)
        centre, scale = robust_scale(combined)
        ref_x = (reference[MATCH_COVARIATES].to_numpy(float) - centre) / scale
        pool_x = (pool[MATCH_COVARIATES].to_numpy(float) - centre) / scale
        rng = np.random.default_rng(stable_seed(base_seed, replicate, values))
        order = rng.permutation(len(reference))
        available = np.ones(len(pool), dtype=bool)
        chosen_for_reference: dict[int, tuple[int, float]] = {}
        for reference_index in order:
            candidate_indices = np.flatnonzero(available)
            distance = np.sum(
                (pool_x[candidate_indices] - ref_x[reference_index]) ** 2,
                axis=1,
            )
            distance = distance + rng.uniform(0.0, 1e-12, size=len(distance))
            local = int(np.argmin(distance))
            chosen = int(candidate_indices[local])
            available[chosen] = False
            chosen_for_reference[int(reference_index)] = (chosen, float(distance[local]))
        selected_indices = [chosen_for_reference[index][0] for index in range(len(reference))]
        selected = pool.iloc[selected_indices].copy()
        selections.append(selected)
        for index, known_row in reference.iterrows():
            chosen_index, distance = chosen_for_reference[int(index)]
            selected_row = pool.iloc[chosen_index]
            pair_rows.append({
                "replicate": replicate,
                **{column: value for column, value in zip(STRATA, values)},
                "known_edge_id": str(known_row["edge_id"]),
                "selected_unresolved_edge_id": str(selected_row["edge_id"]),
                "match_distance": distance,
            })
        strata_rows.append({
            "replicate": replicate,
            **{column: value for column, value in zip(STRATA, values)},
            "known_edges": int(len(reference)),
            "unresolved_edges_available": int(len(pool)),
            "selected_unresolved_edges": int(len(selected)),
        })
    selected_all = pd.concat(selections, ignore_index=True)
    if len(selected_all) != len(known):
        raise RuntimeError("matched reciprocal-edge denominator drift")
    if selected_all["edge_id"].duplicated().any():
        raise RuntimeError("an unresolved reciprocal edge was reused across strata")
    return selected_all, pd.DataFrame(pair_rows), pd.DataFrame(strata_rows)


def balance_table(
    known: pd.DataFrame,
    selected: pd.DataFrame,
    replicate: int,
) -> pd.DataFrame:
    rows = []
    for column in MATCH_COVARIATES:
        left = known[column].to_numpy(float)
        right = selected[column].to_numpy(float)
        pooled = np.sqrt((np.var(left) + np.var(right)) / 2.0)
        difference = float(np.mean(right) - np.mean(left))
        rows.append({
            "replicate": replicate,
            "covariate": column,
            "known_mean": float(np.mean(left)),
            "selected_unresolved_mean": float(np.mean(right)),
            "mean_difference": difference,
            "standardized_mean_difference": difference / pooled if pooled > 1e-12 else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-dir", type=Path,
        default=Path("data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"),
    )
    parser.add_argument(
        "--wse-dir", type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029"),
    )
    parser.add_argument(
        "--alignment-dir", type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_alignment_v3_20261029"),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.resamples < 2:
        raise ValueError("at least two edge-matched resamples are required")

    paths = {
        "pairs": args.axis_dir / "directed_complete_candidate_pairs.tsv.gz",
        "blocks": args.axis_dir / "query_target_source_blocks.tsv.gz",
        "wse": args.wse_dir / "wse_scores_aligned.npz",
        "full_edges": args.alignment_dir / "source_pair_frozen_edges.tsv.gz",
        "alignment_report": args.alignment_dir / "report.json",
        "contrast_manifest": args.collection / "contrast_manifest.tsv",
        "response_entities": args.collection / "response_entities.tsv.gz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    alignment_report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    if alignment_report["provenance"]["edges"] != sha256(paths["full_edges"]):
        raise RuntimeError("full alignment edge checksum mismatch")
    if alignment_report["provenance"]["response_entities_sha256"] != sha256(
        paths["response_entities"]
    ):
        raise RuntimeError("alignment and response collection disagree")

    pairs = pd.read_csv(paths["pairs"], sep="\t", low_memory=False)
    compact = np.load(paths["wse"], allow_pickle=False)
    score = compact["wse_score"].astype(np.float64)
    matched = compact["wse_matched_fragments"].astype(np.int64)
    available = compact["wse_availability"].astype(bool)
    if not (len(pairs) == len(score) == len(matched) == len(available)):
        raise RuntimeError("complete candidate and WSE axes are not aligned")
    edges = pd.read_csv(paths["full_edges"], sep="\t")
    contrast_manifest = pd.read_csv(paths["contrast_manifest"], sep="\t", dtype=str)
    contrast_counts = (
        contrast_manifest.groupby("study_id")["contrast_id"].nunique().astype(int).to_dict()
    )

    known_metrics = directional_metrics(pairs, score, matched, available, "known")
    unresolved_metrics = directional_metrics(pairs, score, matched, available, "unresolved")
    known = mutual_edge_table(
        edges.loc[edges["evidence_world"].eq("known")].copy(),
        known_metrics,
        contrast_counts,
    )
    unresolved = mutual_edge_table(
        edges.loc[edges["evidence_world"].eq("unresolved")].copy(),
        unresolved_metrics,
        contrast_counts,
    )
    if known.empty or unresolved.empty:
        raise RuntimeError("known or unresolved reciprocal graph is empty")

    args.output_root.mkdir(parents=True)
    match_frames: list[pd.DataFrame] = []
    strata_frames: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    unresolved_edges = edges.loc[edges["evidence_world"].eq("unresolved")].copy()
    for replicate in range(args.resamples):
        selected, matches, strata = match_edges(
            known, unresolved, args.seed, replicate
        )
        selected_keys = selected[KEY].copy()
        output_edges = unresolved_edges.merge(
            selected_keys, on=KEY, how="inner", validate="many_to_one"
        ).sort_values(KEY + ["direction"], kind="stable")
        if len(output_edges) != 2 * len(known):
            raise RuntimeError("each selected reciprocal edge must retain both directions")
        directory = args.output_root / f"resample_{replicate:02d}"
        directory.mkdir()
        edge_path = directory / "source_pair_frozen_edges.tsv.gz"
        output_edges.to_csv(
            edge_path, sep="\t", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        balance = balance_table(known, selected, replicate)
        report = {
            "schema": "contrast_atlas_wse_reciprocal_edge_opportunity_matched_alignment_v1",
            "status": "DEVELOPMENT_CONSUMED_EDGE_OPPORTUNITY_MATCHED_WSE_ALIGNMENT_FROZEN",
            "data_role": "development_consumed",
            "chemical_graph": (
                "complete-axis unresolved WSE reciprocal identity edges matched one-to-one "
                "to strict-known reciprocal edges by source pair, compatibility grade and "
                "response-contrast availability before any response effect is read"
            ),
            "replicate": replicate,
            "selection_seed": args.seed,
            "matching": {
                "unit": "canonical_reciprocal_wse_edge",
                "exact_strata": STRATA,
                "numeric_covariates": MATCH_COVARIATES,
                "without_replacement": True,
                "known_reciprocal_edges": int(len(known)),
                "unresolved_reciprocal_edges_available": int(len(unresolved)),
                "selected_unresolved_reciprocal_edges": int(len(selected)),
                "maximum_absolute_standardized_mean_difference": float(
                    balance["standardized_mean_difference"].abs().max()
                ),
            },
            "denominators": {
                "studies": int(contrast_manifest["study_id"].nunique()),
                "entities": int(alignment_report["denominators"]["entities"]),
                "frozen_directional_edges": int(len(output_edges)),
                "worlds": [{
                    "evidence_world": "unresolved",
                    "directional_top1_edges": int(len(output_edges)),
                    "canonical_mutual_edges": int(len(selected)),
                }],
            },
            "sources": alignment_report["sources"],
            "claim_boundary": [
                "This is a reciprocal-edge-opportunity-matched unresolved control, not a measurement-matched cohort.",
                "Platform, peak count, TIC, retention time, ion-family and chemical-class coverage remain uncontrolled.",
                "Endpoint reuse and graph topology are reported but not forced to match exactly.",
                "No beta, SE, q value, phenotype, registered relation, molecule name or structure identifier entered selection.",
                "All inputs are development/consumed; this cannot validate unknown identity or a biological mechanism.",
            ],
            "provenance": {
                "edges": sha256(edge_path),
                "response_entities_sha256": sha256(paths["response_entities"]),
                **{name: sha256(path) for name, path in paths.items() if name not in {"full_edges", "response_entities"}},
                "implementation_sha256": sha256(Path(__file__)),
            },
        }
        report_path = directory / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        match_frames.append(matches)
        strata_frames.append(strata)
        balance_frames.append(balance)
        manifests.append({
            "replicate": replicate,
            "selected_reciprocal_edges": int(len(selected)),
            "selected_unique_endpoints": int(len(set(
                zip(selected["left_study_id"], selected["left_entity_id"])
            ) | set(zip(selected["right_study_id"], selected["right_entity_id"])))),
            "maximum_selected_endpoint_degree": int(max(
                pd.concat([
                    selected[["left_study_id", "left_entity_id"]].rename(
                        columns={"left_study_id": "study", "left_entity_id": "entity"}
                    ),
                    selected[["right_study_id", "right_entity_id"]].rename(
                        columns={"right_study_id": "study", "right_entity_id": "entity"}
                    ),
                ]).value_counts().max(),
                0,
            )),
            "maximum_absolute_smd": float(
                balance["standardized_mean_difference"].abs().max()
            ),
            "alignment_dir": str(directory),
            "edges_sha256": sha256(edge_path),
            "report_sha256": sha256(report_path),
        })

    match_path = args.output_root / "matched_edge_pairs.tsv.gz"
    pd.concat(match_frames, ignore_index=True).to_csv(
        match_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    strata_path = args.output_root / "exact_strata_balance.tsv.gz"
    pd.concat(strata_frames, ignore_index=True).to_csv(
        strata_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    covariate_path = args.output_root / "covariate_balance.tsv"
    pd.concat(balance_frames, ignore_index=True).to_csv(
        covariate_path, sep="\t", index=False
    )
    manifest_path = args.output_root / "resample_manifest.tsv"
    manifest = pd.DataFrame(manifests)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    root_report = {
        "schema": "contrast_atlas_wse_reciprocal_edge_opportunity_matched_controls_v1",
        "status": "DEVELOPMENT_CONSUMED_EDGE_OPPORTUNITY_MATCHED_WSE_CONTROLS_FROZEN",
        "data_role": "development_consumed",
        "resamples": args.resamples,
        "base_seed": args.seed,
        "known_reciprocal_edges": int(len(known)),
        "unresolved_reciprocal_edges_available": int(len(unresolved)),
        "selected_unresolved_reciprocal_edges_per_resample": int(len(known)),
        "exact_strata": int(known.groupby(STRATA, dropna=False).ngroups),
        "strata_with_shortage": 0,
        "selected_unique_endpoints_range": [
            int(manifest["selected_unique_endpoints"].min()),
            int(manifest["selected_unique_endpoints"].max()),
        ],
        "maximum_selected_endpoint_degree_range": [
            int(manifest["maximum_selected_endpoint_degree"].min()),
            int(manifest["maximum_selected_endpoint_degree"].max()),
        ],
        "maximum_absolute_smd_range": [
            float(manifest["maximum_absolute_smd"].min()),
            float(manifest["maximum_absolute_smd"].max()),
        ],
        "claim_boundary": (
            "reciprocal-edge opportunity matched only; not complete measurement matched, "
            "not external validation and not a formal unknown-information pass"
        ),
        "provenance": {
            **{name: sha256(path) for name, path in paths.items()},
            "matched_edge_pairs": sha256(match_path),
            "exact_strata_balance": sha256(strata_path),
            "covariate_balance": sha256(covariate_path),
            "resample_manifest": sha256(manifest_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    root_report_path = args.output_root / "report.json"
    root_report_path.write_text(
        json.dumps(root_report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(root_report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
