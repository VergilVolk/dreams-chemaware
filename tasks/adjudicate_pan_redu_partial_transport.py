#!/usr/bin/env python3
"""Legacy audit-only adjudicator for the superseded Pan-ReDU estimator.

Unlisted query--target pairs remain ``unknown``.  They stay in the retrieval
candidate universe, so they can displace a registered positive, but they are
never counted as biological negatives or false discoveries.  Inference is
performed on query-source-family means rather than on pair rows or transported
edges.  A delete-one-source-node analysis removes a source from both query and
target roles to expose dyadic source dependence.

This file is retained only to reproduce the historical v1 analysis.  Its
outcome-selected ``strongest known`` comparator and legacy world names do not
define the formal claim.  New runs must use
``build_pan_redu_exact_axis_source_outcomes.py`` followed by
``evaluate_contrast_atlas_source_risk.py``.  The CLI therefore fails closed
unless an operator explicitly requests a legacy audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contrast_response_transport import parse_bool


TRUTH_CLASSES = {"replication_expected", "explicit_mismatch", "unknown"}
EXPECTED_DIRECTIONS = {"similar", "opposite", "not_applicable"}
PRIMARY_WORLDS = (
    "all",
    "known_only",
    "measurement_matched_known",
    "measurement_matched_unresolved",
    "unresolved_ms2_reassigned_left_panel",
    "unresolved_ms2_reassigned_right_panel",
    "measurement_matched_unresolved_ms2_reassigned_left_panel",
    "measurement_matched_unresolved_ms2_reassigned_right_panel",
)
KNOWN_CONTROLS = ("known_only", "measurement_matched_known")
FULL_REASSIGN_CONTROLS = (
    "unresolved_ms2_reassigned_left_panel",
    "unresolved_ms2_reassigned_right_panel",
)
MATCHED_UNRESOLVED_REASSIGN_CONTROLS = (
    "measurement_matched_unresolved_ms2_reassigned_left_panel",
    "measurement_matched_unresolved_ms2_reassigned_right_panel",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth_bool(series: pd.Series, label: str) -> pd.Series:
    try:
        return parse_bool(series, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def validate_and_annotate(
    matrix_raw: pd.DataFrame,
    topology_raw: pd.DataFrame,
    truth_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_required = {
        "world", "left_panel_id", "left_contrast_id", "right_panel_id",
        "right_contrast_id", "answerable", "signed_chemical_evidence",
    }
    topology_required = {
        "panel_id", "contrast_id", "preliminary_source_family",
        "comparison_stratum", "topology_status",
    }
    truth_required = {
        "query_panel_id", "query_contrast_id", "target_panel_id",
        "target_contrast_id", "truth_class", "expected_direction",
        "truth_family_id", "eligible_primary", "evidence_basis",
        "source_independence_status", "truth_frozen_before_scores",
        "selection_blind_to_transport_scores",
    }
    for label, frame, required in (
        ("matrix", matrix_raw, matrix_required),
        ("topology", topology_raw, topology_required),
        ("truth registry", truth_raw, truth_required),
    ):
        if missing := required - set(frame):
            raise RuntimeError(f"{label} lacks columns: {sorted(missing)}")

    matrix = matrix_raw.copy()
    matrix["answerable"] = _truth_bool(matrix["answerable"], "answerable")
    matrix["signed_chemical_evidence"] = pd.to_numeric(
        matrix["signed_chemical_evidence"], errors="coerce"
    )
    key = ["world", "left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id"]
    if matrix.duplicated(key).any():
        raise RuntimeError("matrix contains duplicate ordered query-target-world rows")

    topology = topology_raw.loc[
        topology_raw.topology_status.astype(str).str.startswith("PASS")
    ].copy()
    topology_key = ["panel_id", "contrast_id"]
    if topology.duplicated(topology_key).any():
        raise RuntimeError("topology contains duplicate panel/contrast keys")
    topology = topology[
        ["panel_id", "contrast_id", "preliminary_source_family", "comparison_stratum"]
    ].astype(str)
    query_topology = topology.rename(columns={
        "panel_id": "left_panel_id", "contrast_id": "left_contrast_id",
        "preliminary_source_family": "query_source_family",
        "comparison_stratum": "query_comparison_stratum",
    })
    target_topology = topology.rename(columns={
        "panel_id": "right_panel_id", "contrast_id": "right_contrast_id",
        "preliminary_source_family": "target_source_family",
        "comparison_stratum": "target_comparison_stratum",
    })
    annotated = matrix.merge(
        query_topology, on=["left_panel_id", "left_contrast_id"],
        how="left", validate="many_to_one",
    ).merge(
        target_topology, on=["right_panel_id", "right_contrast_id"],
        how="left", validate="many_to_one",
    )
    if annotated[["query_source_family", "target_source_family"]].isna().any().any():
        raise RuntimeError("matrix contains a contrast absent from PASS topology")
    if annotated.query_source_family.eq(annotated.target_source_family).any():
        raise RuntimeError("matrix violates source-family-disjoint candidate rule")

    truth = truth_raw.copy()
    for column in ("eligible_primary", "truth_frozen_before_scores", "selection_blind_to_transport_scores"):
        truth[column] = _truth_bool(truth[column], column)
    truth_key = ["query_panel_id", "query_contrast_id", "target_panel_id", "target_contrast_id"]
    if truth.duplicated(truth_key).any():
        raise RuntimeError("truth registry contains duplicate ordered pairs")
    if not set(truth.truth_class.astype(str)).issubset(TRUTH_CLASSES):
        raise RuntimeError("truth registry contains an unsupported truth_class")
    if not set(truth.expected_direction.astype(str)).issubset(EXPECTED_DIRECTIONS):
        raise RuntimeError("truth registry contains an unsupported expected_direction")
    expected_required = truth.truth_class.eq("replication_expected")
    if not truth.loc[expected_required, "expected_direction"].isin({"similar", "opposite"}).all():
        raise RuntimeError("replication_expected rows require similar or opposite direction")
    if not truth.loc[truth.truth_class.ne("replication_expected"), "expected_direction"].eq("not_applicable").all():
        raise RuntimeError("non-replication truth rows must use not_applicable direction")
    if not truth.truth_frozen_before_scores.all() or not truth.selection_blind_to_transport_scores.all():
        raise RuntimeError("truth registry is not demonstrably frozen and score blind")
    primary = truth.eligible_primary
    if not truth.loc[primary, "truth_class"].eq("replication_expected").all():
        raise RuntimeError("only replication_expected truth can be primary")
    if not truth.loc[primary, "source_independence_status"].eq("confirmed_independent").all():
        raise RuntimeError("primary truth lacks confirmed source independence")

    truth_for_merge = truth.rename(columns={
        "query_panel_id": "left_panel_id", "query_contrast_id": "left_contrast_id",
        "target_panel_id": "right_panel_id", "target_contrast_id": "right_contrast_id",
    })
    annotated = annotated.merge(
        truth_for_merge,
        on=["left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id"],
        how="left", validate="many_to_one",
    )
    annotated["truth_class"] = annotated.truth_class.fillna("unknown")
    annotated["expected_direction"] = annotated.expected_direction.fillna("not_applicable")
    annotated["eligible_primary"] = annotated.eligible_primary.eq(True)
    annotated["truth_family_id"] = annotated.truth_family_id.fillna("")
    return annotated, truth


def build_query_world_outcomes(annotated: pd.DataFrame) -> pd.DataFrame:
    primary_pairs = annotated.loc[
        annotated.eligible_primary & annotated.truth_class.eq("replication_expected"),
        [
            "left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id",
            "expected_direction", "truth_family_id",
        ],
    ].drop_duplicates()
    if primary_pairs.empty:
        return pd.DataFrame(columns=[
            "world", "query_panel_id", "query_contrast_id", "query_source_family",
            "truth_family_id", "expected_direction", "candidate_denominator",
            "answerable_candidates", "positive_targets", "answerable_positive_targets",
            "positive_rank", "reciprocal_rank", "top_truth_class", "top_is_positive",
            "top_is_explicit_mismatch", "top_is_unknown",
        ])
    query_keys = primary_pairs[["left_panel_id", "left_contrast_id"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for query in query_keys.itertuples(index=False):
        positives = primary_pairs.loc[
            primary_pairs.left_panel_id.eq(query.left_panel_id)
            & primary_pairs.left_contrast_id.eq(query.left_contrast_id)
        ]
        directions = set(positives.expected_direction)
        families = set(positives.truth_family_id)
        if len(directions) != 1 or len(families) != 1:
            raise RuntimeError("one primary query cannot mix directions or truth families")
        direction = next(iter(directions))
        positive_targets = set(zip(positives.right_panel_id, positives.right_contrast_id))
        query_rows = annotated.loc[
            annotated.left_panel_id.eq(query.left_panel_id)
            & annotated.left_contrast_id.eq(query.left_contrast_id)
        ]
        for world_name, candidates in query_rows.groupby("world", sort=True):
            candidates = candidates.copy()
            score = candidates.signed_chemical_evidence.astype(float)
            candidates["retrieval_score"] = score if direction == "similar" else -score
            candidates.loc[
                ~candidates.answerable | ~np.isfinite(candidates.retrieval_score), "retrieval_score"
            ] = -np.inf
            candidates["is_positive"] = [
                (panel, contrast) in positive_targets
                for panel, contrast in zip(candidates.right_panel_id, candidates.right_contrast_id)
            ]
            candidates = candidates.sort_values(
                ["retrieval_score", "right_panel_id", "right_contrast_id"],
                ascending=[False, True, True], kind="stable",
            ).reset_index(drop=True)
            candidates["rank"] = np.arange(1, len(candidates) + 1)
            supported = candidates.retrieval_score.gt(-np.inf)
            supported_positive = candidates.is_positive & supported
            positive_rank = (
                int(candidates.loc[supported_positive, "rank"].min())
                if supported_positive.any() else 0
            )
            top = candidates.loc[supported].iloc[0] if supported.any() else None
            rows.append({
                "world": world_name,
                "query_panel_id": query.left_panel_id,
                "query_contrast_id": query.left_contrast_id,
                "query_source_family": str(candidates.query_source_family.iloc[0]),
                "truth_family_id": next(iter(families)),
                "expected_direction": direction,
                "candidate_denominator": len(candidates),
                "answerable_candidates": int(supported.sum()),
                "positive_targets": len(positive_targets),
                "answerable_positive_targets": int(supported_positive.sum()),
                "positive_rank": positive_rank,
                "reciprocal_rank": 1.0 / positive_rank if positive_rank else 0.0,
                "top_truth_class": str(top.truth_class) if top is not None else "abstained",
                "top_is_positive": bool(top.is_positive) if top is not None else False,
                "top_is_explicit_mismatch": (
                    bool(top.truth_class == "explicit_mismatch") if top is not None else False
                ),
                "top_is_unknown": bool(top.truth_class == "unknown") if top is not None else False,
            })
    return pd.DataFrame(rows)


def source_equal_metrics(outcomes: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for world_name, group in outcomes.groupby("world", sort=True):
        per_source = group.groupby("query_source_family", sort=True).agg(
            mean_rr=("reciprocal_rank", "mean"),
            query_coverage=("answerable_candidates", lambda values: float(np.mean(np.asarray(values) > 0))),
            positive_support=("answerable_positive_targets", lambda values: float(np.mean(np.asarray(values) > 0))),
            top1_exact=("top_is_positive", "mean"),
        )
        result[str(world_name)] = {
            "queries": int(len(group)),
            "query_source_families": int(len(per_source)),
            "source_equal_mrr": float(per_source.mean_rr.mean()),
            "source_equal_query_coverage": float(per_source.query_coverage.mean()),
            "source_equal_positive_support": float(per_source.positive_support.mean()),
            "source_equal_top1_exact": float(per_source.top1_exact.mean()),
        }
    return result


def _source_values(outcomes: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return outcomes.pivot_table(
        index="query_source_family", columns="world", values=value_column, aggfunc="mean"
    )


def source_bootstrap(
    outcomes: pd.DataFrame, draws: int, seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rr = _source_values(outcomes, "reciprocal_rank")
    required = set(PRIMARY_WORLDS)
    if missing := required - set(rr.columns):
        return {"status": "insufficient_worlds", "missing_worlds": sorted(missing)}, pd.DataFrame()
    rr = rr.dropna(subset=list(PRIMARY_WORLDS))
    if len(rr) < 3:
        return {"status": "insufficient_query_source_families", "source_families": len(rr)}, pd.DataFrame()

    transition_rows = []
    all_rows = outcomes.loc[outcomes.world.eq("all")]
    keys = ["query_panel_id", "query_contrast_id", "query_source_family"]
    for control in KNOWN_CONTROLS:
        control_rows = outcomes.loc[outcomes.world.eq(control)]
        paired = all_rows.merge(control_rows, on=keys, suffixes=("__all", "__control"), validate="one_to_one")
        paired["control_world"] = control
        paired["corrected"] = (~paired.top_is_positive__control) & paired.top_is_positive__all
        paired["introduced"] = paired.top_is_positive__control & paired.top_is_explicit_mismatch__all
        transition_rows.append(paired[keys + ["control_world", "corrected", "introduced"]])
    transitions = pd.concat(transition_rows, ignore_index=True)
    transition_source = transitions.groupby(
        ["control_world", "query_source_family"], sort=True
    ).agg(corrected=("corrected", "mean"), introduced=("introduced", "mean")).reset_index()

    rng = np.random.default_rng(seed)
    sources = rr.index.to_numpy(str)
    rows = []
    for draw in range(draws):
        sampled = rng.choice(sources, size=len(sources), replace=True)
        endpoint = rr.loc[sampled].mean(axis=0)
        delta_known = float(endpoint["all"] - max(endpoint[world] for world in KNOWN_CONTROLS))
        delta_reassigned = float(endpoint["all"] - max(endpoint[world] for world in FULL_REASSIGN_CONTROLS))
        delta_matched = float(
            endpoint["measurement_matched_unresolved"]
            - max(endpoint[world] for world in MATCHED_UNRESOLVED_REASSIGN_CONTROLS)
        )
        per_control = []
        for control in KNOWN_CONTROLS:
            table = transition_source.loc[
                transition_source.control_world.eq(control)
            ].set_index("query_source_family")
            sampled_table = table.loc[sampled]
            per_control.append((
                float(sampled_table.corrected.mean()),
                float(sampled_table.introduced.mean()),
            ))
        conservative_risk_net = min(value[0] for value in per_control) - 2 * max(value[1] for value in per_control)
        rows.append({
            "draw": draw,
            "delta_all_vs_strongest_known": delta_known,
            "delta_all_vs_strongest_full_reassignment": delta_reassigned,
            "delta_matched_unresolved_vs_strongest_reassignment": delta_matched,
            "conservative_risk_net": conservative_risk_net,
        })
    replicates = pd.DataFrame(rows)

    def interval(column: str) -> dict[str, float]:
        values = replicates[column].to_numpy(float)
        return {
            "median": float(np.median(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "fraction_positive": float(np.mean(values > 0)),
        }

    report = {
        "status": "SOURCE_FAMILY_BOOTSTRAP_COMPLETE",
        "source_families": len(sources),
        "draws": draws,
        "seed": seed,
        "intervals": {
            column: interval(column) for column in replicates.columns if column != "draw"
        },
        "resampling_unit": "query source family; all queries from a sampled source move together",
        "dyadic_caveat": "target-source dependence is assessed separately by delete-one-source-node",
    }
    return report, replicates


def delete_one_source_node(annotated: pd.DataFrame) -> pd.DataFrame:
    sources = sorted(set(annotated.query_source_family) | set(annotated.target_source_family))
    rows = []
    for source in sources:
        reduced = annotated.loc[
            annotated.query_source_family.ne(source) & annotated.target_source_family.ne(source)
        ]
        outcomes = build_query_world_outcomes(reduced)
        metrics = source_equal_metrics(outcomes) if not outcomes.empty else {}
        if not set(PRIMARY_WORLDS).issubset(metrics):
            rows.append({"deleted_source_family": source, "status": "insufficient_after_delete"})
            continue
        endpoint = {world: float(metrics[world]["source_equal_mrr"]) for world in PRIMARY_WORLDS}
        rows.append({
            "deleted_source_family": source,
            "status": "complete",
            "remaining_query_sources": int(metrics["all"]["query_source_families"]),
            "delta_all_vs_strongest_known": endpoint["all"] - max(endpoint[w] for w in KNOWN_CONTROLS),
            "delta_all_vs_strongest_full_reassignment": endpoint["all"] - max(
                endpoint[w] for w in FULL_REASSIGN_CONTROLS
            ),
            "delta_matched_unresolved_vs_strongest_reassignment": (
                endpoint["measurement_matched_unresolved"]
                - max(endpoint[w] for w in MATCHED_UNRESOLVED_REASSIGN_CONTROLS)
            ),
        })
    return pd.DataFrame(rows)


def adjudicate(
    matrix: pd.DataFrame,
    topology: pd.DataFrame,
    truth: pd.DataFrame,
    draws: int = 9999,
    seed: int = 20261018,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotated, validated_truth = validate_and_annotate(matrix, topology, truth)
    primary_truth = validated_truth.loc[
        validated_truth.eligible_primary & validated_truth.truth_class.eq("replication_expected")
    ]
    coverage = pd.DataFrame([{
        "matrix_ordered_pairs": int(annotated[
            ["left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id"]
        ].drop_duplicates().shape[0]),
        "explicit_truth_pairs": int(len(validated_truth)),
        "primary_replication_pairs": int(len(primary_truth)),
        "primary_truth_families": int(primary_truth.truth_family_id.nunique()),
        "primary_query_sources": int(annotated.loc[
            annotated.eligible_primary, "query_source_family"
        ].nunique()),
        "unlabeled_ordered_pairs": int(annotated.loc[
            annotated.truth_class.eq("unknown"),
            ["left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id"],
        ].drop_duplicates().shape[0]),
    }])
    if primary_truth.empty:
        report = {
            "schema": "pan_redu_partial_label_response_retrieval_adjudication_v1",
            "status": "ABSTAIN_NO_CLAIM_BEARING_GOLD_TRUTH",
            "formal": False,
            "denominators": coverage.iloc[0].to_dict(),
            "pass_systemic_unknown_increment": False,
            "reason": (
                "No score-blind, source-independent replication truth is eligible. "
                "Unlisted pairs remain unknown and cannot be converted to negatives."
            ),
        }
        return report, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), coverage

    outcomes = build_query_world_outcomes(annotated)
    metrics = source_equal_metrics(outcomes)
    bootstrap, replicates = source_bootstrap(outcomes, draws, seed)
    delete_one = delete_one_source_node(annotated)
    complete_delete = delete_one.loc[delete_one.status.eq("complete")]
    intervals = bootstrap.get("intervals", {}) if bootstrap.get("status") == "SOURCE_FAMILY_BOOTSTRAP_COMPLETE" else {}
    systemic_denominators = {
        "truth_families": int(primary_truth.truth_family_id.nunique()),
        "query_source_families": int(outcomes.query_source_family.nunique()),
        "primary_queries": int(outcomes[["query_panel_id", "query_contrast_id"]].drop_duplicates().shape[0]),
    }

    def lower_positive(name: str) -> bool:
        return bool(name in intervals and float(intervals[name]["lower_95"]) > 0)

    delete_columns = [
        "delta_all_vs_strongest_known",
        "delta_all_vs_strongest_full_reassignment",
        "delta_matched_unresolved_vs_strongest_reassignment",
    ]
    delete_consistent = bool(
        len(complete_delete) >= 3
        and all((pd.to_numeric(complete_delete[column], errors="coerce") > 0).all() for column in delete_columns)
    )
    gates = {
        "at_least_six_truth_families": systemic_denominators["truth_families"] >= 6,
        "at_least_six_query_source_families": systemic_denominators["query_source_families"] >= 6,
        "all_beats_strongest_known_ci": lower_positive("delta_all_vs_strongest_known"),
        "all_beats_endpoint_reassignment_ci": lower_positive("delta_all_vs_strongest_full_reassignment"),
        "matched_unresolved_beats_reassignment_ci": lower_positive(
            "delta_matched_unresolved_vs_strongest_reassignment"
        ),
        "conservative_risk_net_ci": lower_positive("conservative_risk_net"),
        "every_delete_one_source_delta_positive": delete_consistent,
    }
    report = {
        "schema": "pan_redu_partial_label_response_retrieval_adjudication_v1",
        "status": "PAN_REDU_PARTIAL_LABEL_ADJUDICATION_COMPLETE",
        "formal": True,
        "endpoint": "source-equal unconditional reciprocal rank; unsupported positive receives zero",
        "denominators": {**coverage.iloc[0].to_dict(), **systemic_denominators},
        "world_metrics": metrics,
        "source_family_bootstrap": bootstrap,
        "gates": gates,
        "pass_systemic_unknown_increment": all(gates.values()),
        "claim_boundary": [
            "Unknown candidate pairs may affect rank but are not called false biological relations.",
            "This gate establishes retrieval increment, not molecular identity or mechanism.",
            "Participant-level label nulls and effect re-estimation remain required for each claim-bearing family.",
        ],
    }
    return report, outcomes, replicates, delete_one, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--truth-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--allow-legacy-audit",
        action="store_true",
        help="reproduce the superseded v1 adjudication; never use for a formal claim",
    )
    args = parser.parse_args()
    if not args.allow_legacy_audit:
        raise RuntimeError(
            "LEGACY_AUDIT_ONLY: use build_pan_redu_exact_axis_source_outcomes.py "
            "then evaluate_contrast_atlas_source_risk.py for formal inference"
        )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bootstrap_draws < 499:
        raise ValueError("at least 499 source-family bootstrap draws are required")
    matrix = pd.read_csv(args.matrix, sep="\t", dtype={
        "left_panel_id": str, "left_contrast_id": str,
        "right_panel_id": str, "right_contrast_id": str,
    })
    topology = pd.read_csv(args.topology, sep="\t", dtype=str)
    truth = pd.read_csv(args.truth_registry, sep="\t", dtype=str)
    report, outcomes, replicates, delete_one, coverage = adjudicate(
        matrix, topology, truth, args.bootstrap_draws, args.seed,
    )
    args.output_dir.mkdir(parents=True)
    outcomes.to_csv(args.output_dir / "query_world_outcomes.tsv.gz", sep="\t", index=False, compression="gzip")
    replicates.to_csv(args.output_dir / "source_bootstrap_replicates.tsv.gz", sep="\t", index=False, compression="gzip")
    delete_one.to_csv(args.output_dir / "source_delete_one.tsv", sep="\t", index=False)
    coverage.to_csv(args.output_dir / "truth_coverage.tsv", sep="\t", index=False)
    report["provenance"] = {
        "matrix_sha256": sha256(args.matrix),
        "topology_sha256": sha256(args.topology),
        "truth_registry_sha256": sha256(args.truth_registry),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "denominators": report["denominators"],
        "pass_systemic_unknown_increment": report["pass_systemic_unknown_increment"],
    }, indent=2))


if __name__ == "__main__":
    main()
