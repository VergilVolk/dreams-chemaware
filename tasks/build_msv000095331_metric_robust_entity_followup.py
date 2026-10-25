#!/usr/bin/env python3
"""Prioritize response-attributed FullMSG identity changes by metric robustness.

The metric variants were fixed without identity labels or biological responses.
This post-response analysis does not choose a metric or create a new retrieval
method.  It asks whether entity changes that supported raw-FullMSG response
gains persist when the same frozen embedding is read with four label-free
geometries.  Both favorable and unfavorable edges are retained for follow-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "MSV000095331_METRIC_ROBUST_ENTITY_FOLLOWUP_COMPLETE"
RAW_FULLMSG = "metric_fullmsg_raw_cosine"
FULLMSG_METRICS = (
    "metric_fullmsg_raw_cosine",
    "metric_fullmsg_study_centered_cosine",
    "metric_fullmsg_diagonal_whitened_cosine",
    "metric_fullmsg_simhash256",
)
ATTRIBUTION_METHOD = "noise_chemaware_fullmsg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attribution-dir",
        type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_multiencoder_response_gain_attribution_v3_20261025"
        ),
    )
    parser.add_argument(
        "--metric-increment-dir",
        type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_entity_method_increment_metric_variants_v1_20261025"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    edge_path = args.attribution_dir / "top1_gain_edge_attribution.tsv.gz"
    context_path = args.attribution_dir / "top1_gain_contexts.tsv"
    decision_path = args.metric_increment_dir / "entity_method_decisions.tsv.gz"
    overlap_path = args.metric_increment_dir / "prespecified_overlap_results.tsv"

    edges = pd.read_csv(edge_path, sep="\t", dtype={"candidate_pair_id": str})
    edges = edges.loc[
        edges["method_name"].eq(ATTRIBUTION_METHOD)
        & edges["edge_state_vs_noise"].ne("retained")
    ].copy()
    contexts = pd.read_csv(context_path, sep="\t")
    contexts = contexts.loc[contexts["method_name"].eq(ATTRIBUTION_METHOD)].copy()

    decisions = pd.read_csv(
        decision_path,
        sep="\t",
        dtype={"selected_candidate_pair_id": str},
        usecols=[
            "method_name", "query_study_id", "candidate_study_id",
            "selected_candidate_pair_id", "mutual_top1",
        ],
    )
    decisions = decisions.loc[
        decisions["method_name"].isin(FULLMSG_METRICS)
        & decisions["query_study_id"].eq("MSV000095331")
        & bool_series(decisions["mutual_top1"])
        & decisions["selected_candidate_pair_id"].notna()
    ].copy()
    selected_by_metric = {
        method: set(block["selected_candidate_pair_id"].astype(str))
        for method, block in decisions.groupby("method_name", sort=False)
    }

    metric_columns: list[str] = []
    for method in FULLMSG_METRICS:
        column = "mutual_" + method.removeprefix("metric_fullmsg_")
        metric_columns.append(column)
        edges[column] = edges["candidate_pair_id"].isin(
            selected_by_metric.get(method, set())
        )
    edges["fullmsg_metric_selection_count"] = edges[metric_columns].sum(axis=1)
    edges["fullmsg_metric_action_support_count"] = np.where(
        edges["edge_state_vs_noise"].eq("gained"),
        edges["fullmsg_metric_selection_count"],
        len(FULLMSG_METRICS) - edges["fullmsg_metric_selection_count"],
    ).astype(int)
    edges["absolute_delta_response_numerator"] = edges[
        "delta_response_numerator"
    ].abs()

    # Preserve both sides of every gain: five most robust/largest helpful and
    # five most robust/largest harmful actions per context.
    shortlist = edges.sort_values(
        ["fullmsg_metric_action_support_count", "absolute_delta_response_numerator"],
        ascending=[False, False],
        kind="mergesort",
    ).groupby(
        [
            "query_contrast_id", "candidate_study_id", "candidate_contrast_id",
            "delta_role",
        ],
        sort=False,
    ).head(5)

    overlaps = pd.read_csv(overlap_path, sep="\t")
    overlaps = overlaps.loc[
        overlaps["method_name"].isin(FULLMSG_METRICS)
        & overlaps["prespecified_overlap"].eq(True)
    ].copy()
    context_rows: list[dict] = []
    for context in contexts.to_dict("records"):
        mask = (
            overlaps["query_contrast_id"].eq(context["query_contrast_id"])
            & overlaps["candidate_study_id"].eq(context["candidate_study_id"])
            & overlaps["candidate_contrast_id"].eq(context["candidate_contrast_id"])
        )
        block = overlaps.loc[mask].set_index("method_name")
        row = dict(context)
        ranks: list[float] = []
        for method in FULLMSG_METRICS:
            if method not in block.index:
                raise RuntimeError(f"missing metric result for {method}: {context}")
            rank = float(block.loc[method, "candidate_rank"])
            name = method.removeprefix("metric_fullmsg_") + "_rank"
            row[name] = rank
            ranks.append(rank)
        row["fullmsg_metric_top1_count"] = int(np.equal(ranks, 1.0).sum())
        row["fullmsg_metric_median_rank"] = float(np.median(ranks))
        context_rows.append(row)
    context_robustness = pd.DataFrame(context_rows)

    action_support = (
        edges.groupby(
            ["delta_role", "fullmsg_metric_action_support_count"], sort=True
        ).size().rename("changed_edges").reset_index()
    )

    consistency = edges.groupby("candidate_pair_id").agg(
        edge_states=("edge_state_vs_noise", "nunique"),
        action_support_values=("fullmsg_metric_action_support_count", "nunique"),
    )
    if not consistency["edge_states"].eq(1).all():
        raise RuntimeError("an identity action changed state between response contexts")
    if not consistency["action_support_values"].eq(1).all():
        raise RuntimeError("metric support changed for the same identity action")

    identity_columns = [
        "candidate_pair_id", "query_id", "query_raw_entity_id", "candidate_id",
        "candidate_raw_entity_id", "edge_state_vs_noise", "query_identity_status",
        "candidate_identity_status", "identity_evidence_class", "query_precursor_mz",
        "candidate_precursor_mz", "precursor_ppm_difference", "wse_score",
        "wse_matched_fragments", "published_product_target",
        "fullmsg_metric_selection_count", "fullmsg_metric_action_support_count",
        *metric_columns,
    ]
    unique_pairs = edges[identity_columns].drop_duplicates("candidate_pair_id").copy()
    pair_impact = edges.groupby("candidate_pair_id", sort=False).agg(
        response_context_instances=("query_contrast_id", "size"),
        helpful_context_instances=("delta_response_numerator", lambda x: int((x > 0).sum())),
        harmful_context_instances=("delta_response_numerator", lambda x: int((x < 0).sum())),
        maximum_absolute_response_contribution=(
            "absolute_delta_response_numerator", "max"
        ),
    ).reset_index()
    pair_contexts = edges.groupby("candidate_pair_id", sort=False).apply(
        lambda block: ";".join(sorted({
            f"{row.query_contrast_id}|{row.candidate_study_id}|{row.candidate_contrast_id}"
            for row in block.itertuples()
        })),
        include_groups=False,
    ).rename("response_contexts").reset_index()
    unique_pairs = unique_pairs.merge(
        pair_impact, on="candidate_pair_id", how="left", validate="one_to_one"
    ).merge(
        pair_contexts, on="candidate_pair_id", how="left", validate="one_to_one"
    )
    unique_pairs["response_contribution_class"] = np.select(
        [
            unique_pairs["harmful_context_instances"].eq(0),
            unique_pairs["helpful_context_instances"].eq(0),
        ],
        ["helpful_only", "harmful_only"],
        default="context_dependent",
    )
    unique_pairs = unique_pairs.sort_values(
        ["fullmsg_metric_action_support_count", "maximum_absolute_response_contribution"],
        ascending=[False, False], kind="mergesort",
    )

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    outputs = {
        "gain_context_robustness": args.output_dir / "gain_context_metric_robustness.tsv",
        "changed_edge_robustness": args.output_dir / "changed_edge_metric_robustness.tsv.gz",
        "followup_shortlist": args.output_dir / "orthogonal_followup_shortlist.tsv",
        "action_support_distribution": args.output_dir / "action_support_distribution.tsv",
        "unique_pair_registry": args.output_dir / "unique_identity_pair_followup_registry.tsv",
    }
    context_robustness.to_csv(outputs["gain_context_robustness"], sep="\t", index=False)
    edges.to_csv(
        outputs["changed_edge_robustness"],
        sep="\t",
        index=False,
        compression=compression,
    )
    shortlist.to_csv(outputs["followup_shortlist"], sep="\t", index=False)
    action_support.to_csv(outputs["action_support_distribution"], sep="\t", index=False)
    unique_pairs.to_csv(outputs["unique_pair_registry"], sep="\t", index=False)

    report = {
        "status": STATUS,
        "dataset_role": "development_consumed_not_external_validation",
        "analysis_role": "post_response_robustness_and_followup_not_metric_selection",
        "fullmsg_metrics": list(FULLMSG_METRICS),
        "denominators": {
            "raw_fullmsg_top1_gain_contexts": int(len(context_robustness)),
            "raw_fullmsg_changed_edge_context_instances": int(len(edges)),
            "raw_fullmsg_unique_changed_identity_pairs": int(
                edges["candidate_pair_id"].nunique()
            ),
            "helpful_changed_edges": int(edges["delta_role"].eq("supports_gain").sum()),
            "harmful_changed_edges": int(edges["delta_role"].eq("opposes_gain").sum()),
            "changed_actions_supported_by_all_four_metrics": int(
                edges["fullmsg_metric_action_support_count"].eq(4).sum()
            ),
            "unique_changed_identity_pairs_supported_by_all_four_metrics": int(
                edges.loc[
                    edges["fullmsg_metric_action_support_count"].eq(4),
                    "candidate_pair_id",
                ].nunique()
            ),
            "helpful_actions_supported_by_all_four_metrics": int(
                (
                    edges["delta_role"].eq("supports_gain")
                    & edges["fullmsg_metric_action_support_count"].eq(4)
                ).sum()
            ),
            "harmful_actions_supported_by_all_four_metrics": int(
                (
                    edges["delta_role"].eq("opposes_gain")
                    & edges["fullmsg_metric_action_support_count"].eq(4)
                ).sum()
            ),
            "shortlisted_edges": int(len(shortlist)),
        },
        "gain_context_top1_support": [
            {
                "query_contrast_id": str(row["query_contrast_id"]),
                "candidate_study_id": str(row["candidate_study_id"]),
                "candidate_contrast_id": str(row["candidate_contrast_id"]),
                "top1_metrics": int(row["fullmsg_metric_top1_count"]),
                "metrics_total": len(FULLMSG_METRICS),
            }
            for row in context_robustness.to_dict("records")
        ],
        "inputs": {
            "edge_attribution": {"path": str(edge_path), "sha256": sha256(edge_path)},
            "gain_contexts": {"path": str(context_path), "sha256": sha256(context_path)},
            "metric_decisions": {
                "path": str(decision_path), "sha256": sha256(decision_path)
            },
            "metric_overlap_results": {
                "path": str(overlap_path), "sha256": sha256(overlap_path)
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "Metric variants were fixed label-free and are not selected by this analysis.",
            "Robust identity actions remain unconfirmed identity hypotheses.",
            "Biological responses are used only for frozen post-hoc attribution and prioritization.",
            "Repeated microbial strata are contexts within one experiment, not replications.",
            "The shortlist includes helpful and harmful actions and cannot be used for model fitting.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
