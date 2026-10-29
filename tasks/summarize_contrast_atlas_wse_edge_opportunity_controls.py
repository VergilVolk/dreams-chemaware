#!/usr/bin/env python3
"""Aggregate full-panel reciprocal-edge-opportunity-matched WSE retrievals."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_contrast_atlas_complete_identity_wse_global import (
    CHEMPROP_STUDIES,
    IDENTITY_TRACK,
    canonical_key,
    records_without_nan,
    sha256,
)


SOURCE_INDEPENDENT = {
    "prespecified_exact_intervention_family",
    "prespecified_near_exact_condition_family",
}
EXACT_DRUG = {"consumed_exact_drug_cross_design"}


def subset_summary(
    registered: pd.DataFrame,
    relation_classes: set[str],
    world: str,
) -> dict[str, object]:
    selected = registered.loc[
        registered["evidence_track"].eq(IDENTITY_TRACK)
        & registered["evidence_world"].eq(world)
        & registered["relation_class"].isin(relation_classes)
    ].copy()
    selected["canonical_relation_key"] = canonical_key(selected)
    rank = pd.to_numeric(selected["best_face_rank"], errors="coerce")
    similar_rank = pd.to_numeric(selected["similar_rank"], errors="coerce")
    opposite_rank = pd.to_numeric(selected["opposite_rank"], errors="coerce")
    similar_p = pd.to_numeric(selected["similar_query_maxT_p"], errors="coerce")
    opposite_p = pd.to_numeric(selected["opposite_query_maxT_p"], errors="coerce")
    # The registered relation does not pre-specify a response face.  Each face
    # has a candidate-wise max-T p value, so choosing the smaller of two faces
    # requires an additional two-test correction.
    pvalue = (2.0 * pd.concat([similar_p, opposite_p], axis=1).min(axis=1)).clip(upper=1.0)
    canonical_rows: list[dict[str, object]] = []
    for _key, block in selected.groupby("canonical_relation_key", sort=False):
        pair_rank = pd.to_numeric(block["best_face_rank"], errors="coerce")
        block_similar_p = pd.to_numeric(block["similar_query_maxT_p"], errors="coerce")
        block_opposite_p = pd.to_numeric(block["opposite_query_maxT_p"], errors="coerce")
        pair_p = (
            2.0
            * pd.concat([block_similar_p, block_opposite_p], axis=1).min(axis=1)
        ).clip(upper=1.0)
        canonical_rows.append({
            "rank_any": float(pair_rank.min()) if pair_rank.notna().any() else np.nan,
            "rank_both": float(pair_rank.max()) if pair_rank.notna().all() else np.nan,
            "p_any": float(pair_p.min()) if pair_p.notna().any() else np.nan,
            "p_both": float(pair_p.max()) if pair_p.notna().all() else np.nan,
        })
    canonical = pd.DataFrame(canonical_rows)
    return {
        "world": world,
        "relation_classes": ";".join(sorted(relation_classes)),
        "canonical_relations": int(len(canonical)),
        "directional_relations": int(len(selected)),
        "directional_available": int(rank.notna().sum()),
        "directional_top1": int(rank.eq(1).sum()),
        "directional_top5": int(rank.le(5).sum()),
        "directional_similar_top5": int(similar_rank.le(5).sum()),
        "directional_opposite_top5": int(opposite_rank.le(5).sum()),
        "directional_median_rank": (
            np.nan if not rank.notna().any() else float(rank.median())
        ),
        "directional_best_face_bonferroni_005": int(pvalue.le(0.05).sum()),
        "directional_similar_face_maxT_005": int(similar_p.le(0.05).sum()),
        "directional_opposite_face_maxT_005": int(opposite_p.le(0.05).sum()),
        "canonical_any_top1": int(canonical.get("rank_any", pd.Series(dtype=float)).eq(1).sum()),
        "canonical_both_top1": int(canonical.get("rank_both", pd.Series(dtype=float)).eq(1).sum()),
        "canonical_any_top5": int(canonical.get("rank_any", pd.Series(dtype=float)).le(5).sum()),
        "canonical_both_top5": int(canonical.get("rank_both", pd.Series(dtype=float)).le(5).sum()),
        "canonical_any_best_face_bonferroni_005": int(
            canonical.get("p_any", pd.Series(dtype=float)).le(0.05).sum()
        ),
        "canonical_both_best_face_bonferroni_005": int(
            canonical.get("p_both", pd.Series(dtype=float)).le(0.05).sum()
        ),
    }


def open_summary(top: pd.DataFrame) -> list[dict[str, object]]:
    selected = top.loc[
        top["evidence_world"].eq("unresolved")
        & top["evidence_track"].eq(IDENTITY_TRACK)
    ].copy()
    rows = []
    for face, block in selected.groupby("response_face", sort=True):
        called = block["result_class"].eq(f"{face}_response")
        query_chemprop = block["query_study_id"].isin(CHEMPROP_STUDIES)
        candidate_chemprop = block["candidate_study_id"].isin(CHEMPROP_STUDIES)
        rows.append({
            "response_face": str(face),
            "queries": int(len(block)),
            "calibrated_calls": int(called.sum()),
            "no_calibrated_match": int(block["result_class"].eq("no_calibrated_match").sum()),
            "insufficient_evidence": int(
                block["result_class"].eq("abstain_insufficient_evidence").sum()
            ),
            "chemprop_candidate_hits": int(candidate_chemprop.sum()),
            "nonchemprop_to_chemprop_hits": int((~query_chemprop & candidate_chemprop).sum()),
            "chemprop_to_chemprop_hits": int((query_chemprop & candidate_chemprop).sum()),
            "unique_candidate_parent_projects": int(
                block["candidate_study_id"].map(
                    lambda value: "CHEMPROP_PARENT_PROJECT"
                    if str(value) in CHEMPROP_STUDIES else str(value)
                ).nunique()
            ),
            "largest_candidate_contrast_hub": int(
                block.groupby(["candidate_study_id", "candidate_contrast_id"]).size().max()
            ),
        })
    return rows


def distribution(frame: pd.DataFrame, group: str, metric: str) -> dict[str, object]:
    values = pd.to_numeric(
        frame.loc[frame["analysis_group"].eq(group), metric], errors="coerce"
    ).dropna()
    return {
        "analysis_group": group,
        "metric": metric,
        "resamples": int(len(values)),
        "minimum": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--retrieval-root", required=True, type=Path)
    parser.add_argument("--full-retrieval-dir", required=True, type=Path)
    parser.add_argument("--full-global-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
    parser.add_argument("--permutation-seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.doc.exists():
        raise FileExistsError(args.doc)

    control_report_path = args.control_root / "report.json"
    control_manifest_path = args.control_root / "resample_manifest.tsv"
    control_report = json.loads(control_report_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(control_manifest_path, sep="\t")
    expected = int(control_report["resamples"])
    if len(manifest) != expected:
        raise RuntimeError("control manifest/resample denominator mismatch")

    edge_sets: list[set[tuple[str, str, str, str]]] = []
    for replicate in range(expected):
        edge_path = args.control_root / f"resample_{replicate:02d}" / "source_pair_frozen_edges.tsv.gz"
        edges = pd.read_csv(edge_path, sep="\t", dtype=str)
        edge_sets.append(set(
            edges[["left_study_id", "right_study_id", "left_entity_id", "right_entity_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ))
    common_edges = set.intersection(*edge_sets)
    pairwise_jaccard = np.asarray([
        len(left & right) / len(left | right)
        for left, right in combinations(edge_sets, 2)
    ])

    relation_rows: list[dict[str, object]] = []
    open_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    for replicate in range(expected):
        directory = args.retrieval_root / f"resample_{replicate:02d}"
        paths = {
            "report": directory / "report.json",
            "registered": directory / "registered_relation_ranks.tsv",
            "top": directory / "open_local_edge_top_hits.tsv",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        denominator = report["denominators"]
        if int(denominator["canonical_mutual_edges"]) != int(
            control_report["known_reciprocal_edges"]
        ):
            raise RuntimeError(f"matched edge count drift in replicate {replicate}")
        if int(denominator["directed_query_candidate_pairs"]) != 2286:
            raise RuntimeError("open candidate universe drift")
        if int(denominator["permutations"]) != 256:
            raise RuntimeError("response null denominator drift")
        registered = pd.read_csv(paths["registered"], sep="\t", low_memory=False)
        top = pd.read_csv(paths["top"], sep="\t", low_memory=False)
        for label, classes in (
            ("source_independent_primary", SOURCE_INDEPENDENT),
            ("shared_project_exact_drug", EXACT_DRUG),
        ):
            row = subset_summary(registered, classes, "unresolved")
            row.update({"replicate": replicate, "analysis_group": label})
            relation_rows.append(row)
        for row in open_summary(top):
            row.update({"replicate": replicate})
            open_rows.append(row)
        provenance_rows.append({
            "replicate": replicate,
            "report_sha256": sha256(paths["report"]),
            "registered_sha256": sha256(paths["registered"]),
            "top_sha256": sha256(paths["top"]),
        })

    relations = pd.DataFrame(relation_rows)
    open_frame = pd.DataFrame(open_rows)
    provenance = pd.DataFrame(provenance_rows)

    full_registered = pd.read_csv(
        args.full_retrieval_dir / "registered_relation_ranks.tsv",
        sep="\t", low_memory=False,
    )
    full_top = pd.read_csv(
        args.full_retrieval_dir / "open_local_edge_top_hits.tsv",
        sep="\t", low_memory=False,
    )
    full_edges = pd.read_csv(
        args.full_retrieval_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
        sep="\t", low_memory=False,
    )
    known_edges = full_edges.loc[
        full_edges["evidence_world"].eq("known")
        & full_edges["precursor_compatible_10ppm"].astype(bool)
    ].copy()
    known_endpoint_counts = pd.concat([
        known_edges[["left_study_id", "left_entity_id"]].rename(
            columns={"left_study_id": "study", "left_entity_id": "entity"}
        ),
        known_edges[["right_study_id", "right_entity_id"]].rename(
            columns={"right_study_id": "study", "right_entity_id": "entity"}
        ),
    ]).value_counts()
    full_reference = []
    for label, classes in (
        ("source_independent_primary", SOURCE_INDEPENDENT),
        ("shared_project_exact_drug", EXACT_DRUG),
    ):
        for world in ("known", "unresolved"):
            row = subset_summary(full_registered, classes, world)
            row.update({"analysis_group": label})
            full_reference.append(row)

    metrics = [
        "directional_top1", "directional_top5",
        "directional_similar_top5", "directional_opposite_top5",
        "directional_best_face_bonferroni_005",
        "directional_similar_face_maxT_005", "directional_opposite_face_maxT_005",
        "canonical_any_top1", "canonical_both_top1",
        "canonical_any_top5", "canonical_both_top5",
        "canonical_any_best_face_bonferroni_005",
        "canonical_both_best_face_bonferroni_005",
    ]
    distributions = pd.DataFrame([
        distribution(relations, group, metric)
        for group in ("source_independent_primary", "shared_project_exact_drug")
        for metric in metrics
    ])
    open_distributions = (
        open_frame.groupby("response_face", sort=True)[[
            "calibrated_calls", "no_calibrated_match", "insufficient_evidence",
            "chemprop_candidate_hits", "nonchemprop_to_chemprop_hits",
            "chemprop_to_chemprop_hits", "unique_candidate_parent_projects",
            "largest_candidate_contrast_hub",
        ]]
        .agg(["min", "median", "max"])
    )
    open_distributions.columns = ["_".join(column) for column in open_distributions.columns]
    open_distributions = open_distributions.reset_index()

    args.output_dir.mkdir(parents=True)
    outputs = {
        "relations": args.output_dir / "per_resample_registered_summary.tsv",
        "relation_distributions": args.output_dir / "registered_metric_distributions.tsv",
        "open": args.output_dir / "per_resample_open_summary.tsv",
        "open_distributions": args.output_dir / "open_metric_distributions.tsv",
        "full_reference": args.output_dir / "full_world_reference.tsv",
        "provenance": args.output_dir / "retrieval_provenance.tsv",
    }
    relations.to_csv(outputs["relations"], sep="\t", index=False)
    distributions.to_csv(outputs["relation_distributions"], sep="\t", index=False)
    open_frame.to_csv(outputs["open"], sep="\t", index=False)
    open_distributions.to_csv(outputs["open_distributions"], sep="\t", index=False)
    pd.DataFrame(full_reference).to_csv(outputs["full_reference"], sep="\t", index=False)
    provenance.to_csv(outputs["provenance"], sep="\t", index=False)

    def metric_record(group: str, metric: str) -> dict[str, object]:
        return distributions.loc[
            distributions["analysis_group"].eq(group)
            & distributions["metric"].eq(metric)
        ].iloc[0].to_dict()

    full_global = json.loads(args.full_global_report.read_text(encoding="utf-8"))
    report = {
        "schema": "contrast_atlas_wse_reciprocal_edge_opportunity_matched_summary_v1",
        "status": "DEVELOPMENT_CONSUMED_EDGE_MATCHED_GLOBAL_RETRIEVAL_SUMMARIZED",
        "data_role": "development_consumed",
        "scale": {
            **full_global["scale"],
            "selection_resamples": expected,
            "response_permutations_per_resample": 256,
            "response_permutation_seed": args.permutation_seed,
            "selected_unresolved_reciprocal_edges_per_resample": int(
                control_report["selected_unresolved_reciprocal_edges_per_resample"]
            ),
            "strict_known_reciprocal_edges": int(control_report["known_reciprocal_edges"]),
            "full_unresolved_reciprocal_edges": int(
                control_report["unresolved_reciprocal_edges_available"]
            ),
        },
        "matching_quality": {
            "exact_strata": int(control_report["exact_strata"]),
            "strata_with_shortage": int(control_report["strata_with_shortage"]),
            "maximum_absolute_smd_range": control_report["maximum_absolute_smd_range"],
            "largest_residual_imbalances": {
                "conservative_wse_smd_median": float(
                    pd.read_csv(args.control_root / "covariate_balance.tsv", sep="\t")
                    .loc[lambda x: x["covariate"].eq("conservative_wse"), "standardized_mean_difference"]
                    .median()
                ),
                "minimum_matched_fragments_smd_median": float(
                    pd.read_csv(args.control_root / "covariate_balance.tsv", sep="\t")
                    .loc[lambda x: x["covariate"].eq("log1p_min_matched_fragments"), "standardized_mean_difference"]
                    .median()
                ),
            },
            "graph_topology_not_exactly_matched": {
                "known_unique_endpoints": int(len(known_endpoint_counts)),
                "known_maximum_endpoint_degree": int(known_endpoint_counts.max()),
                "selected_unresolved_unique_endpoints_range": control_report[
                    "selected_unique_endpoints_range"
                ],
                "selected_unresolved_maximum_endpoint_degree_range": control_report[
                    "maximum_selected_endpoint_degree_range"
                ],
            },
            "selection_resample_dependence": {
                "edges_present_in_all_resamples": int(len(common_edges)),
                "fraction_of_each_graph_present_in_all_resamples": float(
                    len(common_edges) / control_report["known_reciprocal_edges"]
                ),
                "pairwise_jaccard_minimum": float(pairwise_jaccard.min()),
                "pairwise_jaccard_median": float(np.median(pairwise_jaccard)),
                "pairwise_jaccard_maximum": float(pairwise_jaccard.max()),
                "interpretation": (
                    "32 order-sensitivity selections from one frozen pool; not independent replicates and ranges are not confidence intervals"
                ),
            },
        },
        "matched_unresolved_distributions": {
            "source_independent_primary_directional_top1": metric_record(
                "source_independent_primary", "directional_top1"
            ),
            "source_independent_primary_directional_top5": metric_record(
                "source_independent_primary", "directional_top5"
            ),
            "source_independent_primary_directional_maxT_005": metric_record(
                "source_independent_primary", "directional_best_face_bonferroni_005"
            ),
            "source_independent_primary_canonical_any_top5": metric_record(
                "source_independent_primary", "canonical_any_top5"
            ),
            "shared_project_exact_drug_directional_top5": metric_record(
                "shared_project_exact_drug", "directional_top5"
            ),
            "shared_project_exact_drug_directional_maxT_005": metric_record(
                "shared_project_exact_drug", "directional_best_face_bonferroni_005"
            ),
            "shared_project_exact_drug_similar_top5": metric_record(
                "shared_project_exact_drug", "directional_similar_top5"
            ),
            "shared_project_exact_drug_opposite_top5": metric_record(
                "shared_project_exact_drug", "directional_opposite_top5"
            ),
            "shared_project_exact_drug_canonical_any_top5": metric_record(
                "shared_project_exact_drug", "canonical_any_top5"
            ),
        },
        "full_world_reference": records_without_nan(pd.DataFrame(full_reference)),
        "open_retrieval_distributions": records_without_nan(open_distributions),
        "decision": {
            "large_scale_opportunity_matched_control_executed": True,
            "unknown_information_increment_pass": False,
            "source_independent_biological_retrieval_pass": False,
            "reasons": [
                "The registered set contains no signed entity-level truth and unregistered pairs are unlabeled, not negative.",
                "Registered relations do not pre-specify similar versus opposite response; unsigned best-face p values receive a two-face Bonferroni correction and best-face ranks remain descriptive.",
                "The matched unresolved graph is equal in reciprocal-edge count and exact source/compatibility strata but retains residual WSE and fragment-quality imbalance.",
                "Endpoint reuse and graph topology are not equal between known and matched unresolved controls.",
                "The 32 selected graphs overlap heavily and quantify matching-order sensitivity rather than independent replication.",
                "The five source-independent HFD/IBD relations are too few to establish general biological specificity.",
                "All sources and relation labels are development/consumed; no sealed external biological validation is present.",
                "Official DreaMS, Noise V1 and Full23 remain unevaluated on this identical complete identity axis.",
            ],
        },
        "claim_boundary": [
            "Each of 32 control graphs contains the same number of reciprocal identity edges as the strict-known graph and the same exact source-pair/compatibility/contrast-count strata.",
            "The 32 graphs are highly overlapping sensitivity selections, not 32 independent biological or statistical replicates; their ranges are not confidence intervals.",
            "This is a system-wide sensitivity analysis over 15 sources and 51 contrasts, not a small disease case study.",
            "The control does not match platform, peak count, TIC, retention time, ion-family or chemical-class coverage.",
            "Response-profile permutation significance is not biological truth, mechanism validation, identity FDR or no-match accuracy.",
            "Best-face Top-k is unsigned relevance retrieval, not recovery of a pre-specified response direction.",
        ],
        "provenance": {
            "control_report": sha256(control_report_path),
            "control_manifest": sha256(control_manifest_path),
            "full_global_report": sha256(args.full_global_report),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    source = report["matched_unresolved_distributions"]
    independent_top5 = source["source_independent_primary_directional_top5"]
    independent_max_t = source["source_independent_primary_directional_maxT_005"]
    exact_top5 = source["shared_project_exact_drug_directional_top5"]
    exact_max_t = source["shared_project_exact_drug_directional_maxT_005"]
    exact_similar_top5 = source["shared_project_exact_drug_similar_top5"]
    exact_opposite_top5 = source["shared_project_exact_drug_opposite_top5"]
    open_similar = next(
        row for row in report["open_retrieval_distributions"]
        if row["response_face"] == "similar"
    )
    open_opposite = next(
        row for row in report["open_retrieval_distributions"]
        if row["response_face"] == "opposite"
    )
    document = f"""# System-wide edge-opportunity-matched WSE Contrast Atlas control

## Scope

This is a full-panel development analysis: 15 sources, 13 parent projects, 51 study-level contrasts, 238,412 response rows, 54,861 study-entities and 2,286 directed cross-source contrast candidates. It is not an IBD case study.

The full unresolved graph contained {control_report['unresolved_reciprocal_edges_available']:,} reciprocal WSE identity edges versus {control_report['known_reciprocal_edges']:,} strict-known edges. Thirty-two response-blind controls therefore selected exactly {control_report['known_reciprocal_edges']:,} unresolved reciprocal edges in each selection, with exact matching across {control_report['exact_strata']} source-pair, physicochemical-compatibility and response-availability strata. Each graph was evaluated separately with the same frozen 256 whole-profile permutations (seed {args.permutation_seed}) and query-level max-T correction over all 2,286 candidate contrasts.

These are matching-order sensitivity selections, not independent replicates: {len(common_edges):,}/{control_report['known_reciprocal_edges']:,} edges occurred in all 32 graphs, and pairwise Jaccard overlap was {pairwise_jaccard.min():.3f}-{pairwise_jaccard.max():.3f} (median {np.median(pairwise_jaccard):.3f}). Ranges below are therefore sensitivity ranges, not confidence intervals.

## Result

- Source-independent HFD/IBD primary directions (10 directions from five canonical relations): unsigned best-face Top-5 ranged **{int(independent_top5['minimum'])}-{int(independent_top5['maximum'])}**, median **{independent_top5['median']:.1f}** across 32 resamples.
- None of those 10 source-independent directions passed the two-face-Bonferroni query-level max-T criterion in any resample: range **{int(independent_max_t['minimum'])}-{int(independent_max_t['maximum'])}**, median **{independent_max_t['median']:.1f}**. The primary signal therefore does not survive edge-opportunity matching under the corrected unsigned test.
- Shared-parent ChemProp exact-drug positive controls (24 directions): unsigned best-face Top-5 ranged **{int(exact_top5['minimum'])}-{int(exact_top5['maximum'])}**, median **{exact_top5['median']:.1f}**. Similar-face Top-5 ranged **{int(exact_similar_top5['minimum'])}-{int(exact_similar_top5['maximum'])}** and opposite-face Top-5 ranged **{int(exact_opposite_top5['minimum'])}-{int(exact_opposite_top5['maximum'])}**; because response face was not registered, these are relevance diagnostics rather than signed response recovery. Two-face-Bonferroni max-T calls ranged **{int(exact_max_t['minimum'])}-{int(exact_max_t['maximum'])}**, median **{exact_max_t['median']:.1f}**. These remain consumed within-project controls, not external replication.
- Open similar-response calibrated calls fell from 49/51 in the complete unresolved graph to **{int(open_similar['calibrated_calls_min'])}-{int(open_similar['calibrated_calls_max'])}** (median **{open_similar['calibrated_calls_median']:.1f}**); opposite calls fell from 46/51 to **{int(open_opposite['calibrated_calls_min'])}-{int(open_opposite['calibrated_calls_max'])}** (median **{open_opposite['calibrated_calls_median']:.1f}**).
- ChemProp remained a response hub after edge-count matching: median Top hits into its three accessions were **{open_similar['chemprop_candidate_hits_median']:.1f}/51** for similar and **{open_opposite['chemprop_candidate_hits_median']:.1f}/51** for opposite retrieval.

The control removes the 35-fold reciprocal-edge-count advantage, but does not create a formal unknown-information PASS. Residual evidence-quality imbalance remains: median standardized mean differences were {report['matching_quality']['largest_residual_imbalances']['conservative_wse_smd_median']:.3f} for conservative WSE and {report['matching_quality']['largest_residual_imbalances']['minimum_matched_fragments_smd_median']:.3f} for minimum matched fragments. Endpoint reuse also remains different: the known graph uses {report['matching_quality']['graph_topology_not_exactly_matched']['known_unique_endpoints']:,} unique endpoints, while matched unresolved controls use {report['matching_quality']['graph_topology_not_exactly_matched']['selected_unresolved_unique_endpoints_range'][0]:,}-{report['matching_quality']['graph_topology_not_exactly_matched']['selected_unresolved_unique_endpoints_range'][1]:,}. The strict-known versus unresolved status defines the two pools, although no molecule name or specific structure identifier value enters the matching distance. More importantly, the registered set contains zero signed entity-level truths, unregistered pairs are not negatives, and only five canonical primary relations are source-independent.

## Scientific verdict

The large-scale Atlas computation is now real and system-wide. It shows that broad unresolved evidence can support structured response retrieval even after edge-count matching, but current assets do not establish that those additional answers are biologically correct or specific. The present bottleneck is not graph size; it is independent response truth and same-axis scoring by Official DreaMS, Noise V1 and Full23.

Accordingly:

- large-scale opportunity-matched retrieval: **executed**;
- source-independent biological retrieval: **not passed**;
- unresolved-information increment: **not passed**;
- in-house algorithm increment on the same complete axis: **not yet evaluated**;
- sealed external biological validation: **absent**.

All inputs are development/consumed. Response max-T significance is not mechanism truth, identity FDR, specificity or no-match accuracy.
"""
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document, encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
