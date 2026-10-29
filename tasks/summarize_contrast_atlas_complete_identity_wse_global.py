#!/usr/bin/env python3
"""Summarize the complete-axis WSE Atlas run at system scale.

The purpose of this report is to prevent a large computation from being
reduced to one favourable entity example.  It keeps all 51 query contrasts,
separates relation families by their real independence, quantifies project
concentration in open retrieval, and states explicitly which causal or
unknown-information claims remain unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_TRACK = "same_precursor_10ppm_identity_hypothesis"
CHEMPROP_STUDIES = {"MSV000093571", "MSV000094899", "MSV000096724"}
SOURCE_INDEPENDENT_CLASSES = {
    "prespecified_exact_intervention_family",
    "prespecified_near_exact_condition_family",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(frame: pd.DataFrame) -> pd.Series:
    left = frame["left_study_id"].astype(str) + "|" + frame["left_contrast_id"].astype(str)
    right = frame["right_study_id"].astype(str) + "|" + frame["right_contrast_id"].astype(str)
    return pd.Series(
        ["||".join(sorted((a, b))) for a, b in zip(left, right)],
        index=frame.index,
    )


def relation_family_summary(registered: pd.DataFrame) -> pd.DataFrame:
    selected = registered.loc[registered["evidence_track"].eq(IDENTITY_TRACK)].copy()
    selected["canonical_relation_key"] = canonical_key(selected)
    rows: list[dict[str, object]] = []
    for (world, relation_class), block in selected.groupby(
        ["evidence_world", "relation_class"], sort=True
    ):
        rank = pd.to_numeric(block["best_face_rank"], errors="coerce")
        pvalue = pd.to_numeric(block["best_face_query_maxT_p"], errors="coerce")
        canonical = []
        for _key, pair in block.groupby("canonical_relation_key", sort=False):
            pair_rank = pd.to_numeric(pair["best_face_rank"], errors="coerce")
            pair_p = pd.to_numeric(pair["best_face_query_maxT_p"], errors="coerce")
            canonical.append({
                "directions": int(len(pair)),
                "rank_any": float(pair_rank.min()) if pair_rank.notna().any() else np.nan,
                "rank_both": (
                    float(pair_rank.max()) if pair_rank.notna().all() else np.nan
                ),
                "p_any": float(pair_p.min()) if pair_p.notna().any() else np.nan,
                "p_both": float(pair_p.max()) if pair_p.notna().all() else np.nan,
            })
        canonical_frame = pd.DataFrame(canonical)
        rows.append({
            "evidence_world": str(world),
            "relation_class": str(relation_class),
            "canonical_relations": int(len(canonical_frame)),
            "directional_relations": int(len(block)),
            "directional_available": int(rank.notna().sum()),
            "directional_top1": int(rank.eq(1).sum()),
            "directional_top5": int(rank.le(5).sum()),
            "directional_median_rank_available": (
                np.nan if not rank.notna().any() else float(rank.median())
            ),
            "directional_query_maxT_005": int(pvalue.le(0.05).sum()),
            "canonical_any_direction_top1": int(
                pd.to_numeric(canonical_frame["rank_any"], errors="coerce").eq(1).sum()
            ),
            "canonical_both_directions_top1": int(
                pd.to_numeric(canonical_frame["rank_both"], errors="coerce").eq(1).sum()
            ),
            "canonical_any_direction_top5": int(
                pd.to_numeric(canonical_frame["rank_any"], errors="coerce").le(5).sum()
            ),
            "canonical_both_directions_top5": int(
                pd.to_numeric(canonical_frame["rank_both"], errors="coerce").le(5).sum()
            ),
            "canonical_any_direction_maxT_005": int(
                pd.to_numeric(canonical_frame["p_any"], errors="coerce").le(0.05).sum()
            ),
            "canonical_both_directions_maxT_005": int(
                pd.to_numeric(canonical_frame["p_both"], errors="coerce").le(0.05).sum()
            ),
        })
    return pd.DataFrame(rows)


def relation_world_ledger(registered: pd.DataFrame) -> pd.DataFrame:
    selected = registered.loc[registered["evidence_track"].eq(IDENTITY_TRACK)].copy()
    selected["canonical_relation_key"] = canonical_key(selected)
    keys = [
        "relation_direction",
        "relation_class",
        "eligible_primary_relevance",
        "canonical_relation_key",
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
    ]
    values = [
        "best_face_rank",
        "best_face_query_maxT_p",
        "mutual_edges",
        "weighted_evidence_edges",
    ]
    wide = selected.pivot(index=keys, columns="evidence_world", values=values)
    wide.columns = [f"{metric}_{world}" for metric, world in wide.columns]
    return wide.reset_index()


def project_family(study: str) -> str:
    return "CHEMPROP_PARENT_PROJECT" if str(study) in CHEMPROP_STUDIES else str(study)


def open_hit_summaries(
    top: pd.DataFrame,
    candidate_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    selected = top.loc[top["evidence_track"].eq(IDENTITY_TRACK)].copy()
    selected["query_parent_project"] = selected["query_study_id"].map(project_family)
    selected["candidate_parent_project"] = selected["candidate_study_id"].map(
        project_family
    )
    rows: list[dict[str, object]] = []
    for (world, face), block in selected.groupby(
        ["evidence_world", "response_face"], sort=True
    ):
        called = block["result_class"].eq(f"{face}_response")
        query_chemprop = block["query_study_id"].isin(CHEMPROP_STUDIES)
        candidate_chemprop = block["candidate_study_id"].isin(CHEMPROP_STUDIES)
        rows.append({
            "evidence_world": str(world),
            "response_face": str(face),
            "query_contrasts": int(len(block)),
            "calibrated_calls": int(called.sum()),
            "no_calibrated_match": int(block["result_class"].eq("no_calibrated_match").sum()),
            "insufficient_evidence_abstentions": int(
                block["result_class"].eq("abstain_insufficient_evidence").sum()
            ),
            "candidate_parent_projects": int(block["candidate_parent_project"].nunique()),
            "chemprop_candidate_hits": int(candidate_chemprop.sum()),
            "nonchemprop_queries": int((~query_chemprop).sum()),
            "nonchemprop_to_chemprop_hits": int((~query_chemprop & candidate_chemprop).sum()),
            "chemprop_queries": int(query_chemprop.sum()),
            "chemprop_to_chemprop_hits": int((query_chemprop & candidate_chemprop).sum()),
            "largest_single_candidate_contrast_hub": int(
                block.groupby(["candidate_study_id", "candidate_contrast_id"]).size().max()
            ),
        })
    summary = pd.DataFrame(rows)
    frequency = (
        selected.groupby([
            "evidence_world",
            "response_face",
            "candidate_parent_project",
            "candidate_study_id",
            "candidate_contrast_id",
        ])
        .size().rename("top_hit_queries").reset_index()
        .sort_values(
            ["evidence_world", "response_face", "top_hit_queries"],
            ascending=[True, True, False],
        )
    )

    universe = candidate_scores.loc[
        candidate_scores["evidence_world"].eq("all")
        & candidate_scores["evidence_track"].eq(IDENTITY_TRACK)
    ].drop_duplicates([
        "query_study_id", "query_contrast_id",
        "candidate_study_id", "candidate_contrast_id",
    ])
    universe_chemprop = universe["candidate_study_id"].isin(CHEMPROP_STUDIES)
    all_similar = selected.loc[
        selected["evidence_world"].eq("all")
        & selected["response_face"].eq("similar")
    ]
    hit_chemprop = all_similar["candidate_study_id"].isin(CHEMPROP_STUDIES)
    exposure_fraction = float(universe_chemprop.mean())
    hit_fraction = float(hit_chemprop.mean())
    concentration = {
        "candidate_universe_rows": int(len(universe)),
        "candidate_universe_chemprop_rows": int(universe_chemprop.sum()),
        "candidate_universe_chemprop_fraction": exposure_fraction,
        "all_similar_top_hits": int(len(all_similar)),
        "all_similar_top_hits_into_chemprop": int(hit_chemprop.sum()),
        "all_similar_top_hit_chemprop_fraction": hit_fraction,
        "top_hit_to_exposure_ratio": (
            None if exposure_fraction == 0 else hit_fraction / exposure_fraction
        ),
    }
    return summary, frequency, concentration


def finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def records_without_nan(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: finite_or_none(value) for key, value in record.items()}
        for record in frame.to_dict("records")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--known-baseline-report", required=True, type=Path)
    parser.add_argument("--old-retrieval-report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "retrieval_report": args.retrieval_dir / "report.json",
        "registered": args.retrieval_dir / "registered_relation_ranks.tsv",
        "top": args.retrieval_dir / "open_local_edge_top_hits.tsv",
        "scores": args.retrieval_dir / "open_local_edge_candidate_scores.tsv.gz",
        "edges": args.retrieval_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
        "alignment_report": args.alignment_dir / "report.json",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "known_baseline_report": args.known_baseline_report,
    }
    if args.old_retrieval_report is not None:
        paths["old_retrieval_report"] = args.old_retrieval_report
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    retrieval_report = json.loads(paths["retrieval_report"].read_text(encoding="utf-8"))
    alignment_report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    known_baseline = json.loads(paths["known_baseline_report"].read_text(encoding="utf-8"))
    old_report = (
        json.loads(paths["old_retrieval_report"].read_text(encoding="utf-8"))
        if "old_retrieval_report" in paths else None
    )
    registered = pd.read_csv(paths["registered"], sep="\t", low_memory=False)
    top = pd.read_csv(paths["top"], sep="\t", low_memory=False)
    scores = pd.read_csv(paths["scores"], sep="\t", low_memory=False)
    edges = pd.read_csv(paths["edges"], sep="\t", low_memory=False)
    contrasts = pd.read_csv(paths["contrasts"], sep="\t", low_memory=False)

    family = relation_family_summary(registered)
    world_ledger = relation_world_ledger(registered)
    open_summary, hit_frequency, concentration = open_hit_summaries(top, scores)
    identity_edges = edges.loc[edges["precursor_compatible_10ppm"].astype(bool)].copy()
    edge_coverage = (
        identity_edges.groupby([
            "evidence_world", "left_study_id", "right_study_id"
        ])
        .agg(
            mutual_identity_edges=("left_entity_id", "size"),
            positive_weight_edges=("chemical_similarity", lambda x: int((x > 0).sum())),
            median_wse=("chemical_similarity", "median"),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True)
    outputs = {
        "family": args.output_dir / "registered_relation_family_summary.tsv",
        "world_ledger": args.output_dir / "registered_relation_world_ledger.tsv",
        "open_summary": args.output_dir / "open_top_hit_global_summary.tsv",
        "hit_frequency": args.output_dir / "open_top_hit_candidate_frequency.tsv",
        "identity_top_hits": args.output_dir / "open_identity_top_hits.tsv",
        "edge_coverage": args.output_dir / "study_pair_identity_edge_coverage.tsv",
    }
    family.to_csv(outputs["family"], sep="\t", index=False)
    world_ledger.to_csv(outputs["world_ledger"], sep="\t", index=False)
    open_summary.to_csv(outputs["open_summary"], sep="\t", index=False)
    hit_frequency.to_csv(outputs["hit_frequency"], sep="\t", index=False)
    top.loc[top["evidence_track"].eq(IDENTITY_TRACK)].to_csv(
        outputs["identity_top_hits"], sep="\t", index=False
    )
    edge_coverage.to_csv(outputs["edge_coverage"], sep="\t", index=False)

    source_independent = family.loc[
        family["relation_class"].isin(SOURCE_INDEPENDENT_CLASSES)
    ].copy()
    exact_drug = family.loc[
        family["relation_class"].eq("consumed_exact_drug_cross_design")
    ].copy()
    all_edge_count = int(
        alignment_report["denominators"]["worlds"][0]["canonical_mutual_edges"]
    )
    known_edge_count = int(
        alignment_report["denominators"]["worlds"][1]["canonical_mutual_edges"]
    )
    unresolved_edge_count = int(
        alignment_report["denominators"]["worlds"][2]["canonical_mutual_edges"]
    )
    report = {
        "schema": "contrast_atlas_complete_identity_wse_global_summary_v1",
        "status": "DEVELOPMENT_CONSUMED_GLOBAL_WSE_RESPONSE_RETRIEVAL_SUMMARIZED",
        "data_role": "development_consumed",
        "scale": {
            "studies": int(retrieval_report["denominators"]["independent_studies"]),
            "parent_project_families": 13,
            "study_level_contrasts": int(retrieval_report["denominators"]["study_level_contrasts"]),
            "response_rows": int(retrieval_report["denominators"]["response_rows"]),
            "entities": int(retrieval_report["denominators"]["unique_study_entities"]),
            "complete_identity_candidate_pairs": int(
                alignment_report["denominators"]["complete_directed_candidate_pairs"]
            ),
            "directed_contrast_candidates": int(
                retrieval_report["denominators"]["directed_query_candidate_pairs"]
            ),
            "permutations": int(retrieval_report["denominators"]["permutations"]),
            "registered_directional_relations": int(
                retrieval_report["denominators"]["registered_directional_relations"]
            ),
            "registered_signed_entity_truth": int(
                retrieval_report["denominators"]["registered_signed_truth_relations"]
            ),
        },
        "identity_evidence_opportunity": {
            "all_mutual_edges": all_edge_count,
            "known_mutual_edges": known_edge_count,
            "unresolved_mutual_edges": unresolved_edge_count,
            "unresolved_to_known_edge_ratio": unresolved_edge_count / known_edge_count,
            "strict_known_entities": int(
                alignment_report["denominators"]["identity_status"]["library_supported"]
            ),
            "unresolved_entities": int(
                alignment_report["denominators"]["identity_status"]["unresolved_entity"]
            ),
        },
        "registered_relation_family_results": records_without_nan(family),
        "source_independent_primary_results": records_without_nan(source_independent),
        "shared_project_exact_drug_positive_control": records_without_nan(exact_drug),
        "open_retrieval": {
            "calls": records_without_nan(open_summary),
            "project_concentration": concentration,
        },
        "direct_response_competitor_context": known_baseline[
            "registered_relevance_rank_summary"
        ],
        "old_fragment_admitted_graph_context": (
            None if old_report is None else {
                "status": old_report.get("status"),
                "denominators": old_report.get("denominators"),
                "registered_relation_descriptive_performance": old_report.get(
                    "registered_relation_descriptive_performance"
                ),
                "comparison_boundary": (
                    "different candidate-admission graphs; descriptive context only, "
                    "not a same-axis algorithm comparison"
                ),
            }
        ),
        "decision": {
            "large_scale_response_retrieval_executed": True,
            "source_independent_biological_retrieval_pass": False,
            "unknown_information_increment_pass": False,
            "in_house_algorithm_increment_evaluated": False,
            "external_biological_validation_pass": False,
            "reasons": [
                "The five source-independent canonical HFD/IBD relation units are not improved consistently by all versus known evidence.",
                "Unresolved and known worlds differ by more than an order of magnitude in entity and edge opportunity; no opportunity-matched WSE control is included yet.",
                "The three ChemProp accessions are one parent project and dominate open top hits, so max-T significance does not establish biological specificity.",
                "The 42 registered directions contain zero signed entity-level truths; unregistered pairs are not negatives.",
                "Official DreaMS, Noise V1 and Full23 have not yet been scored on this same complete identity axis.",
            ],
        },
        "claim_boundary": [
            "This is a system-wide development analysis, not a single-disease case study.",
            "The exact-drug rows are a shared-parent-project positive control and are not independent external relations.",
            "Similar/opposite response calls are calibrated against response-profile reassignment but are not mechanism truth.",
            "No-match accuracy, FDR, specificity or AUROC is estimable because unregistered candidates are unlabeled rather than negative.",
            "The identity track does not evaluate different-mass chemical relationships.",
        ],
        "provenance": {
            **{name: sha256(path) for name, path in paths.items()},
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [*outputs.values(), report_path]
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "scale": report["scale"],
        "identity_evidence_opportunity": report["identity_evidence_opportunity"],
        "open_project_concentration": concentration,
        "decision": report["decision"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
