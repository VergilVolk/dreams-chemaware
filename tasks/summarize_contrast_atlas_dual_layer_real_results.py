#!/usr/bin/env python3
"""Summarise the frozen seven-source Contrast Atlas evidence without inventing truth.

This script does not rescore spectra.  It joins the already-frozen identity and
chemical-relation baselines and makes the scientific answer categories explicit:
registered design relevance, unadjudicated open hits, unavailable identity
answers, and the absence of signed-direction truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


RELATION_METRICS = {
    "signed_similar": ("signed_similar_score", "signed_similar_absolute_rank"),
    "signed_opposite": ("signed_opposite_score", "signed_opposite_absolute_rank"),
    "unsigned_response": ("unsigned_response_score", "unsigned_response_absolute_rank"),
    "composition": ("composition_score", "composition_absolute_rank"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def relation_top_hits(scores: pd.DataFrame, registered: pd.DataFrame) -> pd.DataFrame:
    registered_keys = {
        (row.left_study_id, row.left_contrast_id, row.right_study_id, row.right_contrast_id)
        for row in registered.itertuples(index=False)
    }
    rows = []
    keys = ["query_study_id", "query_source_family", "query_contrast_id", "evidence_world"]
    for key, group in scores.groupby(keys, sort=True, dropna=False):
        for metric, (score_col, rank_col) in RELATION_METRICS.items():
            winners = group.loc[group[rank_col] == group[rank_col].min()].copy()
            for _, row in winners.iterrows():
                is_registered = (
                    row["query_study_id"], row["query_contrast_id"],
                    row["candidate_study_id"], row["candidate_contrast_id"],
                ) in registered_keys
                rows.append(
                    {
                        **dict(zip(keys, key)),
                        "metric": metric,
                        "candidate_study_id": row["candidate_study_id"],
                        "candidate_source_family": row["candidate_source_family"],
                        "candidate_contrast_id": row["candidate_contrast_id"],
                        "score": scalar(row[score_col]),
                        "absolute_rank": scalar(row[rank_col]),
                        "tie_count_at_top": len(winners),
                        "biological_truth_status": (
                            "REGISTERED_DESIGN_RELEVANCE_DIRECTION_UNSPECIFIED"
                            if is_registered else "UNADJUDICATED_OPEN_HIT"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def registered_relation_summary(ranks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep = [
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
        "relation_direction", "relation_class", "expected_signed_direction",
        "evidence_world", "candidate_universe_size", "score_available",
    ]
    out = ranks[keep].copy()
    for metric, (score_col, rank_col) in RELATION_METRICS.items():
        out[f"{metric}_score"] = ranks[score_col]
        out[f"{metric}_rank"] = ranks[rank_col]
        out[f"{metric}_top1"] = ranks[rank_col].eq(1)
    out["claim_status"] = "DESCRIPTIVE_REGISTERED_RELEVANCE_DIRECTION_UNSPECIFIED"

    counts = {}
    for world, group in out.groupby("evidence_world", sort=True):
        counts[world] = {
            "registered_directed_relations": int(len(group)),
            "score_available": int(group["score_available"].sum()),
            "top1_counts": {
                metric: int(group[f"{metric}_top1"].sum()) for metric in RELATION_METRICS
            },
        }
    return out, counts


def identity_registered_summary(ranks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = [
        "relation_direction", "left_study_id", "left_contrast_id", "right_study_id",
        "right_contrast_id", "relation_class", "expected_signed_direction",
        "shared_identity_count", "score_available", "unavailability_reason",
        "signed_similar_score", "similar_absolute_rank", "unsigned_response_score",
        "unsigned_absolute_rank", "available_candidate_count",
    ]
    out = ranks[cols].copy()
    out["claim_status"] = "DESCRIPTIVE_REGISTERED_RELEVANCE_DIRECTION_UNSPECIFIED"
    summary = {
        "registered_directed_relations": int(len(out)),
        "score_available": int(out["score_available"].sum()),
        "unavailable": int((~out["score_available"]).sum()),
        "similar_top1": int(out["similar_absolute_rank"].eq(1).sum()),
        "unsigned_top1": int(out["unsigned_absolute_rank"].eq(1).sum()),
        "zero_shared_identity": int(out["shared_identity_count"].eq(0).sum()),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-dir", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v3_ibd_20261019"),
    )
    parser.add_argument(
        "--relation-dir", type=Path,
        default=Path("data/validation/contrast_atlas_sparse_fragment_measure_baseline_v2_ibd_20261019"),
    )
    parser.add_argument(
        "--identity-dir", type=Path,
        default=Path("data/validation/contrast_atlas_strict_identity_known_response_baseline_v1_20261019"),
    )
    parser.add_argument(
        "--identity-calibration-dir", type=Path,
        default=Path("data/validation/contrast_atlas_official_mass_compatible_identity_calibration_v3_20261020"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_dual_layer_real_result_v1_20261020"),
    )
    args = parser.parse_args()

    collection_report_path = args.collection_dir / "report.json"
    relation_report_path = args.relation_dir / "report.json"
    relation_scores_path = args.relation_dir / "fixed_candidate_scores.tsv.gz"
    relation_ranks_path = args.relation_dir / "registered_relation_absolute_ranks.tsv"
    identity_report_path = args.identity_dir / "report.json"
    identity_ranks_path = args.identity_dir / "registered_relation_absolute_ranks.tsv"
    identity_calibration_path = args.identity_calibration_dir / "report.json"
    registered_path = args.collection_dir / "registered_relations.tsv"

    collection = json.loads(collection_report_path.read_text(encoding="utf-8"))
    relation_report = json.loads(relation_report_path.read_text(encoding="utf-8"))
    identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
    identity_calibration = json.loads(identity_calibration_path.read_text(encoding="utf-8"))
    relation_scores = pd.read_csv(relation_scores_path, sep="\t")
    relation_ranks = pd.read_csv(relation_ranks_path, sep="\t")
    identity_ranks = pd.read_csv(identity_ranks_path, sep="\t")
    registered = pd.read_csv(registered_path, sep="\t")

    if collection["denominators"]["registered_signed_direction_truth_relations"] != 0:
        raise ValueError("This summary contract assumes no registered signed-direction truth.")

    top_hits = relation_top_hits(relation_scores, registered)
    relation_registered, relation_counts = registered_relation_summary(relation_ranks)
    identity_registered, identity_counts = identity_registered_summary(identity_ranks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_hits_path = args.output_dir / "open_relation_top_hits.tsv"
    relation_registered_path = args.output_dir / "registered_relation_layer_results.tsv"
    identity_registered_path = args.output_dir / "registered_identity_layer_results.tsv"
    top_hits.to_csv(top_hits_path, sep="\t", index=False)
    relation_registered.to_csv(relation_registered_path, sep="\t", index=False)
    identity_registered.to_csv(identity_registered_path, sep="\t", index=False)

    registered_top_hits = int(
        top_hits["biological_truth_status"].eq(
            "REGISTERED_DESIGN_RELEVANCE_DIRECTION_UNSPECIFIED"
        ).sum()
    )
    report = {
        "schema": "contrast_atlas_dual_layer_real_result_v1",
        "status": "DEVELOPMENT_CONSUMED_CENTRAL_GATE_NOT_PASSED",
        "scientific_target": (
            "study-level directed quantitative biological-response retrieval; identity replication "
            "and chemically related response association are distinct, co-equal evidence layers"
        ),
        "collection_denominators": collection["denominators"],
        "identity_layer": {
            "role": "same-entity cross-study response replication; beta/SE may align only here",
            "registered_relation_results": identity_counts,
            "all_fixed_candidate_pairs": identity_report["denominators"]["fixed_source_disjoint_candidate_pairs"],
            "available_fixed_candidate_pairs": identity_report["denominators"]["available_candidate_pairs"],
            "unavailable_fixed_candidate_pairs": identity_report["denominators"]["unavailable_candidate_pairs"],
            "identity_calibration_subset": {
                "scope": "known-identity positive groups only; not open-set FDR",
                "primary_all_world_result": identity_calibration["primary_all_world_result"],
                "world_summary": identity_calibration["world_summary"],
            },
            "unknown_identity_increment": "NOT_ESTABLISHED",
        },
        "chemical_relation_layer": {
            "role": (
                "different chemically related entities may support response-pattern retrieval; "
                "their beta/SE are never pooled as the same molecule"
            ),
            "method": "transparent sparse fragment-measure baseline",
            "fixed_candidate_rows_per_world": relation_report["denominators"]["fixed_source_disjoint_candidate_rows_per_world"],
            "registered_relation_results_by_world": relation_counts,
            "open_top_hit_rows": int(len(top_hits)),
            "registered_design_relation_top_hit_rows_including_ties": registered_top_hits,
            "open_top_hit_truth_status": "UNADJUDICATED; unregistered pairs are not negatives",
        },
        "central_gate": {
            "registered_design_relevance_relations": collection["denominators"]["registered_directed_relevance_relations"],
            "registered_signed_direction_truth_relations": 0,
            "adjudicated_correct_open_retrievals": 0,
            "adjudicated_incorrect_open_retrievals": 0,
            "reason": (
                "No signed-direction biological truth was registered. The registered HFD and IBD "
                "relations permit descriptive ranking only, and open top hits are unadjudicated."
            ),
            "unknown_information_increment": "NOT_DEMONSTRATED",
            "verdict": (
                "The seven-source substrate is real, but neither a correct open biological discovery "
                "nor an unresolved-MS/MS increment over known-only is established."
            ),
        },
        "leakage_and_claim_status": {
            "data_role": "development_consumed",
            "collection_leakage_status": collection["leakage_status"],
            "external_validation": False,
            "unresolved_definition": "not library-supported under the frozen partition; not a unique metabolite",
        },
        "input_sha256": {
            str(p): sha256(p) for p in [
                collection_report_path, relation_report_path, relation_scores_path,
                relation_ranks_path, identity_report_path, identity_ranks_path,
                identity_calibration_path,
                registered_path,
            ]
        },
        "output_sha256": {
            top_hits_path.name: sha256(top_hits_path),
            relation_registered_path.name: sha256(relation_registered_path),
            identity_registered_path.name: sha256(identity_registered_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["central_gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
