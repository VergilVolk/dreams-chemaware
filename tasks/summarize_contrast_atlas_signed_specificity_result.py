#!/usr/bin/env python
"""Summarise the scientific decision from signed-specificity retrieval runs.

This script does not rescore a response or tune a threshold.  It reads the
already frozen source-disjoint and parent-project-disjoint score ledgers, then
applies the preregistered-answer and target-removal interpretations needed for
the scientific verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CALLS = {"similar_response", "opposite_response"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-disjoint-dir", type=Path, required=True)
    parser.add_argument("--parent-disjoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(
        directory / "signed_response_candidate_scores.tsv.gz", sep="\t"
    )
    top = pd.read_csv(directory / "signed_response_top_hits.tsv", sep="\t")
    ranks = pd.read_csv(directory / "registered_relation_ranks.tsv", sep="\t")
    return scores, top, ranks


def target_map(ranks: pd.DataFrame) -> dict[tuple[str, str], set[tuple[str, str]]]:
    eligible = ranks.loc[ranks["eligible_primary_relevance"].fillna(False)].copy()
    output: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in eligible.itertuples(index=False):
        query = (str(row.query_study_id), str(row.query_contrast_id))
        target = (str(row.candidate_study_id), str(row.candidate_contrast_id))
        output.setdefault(query, set()).add(target)
    return output


def source_decisions(
    scores: pd.DataFrame, top: pd.DataFrame, ranks: pd.DataFrame
) -> pd.DataFrame:
    targets = target_map(ranks)
    rows: list[dict[str, object]] = []
    methods = sorted(top["response_method"].unique())
    for method in methods:
        hits = top.loc[
            (top["evidence_world"] == "all")
            & (top["response_method"] == method)
        ].copy()
        method_scores = scores.loc[
            (scores["evidence_world"] == "all")
            & (scores["response_method"] == method)
            & scores["score_available"].eq(True)
            & (pd.to_numeric(scores["response_overlap_edges"], errors="coerce") >= 3)
        ].copy()
        calls = hits["result_class"].isin(CALLS)
        correct = []
        has_registered = []
        for hit in hits.itertuples(index=False):
            query = (str(hit.query_study_id), str(hit.query_contrast_id))
            candidate = (str(hit.candidate_study_id), str(hit.candidate_contrast_id))
            has_registered.append(query in targets)
            correct.append(query in targets and candidate in targets[query])
        hits["has_registered"] = has_registered
        hits["correct_top_hit"] = correct

        removal_calls = 0
        removal_denominator = len(targets)
        for query, query_targets in targets.items():
            block = method_scores.loc[
                (method_scores["query_study_id"].astype(str) == query[0])
                & (method_scores["query_contrast_id"].astype(str) == query[1])
            ].copy()
            if block.empty:
                continue
            keep = ~block.apply(
                lambda row: (
                    str(row["candidate_study_id"]),
                    str(row["candidate_contrast_id"]),
                )
                in query_targets,
                axis=1,
            )
            block = block.loc[keep]
            if block.empty:
                continue
            best = block.sort_values(
                ["absolute_score", "candidate_study_id", "candidate_contrast_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0]
            if float(best["query_max_absT_p"]) <= 0.05:
                removal_calls += 1

        registered_calls = calls & hits["has_registered"]
        rows.append(
            {
                "response_method": method,
                "queries": int(len(hits)),
                "development_calls": int(calls.sum()),
                "registered_query_count": int(hits["has_registered"].sum()),
                "registered_target_top1": int(hits["correct_top_hit"].sum()),
                "called_registered_target": int(
                    (calls & hits["correct_top_hit"]).sum()
                ),
                "called_nonregistered_target_for_registered_query": int(
                    (registered_calls & ~hits["correct_top_hit"]).sum()
                ),
                "calls_for_queries_without_registered_control": int(
                    (calls & ~hits["has_registered"]).sum()
                ),
                "registered_target_removal_queries": removal_denominator,
                "post_registered_target_removal_calls": removal_calls,
            }
        )
    return pd.DataFrame(rows)


def exact_drug_controls(
    scores: pd.DataFrame, ranks: pd.DataFrame
) -> pd.DataFrame:
    subset = ranks.loc[
        (ranks["evidence_world"] == "all")
        & (ranks["relation_class"] == "consumed_exact_drug_cross_design")
        & ranks["eligible_primary_relevance"].fillna(False)
    ].copy()
    rows = []
    for method, block in subset.groupby("response_method", sort=True):
        block = block.loc[block["score_available"].eq(True)]
        candidate_sizes = pd.to_numeric(
            block["candidate_source_contrasts"], errors="coerce"
        ).dropna()
        random_top1 = float((1.0 / candidate_sizes).sum())
        random_top3 = float(
            np.minimum(3.0, candidate_sizes).div(candidate_sizes).sum()
        )
        random_mrr = float(
            np.mean(
                [
                    sum(1.0 / rank for rank in range(1, int(size) + 1))
                    / float(size)
                    for size in candidate_sizes
                ]
            )
        )
        local_ranks = []
        for relation in block.itertuples(index=False):
            candidates = scores.loc[
                (scores["evidence_world"] == "all")
                & (scores["response_method"] == method)
                & (scores["query_study_id"].astype(str) == str(relation.query_study_id))
                & (scores["query_contrast_id"].astype(str) == str(relation.query_contrast_id))
                & (scores["candidate_study_id"].astype(str) == str(relation.candidate_study_id))
                & scores["score_available"].eq(True)
            ].copy()
            candidates = candidates.sort_values(
                ["absolute_score", "candidate_contrast_id"],
                ascending=[False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            target_positions = np.flatnonzero(
                candidates["candidate_contrast_id"].astype(str).to_numpy()
                == str(relation.candidate_contrast_id)
            )
            if len(target_positions):
                local_ranks.append(int(target_positions[0]) + 1)
        ranks_numeric = pd.Series(local_ranks, dtype=float)
        rows.append(
            {
                "response_method": method,
                "registered_directions": 24,
                "evaluable_directions": int(len(ranks_numeric)),
                "top1": int((ranks_numeric == 1).sum()),
                "top3": int((ranks_numeric <= 3).sum()),
                "median_rank": float(ranks_numeric.median())
                if len(ranks_numeric)
                else None,
                "mrr": float((1.0 / ranks_numeric).mean())
                if len(ranks_numeric)
                else None,
                "uniform_random_expected_top1": random_top1,
                "uniform_random_expected_top3": random_top3,
                "uniform_random_expected_mrr": random_mrr,
                "query_maxT_005": int(
                    (pd.to_numeric(block["query_max_absT_p"], errors="coerce") <= 0.05).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def parent_decisions(
    top: pd.DataFrame, ranks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, hits in top.loc[top["evidence_world"] == "all"].groupby(
        "response_method", sort=True
    ):
        calls = hits.loc[hits["result_class"].isin(CALLS)].copy()
        eligible = ranks.loc[
            (ranks["evidence_world"] == "all")
            & (ranks["response_method"] == method)
            & ranks["eligible_primary_relevance"].fillna(False)
        ].copy()
        score_available = eligible["score_available"].eq(True)
        evaluable = eligible.loc[
            score_available
            & (
                pd.to_numeric(
                    eligible["response_overlap_edges"], errors="coerce"
                )
                >= 3
            )
        ]
        correct_and_specific = evaluable.loc[
            (pd.to_numeric(evaluable["absolute_rank"], errors="coerce") == 1)
            & (pd.to_numeric(evaluable["query_max_absT_p"], errors="coerce") <= 0.05)
        ]
        rows.append(
            {
                "response_method": method,
                "queries": int(len(hits)),
                "development_calls": int(len(calls)),
                "eligible_cross_parent_registered_directions": int(len(eligible)),
                "evaluable_registered_directions": int(len(evaluable)),
                "registered_top1": int(
                    (pd.to_numeric(evaluable["absolute_rank"], errors="coerce") == 1).sum()
                ),
                "registered_top5": int(
                    (pd.to_numeric(evaluable["absolute_rank"], errors="coerce") <= 5).sum()
                ),
                "registered_query_maxT_005": int(
                    (pd.to_numeric(evaluable["query_max_absT_p"], errors="coerce") <= 0.05).sum()
                ),
                "registered_relevance_top1_and_calibrated": int(
                    len(correct_and_specific)
                ),
            }
        )
    call_columns = [
        "query_study_id",
        "query_contrast_id",
        "response_method",
        "candidate_study_id",
        "candidate_contrast_id",
        "response_face",
        "signed_score",
        "query_max_absT_p",
        "response_overlap_edges",
    ]
    calls = top.loc[
        (top["evidence_world"] == "all") & top["result_class"].isin(CALLS),
        call_columns,
    ].copy()
    return pd.DataFrame(rows), calls


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_scores, source_top, source_ranks = load(args.source_disjoint_dir)
    _, parent_top, parent_ranks = load(args.parent_disjoint_dir)

    source_table = source_decisions(source_scores, source_top, source_ranks)
    drug_table = exact_drug_controls(source_scores, source_ranks)
    parent_table, parent_calls = parent_decisions(parent_top, parent_ranks)

    paths = {
        "source_disjoint_decisions.tsv": source_table,
        "source_local_exact_drug_controls.tsv": drug_table,
        "parent_project_disjoint_decisions.tsv": parent_table,
        "parent_project_disjoint_calls.tsv": parent_calls,
    }
    for name, frame in paths.items():
        write_tsv(frame, args.output_dir / name)

    report = {
        "schema": "contrast_atlas_signed_specificity_scientific_summary_v1",
        "status": "CURRENT_WSE_RESPONSE_ENDPOINT_NOT_QUALIFIED_ON_CONSUMED_CONTROLS",
        "data_role": "development_consumed",
        "decision": {
            "response_specificity": "not_qualified_on_available_development_controls",
            "cross_parent_registered_answer": "none_passed_registered_relevance_and_calibration_gate",
            "unresolved_increment": "not_established_no_opportunity_matched_control",
            "full23_increment": "not_tested_wse_edges_only",
            "biological_discovery": "not_established",
        },
        "source_disjoint": source_table.to_dict(orient="records"),
        "source_local_exact_drug_control": drug_table.to_dict(orient="records"),
        "parent_project_disjoint": parent_table.to_dict(orient="records"),
        "parent_project_disjoint_calls": parent_calls.to_dict(orient="records"),
        "claim_boundary": [
            "All studies, response contrasts, and registered relations are development/consumed.",
            "Exact-drug relations are internal controls from one ChemProp parent project, not cross-project validation.",
            "HFD and IBD registrations establish relevance families, not signed entity-level biological truth.",
            "A post-registered-target-removal call is failure to abstain in a stress test, not a proven false biological relation.",
            "The parent-project grouping collapses known shared projects but treats most other accessions as provisionally independent; it is not a fully verified project-independence registry.",
            "The implemented max-null combines source-pair-specific profile reassignments that are not synchronised across candidate sources; it is not an exact joint Westfall-Young FWER calibration.",
            "P values are development quantities within one method and evidence world and are not multiplicity-corrected across all five arms, three worlds, and 53 queries.",
            "A development p<=0.05 unregistered call is an exploratory hypothesis, not a correct answer or discovery.",
            "Known and unresolved worlds have unequal matching opportunities; no unresolved-information increment is claimed.",
            "The chemical graph is WSE-based; Full23, Noise, Official DreaMS, ChemAware, and BioAware increments are not evaluated here.",
        ],
        "provenance": {
            "source_disjoint_report_sha256": sha256(
                args.source_disjoint_dir / "report.json"
            ),
            "source_disjoint_scores_sha256": sha256(
                args.source_disjoint_dir / "signed_response_candidate_scores.tsv.gz"
            ),
            "parent_disjoint_report_sha256": sha256(
                args.parent_disjoint_dir / "report.json"
            ),
            "parent_disjoint_scores_sha256": sha256(
                args.parent_disjoint_dir / "signed_response_candidate_scores.tsv.gz"
            ),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    output_files = [report_path] + [args.output_dir / name for name in paths]
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
