#!/usr/bin/env python3
"""Compare local directed-response retrieval with the mean-embedding baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = (("MTBLS135", "MTBLS3316"), ("MTBLS3316", "MTBLS135"))
CONTRAST = "genotype_by_hfd"
WORLDS = ("all", "known", "unresolved")
TRACKS = ("local_all", "identity_compatible", "chemical_relation")


def truthy(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().isin(["true", "1"]).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-scores", required=True, type=Path)
    parser.add_argument("--edge-contributions", required=True, type=Path)
    parser.add_argument("--mean-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-supporting-edges", type=int, default=25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    local = pd.read_csv(args.local_scores, sep="\t")
    mean = pd.read_csv(args.mean_scores, sep="\t")
    edges = pd.read_csv(args.edge_contributions, sep="\t")
    rows: list[dict[str, object]] = []
    supporting: list[pd.DataFrame] = []
    for world in WORLDS:
        for query_study, target_study in PRIMARY:
            local_block = local.loc[
                local.query_study_id.eq(query_study)
                & local.query_contrast_id.eq(CONTRAST)
                & local.evidence_world.eq(world)
            ].copy()
            mean_block = mean.loc[
                mean.query_study_id.eq(query_study)
                & mean.query_contrast_id.eq(CONTRAST)
                & mean.evidence_world.eq(world)
            ].copy()
            local_target = local_block.loc[
                local_block.target_study_id.eq(target_study)
                & local_block.target_contrast_id.eq(CONTRAST)
            ]
            mean_target = mean_block.loc[
                mean_block.target_study_id.eq(target_study)
                & mean_block.target_contrast_id.eq(CONTRAST)
            ]
            if len(local_target) != 1 or len(mean_target) != 1:
                raise RuntimeError(f"missing unique reciprocal row: {world}/{query_study}")
            local_target = local_target.iloc[0]
            mean_target = mean_target.iloc[0]
            mean_answerable = bool(str(mean_target.answerable).lower() in {"true", "1"})
            mean_rank = (
                int(mean_target.rank_signed_response_score)
                if mean_answerable and np.isfinite(mean_target.rank_signed_response_score)
                else np.nan
            )
            for track in TRACKS:
                score_column = f"{track}_score"
                rank_column = f"rank_{score_column}"
                score = float(local_target[score_column])
                rank = int(local_target[rank_column]) if np.isfinite(local_target[rank_column]) else np.nan
                within_study = local_block.loc[
                    local_block.target_study_id.eq(target_study)
                    & local_block.target_contrast_id.ne(CONTRAST), score_column,
                ].dropna()
                specific = bool(
                    np.isfinite(score) and len(within_study) == 4 and score > float(within_study.max())
                )
                rows.append({
                    "evidence_world": world,
                    "query_study_id": query_study,
                    "target_study_id": target_study,
                    "query_contrast_id": CONTRAST,
                    "target_contrast_id": CONTRAST,
                    "track": track,
                    "local_score": score,
                    "local_rank": rank,
                    "local_contrast_specific": specific,
                    "local_answerable_panel_pairs": int(
                        local_target[f"{track}_answerable_panel_pairs"]
                    ),
                    "local_edges": int(local_target[f"{track}_edges"]),
                    "mean_embedding_score": (
                        float(mean_target.signed_response_score) if mean_answerable else np.nan
                    ),
                    "mean_embedding_rank": mean_rank,
                    "rank_improvement_vs_mean": (
                        float(mean_rank - rank)
                        if np.isfinite(mean_rank) and np.isfinite(rank) else np.nan
                    ),
                    "local_top1": bool(rank == 1) if np.isfinite(rank) else False,
                    "mean_top1": bool(mean_rank == 1) if np.isfinite(mean_rank) else False,
                })
            target_edges = edges.loc[
                edges.query_study_id.eq(query_study)
                & edges.query_contrast_id.eq(CONTRAST)
                & edges.target_study_id.eq(target_study)
                & edges.target_contrast_id.eq(CONTRAST)
                & edges.evidence_world.eq(world)
            ].copy()
            if len(target_edges):
                target_edges = target_edges.sort_values(
                    "response_support_weight", ascending=False,
                ).head(args.top_supporting_edges)
                target_edges.insert(0, "registered_reciprocal", True)
                supporting.append(target_edges)

    adjudication = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (world, track), block in adjudication.groupby(["evidence_world", "track"], sort=True):
        summary_rows.append({
            "evidence_world": world,
            "track": track,
            "reciprocal_queries": len(block),
            "answerable_queries": int(block.local_rank.notna().sum()),
            "local_top1_queries": int(block.local_top1.sum()),
            "local_contrast_specific_queries": int(block.local_contrast_specific.sum()),
            "mean_top1_queries": int(block.mean_top1.sum()),
            "new_top1_over_mean": int((block.local_top1 & ~block.mean_top1).sum()),
            "lost_top1_vs_mean": int((~block.local_top1 & block.mean_top1).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    args.output_dir.mkdir(parents=True)
    adjudication.to_csv(
        args.output_dir / "local_vs_mean_reciprocal_results.tsv", sep="\t", index=False,
    )
    summary.to_csv(args.output_dir / "local_vs_mean_summary.tsv", sep="\t", index=False)
    if supporting:
        pd.concat(supporting, ignore_index=True).to_csv(
            args.output_dir / "top_supporting_local_edges.tsv", sep="\t", index=False,
        )
    else:
        pd.DataFrame().to_csv(
            args.output_dir / "top_supporting_local_edges.tsv", sep="\t", index=False,
        )
    primary = summary.loc[
        summary.evidence_world.eq("all") & summary.track.eq("local_all")
    ].iloc[0]
    report = {
        "schema": "factorial_hfd_local_vs_mean_adjudication_v1",
        "status": (
            "LOCAL_DIRECTED_RECIPROCAL_RETRIEVAL_OBSERVED"
            if int(primary.local_top1_queries) > 0 else
            "LOCAL_DIRECTED_RECIPROCAL_RETRIEVAL_NOT_OBSERVED"
        ),
        "data_role": "development_consumed",
        "denominators": {
            "studies": 3, "registered_reciprocal_queries": 2,
            "evidence_worlds": len(WORLDS), "evidence_tracks": len(TRACKS),
        },
        "primary_observed": {
            "local_top1_queries": int(primary.local_top1_queries),
            "local_contrast_specific_queries": int(primary.local_contrast_specific_queries),
            "mean_embedding_top1_queries": int(primary.mean_top1_queries),
            "new_top1_over_mean": int(primary.new_top1_over_mean),
            "lost_top1_vs_mean": int(primary.lost_top1_vs_mean),
        },
        "claim_boundary": [
            "This is a consumed development comparison across two related Hi-MYC experiments.",
            "Known and unresolved worlds test evidence contribution; unresolved does not mean novel.",
            "The relation track supports study-level chemical-response association, not entity identity.",
            "NQO1 candidates are an open-set distractor, not a declared biological negative.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
