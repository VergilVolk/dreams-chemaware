#!/usr/bin/env python3
"""Open the registered reciprocal Hi-MYC relationship in count-matched scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PRIMARY = (("MTBLS135", "MTBLS3316"), ("MTBLS3316", "MTBLS135"))
CONTRAST = "genotype_by_hfd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--known-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--required-fraction", type=float, default=0.80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    scores = pd.read_csv(args.scores, sep="\t")
    known = pd.read_csv(args.known_scores, sep="\t")
    known = known.loc[known.evidence_world.eq("known")].copy()
    rows: list[dict[str, object]] = []
    for resample, sample in scores.groupby("resample", sort=True):
        joint = True
        for query_study, target_study in PRIMARY:
            block = sample.loc[
                sample.query_study_id.eq(query_study)
                & sample.query_contrast_id.eq(CONTRAST)
                & sample.answerable.astype(str).str.lower().isin(["true", "1"])
            ]
            target = block.loc[
                block.target_study_id.eq(target_study)
                & block.target_contrast_id.eq(CONTRAST)
            ]
            if len(target) != 1:
                passed, rank, specific = False, None, False
            else:
                row = target.iloc[0]
                wrong = block.loc[
                    block.target_study_id.eq(target_study)
                    & block.target_contrast_id.ne(CONTRAST), "signed_response_score"
                ]
                rank = int(row.rank_signed_response_score)
                specific = bool(len(wrong) == 4 and float(row.signed_response_score) > float(wrong.max()))
                passed = bool(rank == 1 and specific)
            joint = joint and passed
            rows.append({
                "resample": int(resample), "query_study_id": query_study,
                "target_study_id": target_study, "signed_rank": rank,
                "counterpart_contrast_specific": specific, "rank_gate_pass": passed,
            })
        rows.append({
            "resample": int(resample), "query_study_id": "JOINT_RECIPROCAL",
            "target_study_id": "JOINT_RECIPROCAL", "signed_rank": None,
            "counterpart_contrast_specific": joint, "rank_gate_pass": joint,
        })
    result = pd.DataFrame(rows)
    resamples = int(scores["resample"].nunique())
    required = int(math.ceil(resamples * args.required_fraction))
    joint_passes = int(result.loc[result.query_study_id.eq("JOINT_RECIPROCAL"), "rank_gate_pass"].sum())

    known_rows = []
    for query_study, target_study in PRIMARY:
        block = known.loc[
            known.query_study_id.eq(query_study) & known.query_contrast_id.eq(CONTRAST)
        ]
        target = block.loc[
            block.target_study_id.eq(target_study) & block.target_contrast_id.eq(CONTRAST)
        ]
        known_rows.append({
            "query_study_id": query_study,
            "answerable": len(target) == 1 and bool(target.iloc[0].answerable),
            "signed_rank": int(target.iloc[0].rank_signed_response_score) if len(target) == 1 and pd.notna(target.iloc[0].rank_signed_response_score) else None,
        })

    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir / "count_matched_primary_adjudication.tsv", sep="\t", index=False)
    report = {
        "schema": "factorial_hfd_count_matched_unresolved_adjudication_v1",
        "status": "COUNT_MATCHED_SENSITIVITY_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "resamples": resamples, "reciprocal_queries_per_resample": 2,
            "required_joint_passes": required,
        },
        "result": {
            "joint_reciprocal_rank_gate_passes": joint_passes,
            "passes_80_percent_sensitivity": joint_passes >= required,
            "known_world_reference": known_rows,
        },
        "claim_boundary": "This coverage-matched sensitivity cannot by itself establish unresolved-entity increment; the separately frozen world-specific null and jackknife gates remain mandatory.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
