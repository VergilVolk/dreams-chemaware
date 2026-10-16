#!/usr/bin/env python3
"""Open only the frozen primary relationship and adjudicate reciprocal retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = (
    ("MTBLS135", "MTBLS3316"),
    ("MTBLS3316", "MTBLS135"),
)
CONTRAST = "genotype_by_hfd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--world", default="all", choices=("all", "known", "unresolved"))
    parser.add_argument("--null", type=Path)
    parser.add_argument("--jackknife", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    scores = pd.read_csv(args.scores, sep="\t")
    score_answerable = scores.answerable.astype(str).str.lower().isin(["true", "1"])
    scores = scores.loc[scores.evidence_world.eq(args.world) & score_answerable].copy()
    null = None
    if args.null:
        with np.load(args.null, allow_pickle=False) as packed:
            null_rows = np.asarray(packed["query_sketch_row"], dtype=int)
            null_values = np.asarray(packed["max_candidate_score"], dtype=float)
        if len(null_rows) != len(null_values) or len(set(null_rows)) != len(null_rows):
            raise ValueError("invalid query-max null archive")
        null = {int(row): null_values[index] for index, row in enumerate(null_rows)}
    jackknife = pd.read_csv(args.jackknife, sep="\t") if args.jackknife else None
    rows: list[dict[str, object]] = []
    for query_study, target_study in PRIMARY:
        block = scores.loc[
            scores.query_study_id.eq(query_study)
            & scores.query_contrast_id.eq(CONTRAST)
        ].copy()
        expected = block.loc[
            block.target_study_id.eq(target_study)
            & block.target_contrast_id.eq(CONTRAST)
        ]
        if len(expected) != 1:
            rows.append({
                "query_study_id": query_study, "target_study_id": target_study,
                "status": "ABSTAIN_MISSING_UNIQUE_EXPECTED_TARGET",
            })
            continue
        target = expected.iloc[0]
        within_target_wrong = block.loc[
            block.target_study_id.eq(target_study)
            & block.target_contrast_id.ne(CONTRAST),
            "signed_response_score",
        ]
        signed_rank = int(target.rank_signed_response_score)
        contrast_specific = bool(
            len(within_target_wrong) == 4
            and float(target.signed_response_score) > float(within_target_wrong.max())
        )
        permutation_p = np.nan
        if null is not None:
            distribution = null.get(int(target.query_sketch_row))
            if distribution is None:
                raise ValueError(f"null lacks query row {int(target.query_sketch_row)}")
            permutation_p = float((1 + np.sum(distribution >= float(target.signed_response_score))) / (len(distribution) + 1))
        jackknife_fraction = np.nan
        jackknife_scenarios = 0
        if jackknife is not None:
            relevant = jackknife.loc[
                jackknife.query_study_id.eq(query_study)
                & jackknife.query_contrast_id.eq(CONTRAST)
                & (
                    jackknife.scenario_type.eq("query_panel_deleted")
                    | (
                        jackknife.scenario_type.eq("target_study_panel_deleted")
                        & jackknife.deleted_study_id.eq(target_study)
                    )
                )
            ].copy()
            successes: list[bool] = []
            for _, scenario in relevant[["scenario_type", "deleted_study_id", "deleted_panel_id"]].drop_duplicates().iterrows():
                scenario_answerable = relevant.answerable.astype(str).str.lower().isin(["true", "1"])
                scenario_block = relevant.loc[
                    relevant.scenario_type.eq(scenario.scenario_type)
                    & relevant.deleted_study_id.eq(scenario.deleted_study_id)
                    & relevant.deleted_panel_id.eq(scenario.deleted_panel_id)
                    & scenario_answerable
                ]
                expected_jk = scenario_block.loc[
                    scenario_block.target_study_id.eq(target_study)
                    & scenario_block.target_contrast_id.eq(CONTRAST)
                ]
                if len(expected_jk) != 1:
                    successes.append(False)
                    continue
                expected_row = expected_jk.iloc[0]
                wrong = scenario_block.loc[
                    scenario_block.target_study_id.eq(target_study)
                    & scenario_block.target_contrast_id.ne(CONTRAST),
                    "signed_response_score",
                ]
                successes.append(bool(
                    int(expected_row.rank_signed_response_score) == 1
                    and len(wrong) == 4
                    and float(expected_row.signed_response_score) > float(wrong.max())
                ))
            jackknife_scenarios = len(successes)
            jackknife_fraction = float(np.mean(successes)) if successes else np.nan
        rows.append({
            "query_study_id": query_study,
            "query_contrast_id": CONTRAST,
            "target_study_id": target_study,
            "target_contrast_id": CONTRAST,
            "candidate_objects": len(block),
            "signed_rank": signed_rank,
            "signed_score": float(target.signed_response_score),
            "magnitude_only_rank": int(target.rank_magnitude_only_score),
            "chemistry_only_rank": int(target.rank_chemistry_only_score),
            "counterpart_contrast_specific": contrast_specific,
            "max_candidate_permutation_p": permutation_p,
            "single_panel_deletion_scenarios": jackknife_scenarios,
            "single_panel_deletion_pass_fraction": jackknife_fraction,
            "provisional_rank_gate": bool(signed_rank == 1 and contrast_specific),
            "status": "RANK_GATE_PASS" if signed_rank == 1 and contrast_specific else "RANK_GATE_FAIL",
        })
    adjudication = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    adjudication.to_csv(args.output_dir / "primary_reciprocal_adjudication.tsv", sep="\t", index=False)
    rank_pass = bool(
        len(adjudication) == 2
        and "provisional_rank_gate" in adjudication
        and adjudication.provisional_rank_gate.all()
    )
    null_pass = bool(
        args.null and len(adjudication) == 2
        and adjudication.max_candidate_permutation_p.notna().all()
        and (adjudication.max_candidate_permutation_p <= 0.025).all()
    )
    jackknife_pass = bool(
        args.jackknife and len(adjudication) == 2
        and adjudication.single_panel_deletion_pass_fraction.notna().all()
        and (adjudication.single_panel_deletion_pass_fraction >= 0.80).all()
    )
    full_pass = bool(rank_pass and null_pass and jackknife_pass)
    report = {
        "schema": "factorial_hfd_primary_adjudication_v1",
        "status": (
            "PRIMARY_PASS_REPLICATED_MODEL_ONLY" if full_pass else
            "RANK_GATE_PASS_PENDING_OR_FAILED_NULL_JACKKNIFE" if rank_pass else
            "PRIMARY_RANK_GATE_FAIL"
        ),
        "evidence_world": args.world,
        "denominators": {"reciprocal_queries": 2, "adjudicated_queries": len(adjudication)},
        "rank_gate_pass": rank_pass,
        "max_candidate_null_gate_pass": null_pass,
        "single_panel_deletion_gate_pass": jackknife_pass,
        "full_primary_pass": full_pass,
        "pending_required_gates": [
            "family-wise within-panel response permutation",
            "single-panel-deletion robustness >=80 percent",
        ] if rank_pass and not full_pass else [],
        "claim_boundary": (
            "A rank gate is insufficient for a scientific pass; the frozen permutation and "
            "single-panel-deletion gates remain mandatory."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
