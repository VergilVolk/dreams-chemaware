#!/usr/bin/env python3
"""Post-hoc source-conditioned obesity-axis generalisation check.

This is not a preregistered endpoint.  It asks whether each obesity contrast in
one four-contrast source ranks both obesity contrasts in the other source above
both diabetes/depression contrasts.  It is used only to test whether the HFD
positive-control behaviour generalises within the consumed collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


AXES = [
    {
        "query_study": "MSV000084112",
        "query_contrasts": ["obesity_with_diabetes", "obesity_without_diabetes"],
        "target_study": "MSV000083077",
        "relevant": {"obesity_high_depression", "obesity_low_depression"},
    },
    {
        "query_study": "MSV000083077",
        "query_contrasts": ["obesity_high_depression", "obesity_low_depression"],
        "target_study": "MSV000084112",
        "relevant": {"obesity_with_diabetes", "obesity_without_diabetes"},
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(frame: pd.DataFrame, include_replicate: bool) -> pd.DataFrame:
    rows = []
    for axis in AXES:
        for query_contrast in axis["query_contrasts"]:
            block = frame.loc[
                frame["query_study_id"].astype(str).eq(axis["query_study"])
                & frame["query_contrast_id"].astype(str).eq(query_contrast)
                & frame["candidate_study_id"].astype(str).eq(axis["target_study"])
            ].copy()
            grouping = ["evidence_world"]
            if include_replicate:
                grouping.insert(0, "null_replicate")
            block["strength"] = block["signed_similar_score"].abs()
            for key, group in block.groupby(grouping, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                values = dict(zip(grouping, key))
                available = group["strength"].notna().all() and len(group) == 4
                record = {
                    **values,
                    "query_study_id": axis["query_study"],
                    "query_contrast_id": query_contrast,
                    "candidate_study_id": axis["target_study"],
                    "score_available": bool(available),
                }
                if available:
                    ranked = group.sort_values("strength", ascending=False)
                    top2 = set(ranked.head(2)["candidate_contrast_id"].astype(str))
                    relevant = group["candidate_contrast_id"].astype(str).isin(axis["relevant"])
                    record.update(
                        {
                            "exact_obesity_top2": top2 == axis["relevant"],
                            "top2_contrasts": ";".join(sorted(top2)),
                            "separation_margin": float(
                                group.loc[relevant, "strength"].min()
                                - group.loc[~relevant, "strength"].max()
                            ),
                        }
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-dir", type=Path,
        default=Path("data/validation/contrast_atlas_centered_response_kernel_v2_20261020"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_obesity_source_conditioned_axis_v1_20261022"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    observed_path = args.kernel_dir / "fixed_candidate_scores.tsv.gz"
    null_path = args.kernel_dir / "stratified_response_reassignment_scores.tsv.gz"
    observed = audit(pd.read_csv(observed_path, sep="\t"), False)
    null = audit(pd.read_csv(null_path, sep="\t"), True)

    summaries = []
    for world, group in observed.groupby("evidence_world", sort=True):
        available = group["score_available"].astype(bool)
        null_world = null.loc[
            null["evidence_world"].eq(world) & null["score_available"].astype(bool)
        ]
        summaries.append(
            {
                "evidence_world": world,
                "queries": int(len(group)),
                "queries_available": int(available.sum()),
                "observed_exact_top2": int(group.loc[available, "exact_obesity_top2"].sum()),
                "null_query_replicates": int(len(null_world)),
                "null_exact_top2_fraction": (
                    float(null_world["exact_obesity_top2"].mean()) if len(null_world) else None
                ),
            }
        )

    args.output_dir.mkdir(parents=True)
    observed_path_out = args.output_dir / "observed_query_results.tsv"
    null_path_out = args.output_dir / "response_reassignment_query_results.tsv.gz"
    observed.to_csv(observed_path_out, sep="\t", index=False)
    null.to_csv(null_path_out, sep="\t", index=False, compression="gzip")
    report = {
        "schema": "contrast_atlas_obesity_source_conditioned_axis_v1",
        "status": "DEVELOPMENT_CONSUMED_POSTHOC_GENERALISATION_CHECK_COMPLETE",
        "denominators": {
            "source_directions": 2,
            "obesity_queries": 4,
            "candidate_contrasts_per_target_source": 4,
            "relevant_obesity_contrasts_per_target_source": 2,
            "uniform_random_exact_top2_probability": 1.0 / 6.0,
            "response_reassignment_replicates": int(null["null_replicate"].nunique()),
        },
        "summary": summaries,
        "verdict": (
            "The HFD within-source axis result does not broadly generalise: exact obesity Top-2 "
            "recovery is sparse and this condition family was not preregistered."
        ),
        "claim_boundary": [
            "This obesity analysis is post-hoc and cannot be a confirmatory endpoint.",
            "Condition-family equality does not guarantee identical biology across diabetes and depression strata.",
            "Unregistered cross-study pairs remain unadjudicated, not negatives.",
            "All inputs are development/consumed.",
        ],
        "input_sha256": {str(path): sha256(path) for path in (observed_path, null_path)},
        "output_sha256": {
            observed_path_out.name: sha256(observed_path_out),
            null_path_out.name: sha256(null_path_out),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
