#!/usr/bin/env python3
"""Audit HFD response-axis specificity within one frozen target source.

The mouse HFD query has four candidate contrasts in MSV000088023: two HFD-vs-
normal contrasts and two sex contrasts.  Ranking within that single source
removes source-identification as an explanation.  This audit consumes frozen
observed and response-reassignment scores; it does not tune or rescore spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUERY = ("MSV000097426", "mouse_hfd_vs_nfd")
TARGET_STUDY = "MSV000088023"
RELEVANT = {"female_hfd_vs_normal", "male_hfd_vs_normal"}
DECOYS = {"female_vs_male_hfd", "female_vs_male_normal"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[
        frame["query_study_id"].astype(str).eq(QUERY[0])
        & frame["query_contrast_id"].astype(str).eq(QUERY[1])
        & frame["candidate_study_id"].astype(str).eq(TARGET_STUDY)
    ].copy()
    observed_candidates = set(result["candidate_contrast_id"].astype(str))
    if observed_candidates != RELEVANT | DECOYS:
        raise ValueError(f"candidate contrast drift: {observed_candidates}")
    result["axis_class"] = np.where(
        result["candidate_contrast_id"].astype(str).isin(RELEVANT), "HFD_VS_NORMAL", "SEX"
    )
    result["alignment_strength"] = result["signed_similar_score"].abs()
    grouping = ["evidence_world"]
    if "null_replicate" in result:
        grouping.insert(0, "null_replicate")
    result["source_conditioned_rank"] = result.groupby(grouping)[
        "alignment_strength"
    ].rank(ascending=False, method="min")
    return result


def summarise(group: pd.DataFrame) -> dict[str, object]:
    available = group["alignment_strength"].notna()
    if not available.all():
        return {
            "score_available": False,
            "reason": "at least one within-source contrast lacks a score",
        }
    top2 = set(group.loc[group["source_conditioned_rank"] <= 2, "candidate_contrast_id"])
    relevant = group["axis_class"].eq("HFD_VS_NORMAL")
    margin = float(
        group.loc[relevant, "alignment_strength"].min()
        - group.loc[~relevant, "alignment_strength"].max()
    )
    return {
        "score_available": True,
        "exact_hfd_top2": top2 == RELEVANT,
        "top2_contrasts": sorted(top2),
        "weakest_hfd_minus_strongest_sex_margin": margin,
        "reason": "available",
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-dir", type=Path,
        default=Path("data/validation/contrast_atlas_centered_response_kernel_v2_20261020"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_hfd_source_conditioned_axis_v3_20261022"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    observed_path = args.kernel_dir / "fixed_candidate_scores.tsv.gz"
    null_path = args.kernel_dir / "stratified_response_reassignment_scores.tsv.gz"
    observed = select(pd.read_csv(observed_path, sep="\t"))
    null = select(pd.read_csv(null_path, sep="\t"))

    observed_rows = []
    null_rows = []
    for world, group in observed.groupby("evidence_world", sort=True):
        observed_rows.append({"evidence_world": world, **summarise(group)})
    for (replicate, world), group in null.groupby(
        ["null_replicate", "evidence_world"], sort=True
    ):
        null_rows.append(
            {"null_replicate": int(replicate), "evidence_world": world, **summarise(group)}
        )
    observed_summary = pd.DataFrame(observed_rows)
    null_summary = pd.DataFrame(null_rows)

    sensitivity = []
    for row in observed_summary.to_dict("records"):
        world = row["evidence_world"]
        block = null_summary.loc[
            null_summary["evidence_world"].eq(world)
            & null_summary["score_available"].astype(bool)
        ]
        if not row["score_available"]:
            sensitivity.append(
                {
                    "evidence_world": world,
                    "score_available": False,
                    "claim_status": "ABSTAIN_INSUFFICIENT_KNOWN_RESPONSE_EVIDENCE",
                }
            )
            continue
        observed_margin = float(row["weakest_hfd_minus_strongest_sex_margin"])
        null_margin = block["weakest_hfd_minus_strongest_sex_margin"].to_numpy(float)
        sensitivity.append(
            {
                "evidence_world": world,
                "score_available": True,
                "observed_exact_hfd_top2": bool(row["exact_hfd_top2"]),
                "observed_separation_margin": observed_margin,
                "null_replicates_available": int(len(block)),
                "null_exact_hfd_top2_count": int(block["exact_hfd_top2"].sum()),
                "null_exact_hfd_top2_fraction": float(block["exact_hfd_top2"].mean()),
                "null_margin_as_large_or_larger_count": int((null_margin >= observed_margin).sum()),
                "null_margin_as_large_or_larger_fraction": float(
                    np.mean(null_margin >= observed_margin)
                ),
                "claim_status": "NONCALIBRATED_RESPONSE_REASSIGNMENT_SENSITIVITY",
            }
        )

    args.output_dir.mkdir(parents=True)
    detail_path = args.output_dir / "source_conditioned_candidate_detail.tsv"
    null_summary_path = args.output_dir / "response_reassignment_summary.tsv"
    detail_columns = [column for column in observed.columns if column != "axis_class"] + [
        "axis_class"
    ]
    observed[detail_columns].to_csv(detail_path, sep="\t", index=False)
    null_summary.to_csv(null_summary_path, sep="\t", index=False)
    report = {
        "schema": "contrast_atlas_hfd_source_conditioned_axis_v1",
        "status": "DEVELOPMENT_CONSUMED_POSITIVE_CONTROL_SENSITIVITY_COMPLETE",
        "question": (
            "Within MSV000088023 only, does the mouse HFD query rank both HFD-vs-normal "
            "contrasts above both sex contrasts by direction-agnostic response alignment?"
        ),
        "denominators": {
            "query_contrasts": 1,
            "target_source_families": 1,
            "target_contrasts": 4,
            "relevant_hfd_contrasts": 2,
            "sex_contrast_decoys": 2,
            "response_reassignment_replicates": int(null["null_replicate"].nunique()),
            "uniform_random_exact_top2_probability": 1.0 / 6.0,
        },
        "observed": observed_summary.to_dict("records"),
        "sensitivity": sensitivity,
        "interpretation": {
            "source_identity_confounding": "controlled by ranking only within one target source",
            "composition_control": (
                "all four candidate contrasts share the same target spectra and composition score; "
                "composition cannot determine their within-source order"
            ),
            "known_only": (
                "abstains because target HFD/sex contrasts contain only five known entities and "
                "effective response support is below the frozen threshold"
            ),
            "unknown_increment": (
                "not established: unresolved-only answers the control, but no measurement-matched "
                "known-only accuracy comparison is available"
            ),
        },
        "claim_boundary": [
            "This is an exact-intervention response-axis positive control, not an open biological discovery.",
            "Alignment strength is direction-agnostic because sex and biofluid differ across studies.",
            "Feature-level response reassignment is a sensitivity, not a calibrated randomisation p-value.",
            "All inputs are development/consumed.",
        ],
        "input_sha256": {
            str(observed_path): sha256(observed_path),
            str(null_path): sha256(null_path),
        },
        "output_sha256": {
            detail_path.name: sha256(detail_path),
            null_summary_path.name: sha256(null_summary_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report = json_safe(report)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
