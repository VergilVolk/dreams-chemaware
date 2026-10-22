#!/usr/bin/env python3
"""Test whether response retrieval beats target-study composition recognition.

For a fixed query and target study, every candidate contrast has the same
spectral entity graph.  Only beta/SE profiles differ.  Registered target axes
must therefore outrank nonregistered axes within that same study before a
result can be described as response-specific rather than source recognition.
Unregistered axes are controls, not asserted biological negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORES = (
    "similar_response_score",
    "opposite_response_score",
    "unsigned_response_score",
    "composition_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_groups(relations: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
        "eligible_primary_relevance",
    }
    if not required.issubset(relations.columns):
        raise ValueError(f"relation table missing {sorted(required - set(relations.columns))}")
    eligible = relations["eligible_primary_relevance"].astype(str).str.lower().isin({"true", "1"})
    frame = relations.loc[eligible].copy()
    rows: list[dict[str, object]] = []
    for key, block in frame.groupby(
        ["left_study_id", "left_contrast_id", "right_study_id"], sort=False,
    ):
        rows.append({
            "query_study_id": str(key[0]),
            "query_contrast_id": str(key[1]),
            "candidate_study_id": str(key[2]),
            "registered_candidate_contrasts": tuple(sorted(set(block["right_contrast_id"].astype(str)))),
        })
    return rows


def audit_arm(scores: pd.DataFrame, relations: pd.DataFrame, arm_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in registered_groups(relations):
        for world in sorted(scores["evidence_world"].astype(str).unique()):
            block = scores.loc[
                scores["query_study_id"].astype(str).eq(group["query_study_id"])
                & scores["query_contrast_id"].astype(str).eq(group["query_contrast_id"])
                & scores["candidate_study_id"].astype(str).eq(group["candidate_study_id"])
                & scores["evidence_world"].astype(str).eq(world)
            ].copy()
            if block["candidate_contrast_id"].astype(str).duplicated().any():
                raise RuntimeError("candidate contrasts are duplicated within a source-conditioned query")
            registered = set(group["registered_candidate_contrasts"])
            observed = set(block["candidate_contrast_id"].astype(str))
            if not registered.issubset(observed):
                raise RuntimeError("a registered candidate contrast is absent from the fixed score table")
            is_positive = block["candidate_contrast_id"].astype(str).isin(registered)
            controls = ~is_positive
            base = {
                **{key: value for key, value in group.items() if key != "registered_candidate_contrasts"},
                "analysis_arm": arm_label,
                "evidence_world": world,
                "registered_candidate_contrasts": ";".join(group["registered_candidate_contrasts"]),
                "candidate_contrasts_in_target_study": len(block),
                "registered_axes": int(is_positive.sum()),
                "nonregistered_axis_controls": int(controls.sum()),
                "axis_specificity_evaluable": bool(is_positive.any() and controls.any()),
            }
            for score in SCORES:
                values = pd.to_numeric(block[score], errors="coerce")
                positive = values[is_positive].dropna()
                control = values[controls].dropna()
                prefix = score.removesuffix("_score")
                evaluable = len(positive) == int(is_positive.sum()) and len(control) == int(controls.sum()) and len(control) > 0
                base[f"{prefix}_evaluable"] = bool(evaluable)
                if evaluable:
                    minimum_positive = float(positive.min())
                    maximum_control = float(control.max())
                    base[f"{prefix}_minimum_registered"] = minimum_positive
                    base[f"{prefix}_maximum_axis_control"] = maximum_control
                    base[f"{prefix}_strict_margin"] = minimum_positive - maximum_control
                    base[f"{prefix}_strict_separation"] = bool(minimum_positive > maximum_control)
                    base[f"{prefix}_mean_registered_minus_control"] = float(positive.mean() - control.mean())
                else:
                    for suffix in (
                        "minimum_registered", "maximum_axis_control", "strict_margin",
                        "mean_registered_minus_control",
                    ):
                        base[f"{prefix}_{suffix}"] = np.nan
                    base[f"{prefix}_strict_separation"] = False
            rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-scores", required=True, type=Path)
    parser.add_argument("--reassigned-scores", required=True, type=Path)
    parser.add_argument("--relations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    observed = pd.read_csv(args.observed_scores, sep="\t")
    reassigned = pd.read_csv(args.reassigned_scores, sep="\t")
    relations = pd.read_csv(args.relations, sep="\t", dtype=str)
    required_scores = {
        "query_study_id", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id", "evidence_world", *SCORES,
    }
    for name, frame in (("observed", observed), ("reassigned", reassigned)):
        if not required_scores.issubset(frame.columns):
            raise ValueError(f"{name} score table lacks required fields")
    result = pd.concat(
        [
            audit_arm(observed, relations, "observed"),
            audit_arm(
                reassigned,
                relations,
                "deterministic_within_study_identity_stratum_profile_reassignment",
            ),
        ],
        ignore_index=True,
    )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "same_source_axis_specificity.tsv"
    result.to_csv(output, sep="\t", index=False)

    summary: dict[str, object] = {}
    for arm, block in result.groupby("analysis_arm", sort=False):
        summary[str(arm)] = {}
        for world, world_block in block.groupby("evidence_world", sort=False):
            evaluable = world_block["axis_specificity_evaluable"].astype(bool)
            summary[str(arm)][str(world)] = {
                "registered_query_target_groups": len(world_block),
                "axis_specificity_evaluable_groups": int(evaluable.sum()),
                **{
                    f"{score.removesuffix('_score')}_strict_separation": int(
                        world_block.loc[evaluable, f"{score.removesuffix('_score')}_strict_separation"].sum()
                    )
                    for score in SCORES
                },
            }
    report = {
        "schema": "contrast_atlas_same_source_axis_specificity_v1",
        "status": "DEVELOPMENT_CONSUMED_RESPONSE_SPECIFICITY_AUDIT_COMPLETE",
        "primary_endpoint": (
            "unsigned strict separation: the least-scoring registered response axis must exceed "
            "the best nonregistered axis from the same target study"
        ),
        "summary": summary,
        "input_sha256": {
            str(path): sha256(path)
            for path in (args.observed_scores, args.reassigned_scores, args.relations)
        },
        "output_sha256": sha256(output),
        "claim_boundary": [
            "Nonregistered response axes are specificity controls, not asserted biological negatives.",
            "The deterministic reassignment is one sensitivity stress, not a calibrated permutation null.",
            "A response-specific claim requires observed separation plus failure under calibrated profile reassignment; this audit alone cannot pass that gate.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
