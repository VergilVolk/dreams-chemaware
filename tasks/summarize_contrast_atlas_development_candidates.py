#!/usr/bin/env python3
"""Screen consumed exact-drug controls for separable biology increments.

This combines already-frozen result tables.  It does not rescore spectra and
does not turn consumed drug labels into external biological validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN = ROOT / "data/validation/contrast_atlas_known_response_baselines_v2_20261029"
DEFAULT_FRAGMENT = ROOT / "data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027"
DEFAULT_OFFICIAL = ROOT / "data/validation/contrast_atlas_open_local_edge_enrichment_official_v1_20261027"
DEFAULT_STAGE1 = ROOT / "data/validation/contrast_atlas_open_local_edge_enrichment_stage1_v1_20261027"
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_development_candidate_screen_v2_20261029"
KEYS = ["left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def method_rows(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "registered_relation_ranks.tsv", sep="\t")
    frame = frame.loc[
        frame["relation_class"].eq("consumed_exact_drug_cross_design")
        & frame["evidence_track"].eq("nonmass_chemical_relation_hypothesis")
    ].copy()
    keep = KEYS + ["evidence_world", "best_face_rank", "best_face_query_maxT_p"]
    frame = frame[keep]
    wide = frame.pivot(index=KEYS, columns="evidence_world", values=[
        "best_face_rank", "best_face_query_maxT_p"
    ])
    wide.columns = [f"{method}_{world}_{'rank' if metric == 'best_face_rank' else 'maxT_p'}"
                    for metric, world in wide.columns]
    return wide.reset_index()


def run(known_dir: Path, fragment_dir: Path, official_dir: Path, stage1_dir: Path, output_dir: Path) -> dict[str, Any]:
    known_dir, fragment_dir, official_dir, stage1_dir, output_dir = map(
        Path.resolve, [known_dir, fragment_dir, official_dir, stage1_dir, output_dir]
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    known_path = known_dir / "registered_relation_ranks.tsv"
    known = pd.read_csv(known_path, sep="\t")
    known = known.loc[known["relation_class"].eq("consumed_exact_drug_cross_design")][
        KEYS + ["continuous_absolute_rank", "idmet_absolute_rank", "ora_best_rank",
                "shared_known_identity_count"]
    ]
    screen = known
    for method, directory in [
        ("fragment", fragment_dir), ("official", official_dir), ("stage1", stage1_dir)
    ]:
        screen = screen.merge(method_rows(directory, method), on=KEYS, how="left", validate="one_to_one")

    screen["stage1_vs_fragment_rank_gain"] = screen["fragment_all_rank"] - screen["stage1_all_rank"]
    screen["stage1_unknown_rank_gain"] = screen["stage1_known_rank"] - screen["stage1_all_rank"]
    screen["unknown_information_candidate"] = (
        screen["stage1_all_rank"].le(5)
        & screen["stage1_known_rank"].gt(5)
        & screen["stage1_all_maxT_p"].le(0.05)
    )
    screen["stage1_algorithm_increment_candidate"] = (
        screen["stage1_all_rank"].le(5)
        & screen["fragment_all_rank"].gt(5)
        & screen["stage1_all_maxT_p"].le(0.05)
    )
    screen["joint_increment_candidate"] = (
        screen["unknown_information_candidate"]
        & screen["stage1_algorithm_increment_candidate"]
    )
    screen["screen_interpretation"] = "neither_increment_gate"
    screen.loc[screen["unknown_information_candidate"], "screen_interpretation"] = "unknown_information_only"
    screen.loc[screen["stage1_algorithm_increment_candidate"], "screen_interpretation"] = "stage1_algorithm_only"
    screen.loc[screen["joint_increment_candidate"], "screen_interpretation"] = "joint_increment_candidate"
    screen = screen.sort_values([
        "joint_increment_candidate", "unknown_information_candidate",
        "stage1_algorithm_increment_candidate", "stage1_all_rank",
    ], ascending=[False, False, False, True], kind="stable")

    output_dir.mkdir(parents=True)
    screen_path = output_dir / "exact_drug_direction_screen.tsv"
    screen.to_csv(screen_path, sep="\t", index=False)
    report = {
        "schema": "contrast_atlas_development_candidate_screen_v2",
        "status": "DEVELOPMENT_CONSUMED_CANDIDATE_SCREEN_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "exact_drug_directions": int(len(screen)),
            "unknown_information_candidates": int(screen["unknown_information_candidate"].sum()),
            "stage1_algorithm_increment_candidates": int(screen["stage1_algorithm_increment_candidate"].sum()),
            "joint_increment_candidates": int(screen["joint_increment_candidate"].sum()),
        },
        "screen_rule": {
            "unknown_information": "Stage-1 all rank <=5, Stage-1 known rank >5, all max-T p<=0.05",
            "stage1_algorithm_increment": "Stage-1 all rank <=5, fragment all rank >5, Stage-1 all max-T p<=0.05",
            "joint": "both rules",
        },
        "main_observation": {
            "unknown_information_positive_control": "cilnidipine in both directions",
            "joint_unknown_and_algorithm_candidate": None,
            "interpretation": "current consumed controls support an unresolved-signal hypothesis but do not yet close the algorithm-to-biology loop",
        },
        "claim_boundary": [
            "The screen uses consumed labels from one broader ChemProp project and is not external validation.",
            "Max-T calibration controls the specified retrieval null; it does not establish a biological mechanism.",
            "Cilnidipine is not an algorithm-increment example because fragment, Official and Stage-1 all reach rank 1.",
            "No row satisfies both the unknown-information and Stage-1 algorithm-increment screen.",
            "The relation track is not a legal Full23 evaluation domain.",
        ],
        "input_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256(path)
            for path in [
                known_path,
                fragment_dir / "registered_relation_ranks.tsv",
                official_dir / "registered_relation_ranks.tsv",
                stage1_dir / "registered_relation_ranks.tsv",
            ]
        },
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": {screen_path.name: sha256(screen_path)},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-dir", type=Path, default=DEFAULT_KNOWN)
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(
        args.known_dir, args.fragment_dir, args.official_dir, args.stage1_dir, args.output_dir
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
