#!/usr/bin/env python
"""Summarise threshold-robust recursive BioAware reachability headroom."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def robust_rows(frame: pd.DataFrame, step: int, depth: int, minimum_rotations: int) -> pd.DataFrame:
    subset = frame[
        frame["graph_step"].eq(step)
        & frame["maximum_depth"].eq(depth)
        & ~frame["baseline_correct"]
    ]
    grouped = subset.groupby(
        ["query_id", "truth_candidate_id"], as_index=False
    ).agg(
        heldout_rotations=("fold", "size"),
        rescue_rotations=("strict_rescue_headroom", "sum"),
        truth_supported_rotations=("truth_supported", "sum"),
        wrong_supported_rotations=("wrong_supported", "sum"),
    )
    if not grouped["heldout_rotations"].eq(7).all():
        raise RuntimeError("each frozen query must have seven held-out rotations")
    return grouped[grouped["rescue_rotations"].ge(minimum_rotations)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold-dirs", nargs=3, type=Path,
        default=[
            Path("data/validation/bioaware_metdna3_recursive_headroom_threshold1000_v1"),
            Path("data/validation/bioaware_metdna3_recursive_headroom_v1"),
            Path("data/validation/bioaware_metdna3_recursive_headroom_threshold100000_v1"),
        ],
    )
    parser.add_argument(
        "--queries", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/bioaware_metdna3_recursive_sensitivity_v2.json"),
    )
    parser.add_argument("--minimum-rescue-rotations", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    query_meta = pd.read_csv(args.queries)[["query_id", "truth_formula"]].drop_duplicates()
    threshold_frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for directory in args.threshold_dirs:
        report_path = directory / "report.json"
        rotation_path = directory / "per_rotation.csv.gz"
        for path in (report_path, rotation_path):
            if not path.exists():
                raise FileNotFoundError(path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "bioaware_metdna3_recursive_headroom_complete":
            raise RuntimeError(f"invalid recursive report: {report_path}")
        threshold = f"{float(report['selected_noise_threshold']):g}"
        if threshold in threshold_frames:
            raise RuntimeError(f"duplicate threshold {threshold}")
        frame = pd.read_csv(rotation_path)
        threshold_frames[threshold] = frame
        provenance[threshold] = {
            "report_sha256": sha256(report_path),
            "per_rotation_sha256": sha256(rotation_path),
        }

    configurations: dict[str, dict] = {}
    robust_sets: dict[tuple[int, int], dict[str, set[str]]] = {}
    for step in (0, 1):
        for depth in (1, 2, 3):
            per_threshold: dict[str, dict] = {}
            identity_sets: dict[str, set[str]] = {}
            for threshold, frame in threshold_frames.items():
                robust = robust_rows(frame, step, depth, args.minimum_rescue_rotations)
                robust = robust.merge(query_meta, on="query_id", validate="many_to_one")
                identity_sets[threshold] = set(robust["truth_candidate_id"])
                per_threshold[threshold] = {
                    "robust_queries": int(robust["query_id"].nunique()),
                    "robust_identities": int(robust["truth_candidate_id"].nunique()),
                    "robust_formulas": int(robust["truth_formula"].nunique()),
                    "query_ids": sorted(robust["query_id"].unique()),
                    "identity_ids": sorted(robust["truth_candidate_id"].unique()),
                }
            robust_sets[(step, depth)] = identity_sets
            common = set.intersection(*identity_sets.values())
            configurations[f"step{step}|depth{depth}"] = {
                "by_threshold": per_threshold,
                "identities_common_to_all_thresholds": len(common),
                "common_identity_ids": sorted(common),
            }

    common_step0_depth1 = set.intersection(*robust_sets[(0, 1)].values())
    common_step0_depth2 = set.intersection(*robust_sets[(0, 2)].values())
    common_step0_depth3 = set.intersection(*robust_sets[(0, 3)].values())
    common_step1_depth3 = set.intersection(*robust_sets[(1, 3)].values())
    report = {
        "status": "bioaware_metdna3_recursive_sensitivity_complete",
        "formal": True,
        "thresholds": sorted(float(value) for value in threshold_frames),
        "robust_definition": (
            f"strict truth rescue in at least {args.minimum_rescue_rotations}/7 held-out seed rotations; "
            "truth/wrong distance ties count against rescue"
        ),
        "configurations": configurations,
        "decomposition": {
            "one_hop_common_identities": len(common_step0_depth1),
            "known_network_two_hop_common_identities": len(common_step0_depth2),
            "known_network_three_hop_common_identities": len(common_step0_depth3),
            "step1_three_hop_common_identities": len(common_step1_depth3),
            "recursive_gain_step0_depth2_vs_depth1": len(
                common_step0_depth2 - common_step0_depth1
            ),
            "depth3_gain_within_step0": len(common_step0_depth3 - common_step0_depth2),
            "predicted_step1_increment_at_depth3": len(
                common_step1_depth3 - common_step0_depth3
            ),
        },
        "gates": {
            "recursive_known_network_headroom_ge_3_identities": len(common_step0_depth2) >= 3,
            "step1_has_independent_increment": bool(common_step1_depth3 - common_step0_depth3),
            "pass_to_step0_candidate_specific_path_scoring": len(common_step0_depth2) >= 3,
            "pass_to_predicted_step1_scoring": bool(
                common_step1_depth3 - common_step0_depth3
            ),
            "pass_to_candidate_specific_path_scoring": len(common_step0_depth2) >= 3,
        },
        "provenance": provenance | {"queries_sha256": sha256(args.queries)},
        "contracts": {
            "all_thresholds_reported": True,
            "primary_threshold_remains_10000": True,
            "headroom_is_not_performance": True,
            "RP_opened": False,
        },
        "claim_limit": (
            "Threshold-robust recursive reachability only. A candidate-specific "
            "path scorer must still demonstrate corrected > introduced under formula OOF."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
