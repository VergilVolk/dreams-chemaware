#!/usr/bin/env python3
"""Freeze one MetDNA2 recursive-edge arm against the untouched author run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import binomtest

from tasks.freeze_kgmn_metdna2_baseline import evaluate_200std_truth


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_csv_multisets(left: Path, right: Path) -> tuple[bool, str | None]:
    left_frame = pd.read_csv(left)
    right_frame = pd.read_csv(right)
    if list(left_frame.columns) != list(right_frame.columns):
        return False, "column order or names differ"
    columns = list(left_frame.columns)
    try:
        left_frame = left_frame.sort_values(columns, kind="mergesort", na_position="first").reset_index(drop=True)
        right_frame = right_frame.sort_values(columns, kind="mergesort", na_position="first").reset_index(drop=True)
        assert_frame_equal(
            left_frame,
            right_frame,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except (AssertionError, TypeError, ValueError) as error:
        return False, str(error)[:2000]
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--author-baseline", type=Path, required=True)
    parser.add_argument("--arm", choices=("noop_author", "official_dreams", "author_official_intersection"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or (run_dir / "frozen_arm.json")
    required = {
        "truth": run_dir / "annotation_initial.csv",
        "parameters": run_dir / "frozen_parameters.R",
        "run_log": run_dir / "run.log.txt",
        "hook_summary": run_dir / "recursive_hook_summary.json",
        "credential": run_dir / "03_annotation_credential" / "annontation_credential_long.csv",
        "final_identification": run_dir / "00_annotation_table" / "table1_identification.csv",
        "final_pairs": run_dir / "00_annotation_table" / "table3_identification_pair.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete DreaMS arm run; missing: {missing}")
    if not args.author_baseline.is_file():
        raise FileNotFoundError(args.author_baseline)
    for path in (
        args.calibration,
        args.calibration_report,
        args.embedding_file,
        args.embedding_report,
        args.overlay_manifest,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    author = json.loads(args.author_baseline.read_text(encoding="utf-8"))
    if author.get("status") != "kgmn_metdna2_200std_author_baseline_frozen":
        raise RuntimeError("invalid untouched author baseline")
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if calibration_report.get("status") != "kgmn_dreams_edge_calibration_complete":
        raise RuntimeError("invalid DreaMS edge calibration report")
    if calibration_report.get("provenance", {}).get("calibration_artifact_sha256") != sha256(args.calibration):
        raise RuntimeError("DreaMS edge calibration artifact hash mismatch")
    embedding_report = json.loads(args.embedding_report.read_text(encoding="utf-8"))
    if embedding_report.get("status") != "kgmn_200std_official_dreams_embeddings_frozen":
        raise RuntimeError("invalid frozen 200STD DreaMS embedding report")
    if embedding_report.get("provenance", {}).get("embeddings_sha256") != sha256(args.embedding_file):
        raise RuntimeError("200STD DreaMS embedding artifact hash mismatch")
    overlay_manifest = json.loads(args.overlay_manifest.read_text(encoding="utf-8"))
    if overlay_manifest.get("status") != "kgmn_metdna2_dreams_edge_overlay_built":
        raise RuntimeError("invalid MetDNA2 DreaMS overlay manifest")
    if overlay_manifest.get("author_commit") != author.get("source_commit"):
        raise RuntimeError("overlay and untouched author baseline source commits differ")
    hook = json.loads(required["hook_summary"].read_text(encoding="utf-8"))
    if hook.get("status") != "kgmn_metdna2_recursive_hook_executed" or hook.get("arm") != args.arm:
        raise RuntimeError("recursive hook execution proof does not match requested arm")
    if hook.get("psock_worker_execution_proven") is not True or int(hook.get("dynamic_edges_scored", 0)) <= 0:
        raise RuntimeError("recursive edge hook was not proven on PSOCK workers")

    arm_evaluation = evaluate_200std_truth(required["final_pairs"], required["truth"])
    author_evaluation = author["author_200std_truth_evaluation"]
    author_features = {row["feature"]: row for row in author_evaluation["per_feature"]}
    arm_features = {row["feature"]: row for row in arm_evaluation["per_feature"]}
    if set(author_features) != set(arm_features):
        raise RuntimeError("arm and author truth denominators differ")
    corrected = sorted(
        feature for feature in author_features
        if not author_features[feature]["top1_correct"] and arm_features[feature]["top1_correct"]
    )
    introduced = sorted(
        feature for feature in author_features
        if author_features[feature]["top1_correct"] and not arm_features[feature]["top1_correct"]
    )
    discordant = len(corrected) + len(introduced)
    mcnemar_p = float(binomtest(min(len(corrected), len(introduced)), discordant, 0.5).pvalue) if discordant else 1.0

    author_run_dir = Path(author["run_dir"])
    no_op_comparisons: dict[str, object] | None = None
    if args.arm == "noop_author":
        comparison_paths = {
            "credential": author_run_dir / "03_annotation_credential" / "annontation_credential_long.csv",
            "final_identification": author_run_dir / "00_annotation_table" / "table1_identification.csv",
            "final_pairs": author_run_dir / "00_annotation_table" / "table3_identification_pair.csv",
        }
        no_op_comparisons = {}
        for name, author_path in comparison_paths.items():
            if not author_path.is_file():
                raise FileNotFoundError(author_path)
            equal, detail = compare_csv_multisets(author_path, required[name])
            no_op_comparisons[name] = {"equal": equal, "difference": detail}
        if not all(item["equal"] for item in no_op_comparisons.values()):
            raise RuntimeError(f"no-op overlay failed to reproduce author tables: {no_op_comparisons}")
        if corrected or introduced:
            raise RuntimeError("no-op overlay changed author Top-1 results")

    report = {
        "status": "kgmn_metdna2_200std_dreams_arm_frozen",
        "formal": True,
        "arm": args.arm,
        "protocol": "same MetDNA2 1.2.10 200STD inputs and full credential; recursive MS2 edge arm only",
        "author": {
            "recall1": float(author_evaluation["recall1"]),
            "recall5": float(author_evaluation["recall5"]),
            "coverage": float(author_evaluation["coverage"]),
        },
        "candidate": {
            "recall1": float(arm_evaluation["recall1"]),
            "recall5": float(arm_evaluation["recall5"]),
            "coverage": float(arm_evaluation["coverage"]),
        },
        "delta_recall1": float(arm_evaluation["recall1"] - author_evaluation["recall1"]),
        "corrected": len(corrected),
        "introduced": len(introduced),
        "corrected_features": corrected,
        "introduced_features": introduced,
        "mcnemar_exact_p": mcnemar_p,
        "hook_execution": hook,
        "noop_author_table_reproduction": no_op_comparisons,
        "contracts": {
            "candidate_generation_changed": False,
            "reaction_network_changed": False,
            "credential_changed": False,
            "recursive_edge_score_only": True,
            "author_noop_must_reproduce": args.arm != "noop_author" or no_op_comparisons is not None,
        },
        "artifacts": {
            str(path.relative_to(run_dir)).replace("\\", "/"): sha256(path)
            for path in required.values()
        },
        "external_provenance": {
            "author_baseline_sha256": sha256(args.author_baseline),
            "calibration_artifact_sha256": sha256(args.calibration),
            "calibration_report_sha256": sha256(args.calibration_report),
            "embedding_file_sha256": sha256(args.embedding_file),
            "embedding_report_sha256": sha256(args.embedding_report),
            "overlay_manifest_sha256": sha256(args.overlay_manifest),
        },
        "claim_limit": (
            "Technical author-demo matched-protocol comparison. It does not establish independent-study "
            "annotation improvement, biology performance, shared-embedding improvement, or SOTA."
        ),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
