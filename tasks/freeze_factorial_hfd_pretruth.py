#!/usr/bin/env python3
"""Freeze and validate the factorial-HFD pre-truth artifacts before adjudication."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


EXPECTED = {
    "index": "development_pretruth_factorial_response_index",
    "retrieval": "pretruth_scores_frozen",
    "null": "pretruth_max_candidate_null_frozen",
    "jackknife": "pretruth_single_panel_deletion_scores_frozen",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(directory: Path, label: str) -> tuple[dict[str, object], Path]:
    path = directory / "report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != EXPECTED[label]:
        raise RuntimeError(f"{label} status is not frozen: {payload.get('status')}")
    if label != "index" and payload.get("truth_opened") is not False:
        raise RuntimeError(f"{label} did not preserve truth-blind state")
    return payload, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--retrieval-dir", required=True, type=Path)
    parser.add_argument("--null-dir", required=True, type=Path)
    parser.add_argument("--jackknife-dir", required=True, type=Path)
    parser.add_argument("--known-null-dir", required=True, type=Path)
    parser.add_argument("--known-jackknife-dir", required=True, type=Path)
    parser.add_argument("--unresolved-null-dir", required=True, type=Path)
    parser.add_argument("--unresolved-jackknife-dir", required=True, type=Path)
    parser.add_argument("--identity-table", required=True, type=Path)
    parser.add_argument("--count-matched-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    reports: dict[str, dict[str, object]] = {}
    report_paths: dict[str, Path] = {}
    for label, directory in (
        ("index", args.index_dir),
        ("retrieval", args.retrieval_dir),
        ("null", args.null_dir),
        ("jackknife", args.jackknife_dir),
    ):
        reports[label], report_paths[label] = load_report(directory, label)

    world_reports: dict[str, tuple[dict[str, object], dict[str, object], Path, Path]] = {}
    for world, null_dir, jackknife_dir in (
        ("known", args.known_null_dir, args.known_jackknife_dir),
        ("unresolved", args.unresolved_null_dir, args.unresolved_jackknife_dir),
    ):
        null_report, null_path = load_report(null_dir, "null")
        jack_report, jack_path = load_report(jackknife_dir, "jackknife")
        if null_report.get("evidence_world") != world or jack_report.get("evidence_world") != world:
            raise RuntimeError(f"{world} calibration report has the wrong evidence world")
        world_reports[world] = (null_report, jack_report, null_path, jack_path)

    study_manifest = pd.read_csv(args.index_dir / "study_contrast_manifest.tsv", sep="\t")
    retrieval = pd.read_csv(args.retrieval_dir / "retrieval_scores.tsv.gz", sep="\t")
    if len(study_manifest) != 15 or study_manifest.study_id.nunique() != 3:
        raise RuntimeError("expected 15 response objects from three studies")
    identity_available = study_manifest["identity_partition_available"].astype(str).str.lower().isin(["true", "1"])
    if not identity_available.all():
        raise RuntimeError("study response objects lack the frozen identity partition")
    if int(study_manifest["n_entities_known"].sum()) <= 0 or int(study_manifest["n_entities_unresolved"].sum()) <= 0:
        raise RuntimeError("known/unresolved evidence worlds are empty")
    if reports["index"].get("identity_table_sha256") != sha256(args.identity_table):
        raise RuntimeError("response-index identity partition provenance differs from the frozen table")
    expected_rows = 15 * 10 * 3
    if len(retrieval) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} three-world candidate scores, observed {len(retrieval)}")
    if set(retrieval.evidence_world.astype(str)) != {"all", "known", "unresolved"}:
        raise RuntimeError("retrieval worlds are incomplete")
    if int(reports["null"].get("permutations", 0)) != 1999:
        raise RuntimeError("the frozen null must contain 1999 permutations")
    if reports["null"].get("evidence_world") != "all" or reports["jackknife"].get("evidence_world") != "all":
        raise RuntimeError("primary calibration reports are not the all-entity world")
    if int(reports["jackknife"].get("response_objects", 0)) != 15:
        raise RuntimeError("jackknife response-object count differs from the index")
    count_matched_report_path = args.count_matched_dir / "report.json"
    count_matched_report = json.loads(count_matched_report_path.read_text(encoding="utf-8"))
    if count_matched_report.get("status") != "PRETRUTH_COUNT_MATCHED_UNRESOLVED_SCORES_FROZEN":
        raise RuntimeError("count-matched unresolved scores are not frozen")
    if count_matched_report.get("truth_opened") is not False:
        raise RuntimeError("count-matched control did not preserve truth-blind state")
    if int(count_matched_report.get("denominators", {}).get("resamples", 0)) != 16:
        raise RuntimeError("count-matched control must contain 16 frozen resamples")

    files = {
        "index_report": report_paths["index"],
        "study_manifest": args.index_dir / "study_contrast_manifest.tsv",
        "study_sketches": args.index_dir / "study_contrast_sketches.npz",
        "panel_manifest": args.index_dir / "panel_contrast_manifest.tsv",
        "panel_sketches": args.index_dir / "panel_contrast_sketches.npz",
        "retrieval_report": report_paths["retrieval"],
        "retrieval_scores": args.retrieval_dir / "retrieval_scores.tsv.gz",
        "null_report": report_paths["null"],
        "null_archive": args.null_dir / "response_assignment_null.npz",
        "jackknife_report": report_paths["jackknife"],
        "jackknife_scores": args.jackknife_dir / "single_panel_deletion_scores.tsv.gz",
        "identity_partition": args.identity_table,
        "count_matched_report": count_matched_report_path,
        "count_matched_scores": args.count_matched_dir / "count_matched_retrieval_scores.tsv.gz",
        "count_matched_ledger": args.count_matched_dir / "resample_entity_ledger.tsv.gz",
    }
    for world, (null_report, jack_report, null_path, jack_path) in world_reports.items():
        if int(null_report.get("permutations", 0)) != 1999:
            raise RuntimeError(f"{world} null must contain 1999 permutations")
        if int(jack_report.get("response_objects", 0)) != 15:
            raise RuntimeError(f"{world} jackknife response-object count differs from the index")
        files[f"{world}_null_report"] = null_path
        files[f"{world}_null_archive"] = (
            args.known_null_dir if world == "known" else args.unresolved_null_dir
        ) / "response_assignment_null.npz"
        files[f"{world}_jackknife_report"] = jack_path
        files[f"{world}_jackknife_scores"] = (
            args.known_jackknife_dir if world == "known" else args.unresolved_jackknife_dir
        ) / "single_panel_deletion_scores.tsv.gz"
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"

    args.output_dir.mkdir(parents=True)
    receipt = {
        "schema": "factorial_hfd_pretruth_freeze_v1",
        "status": "PRETRUTH_ARTIFACTS_HASH_FROZEN",
        "data_role": "development_consumed",
        "denominators": {
            "studies": 3,
            "response_objects": 15,
            "candidates_per_query": 10,
            "evidence_worlds": 3,
            "retrieval_rows": len(retrieval),
            "permutations": 1999,
            "separately_calibrated_evidence_worlds": 3,
            "count_matched_unresolved_resamples": 16,
            "jackknife_rows": int(reports["jackknife"].get("scored_rows", 0)),
        },
        "truth_opened": False,
        "leakage_status": "all response objects, candidate scores, nulls and deletion sensitivities were hash-frozen before the two registered Hi-MYC reciprocal targets were adjudicated",
        "git_commit": git_commit,
        "sha256": {name: sha256(path) for name, path in files.items()},
        "claim_boundary": [
            "This is a development/consumed replicated-model test, not independent-laboratory confirmation.",
            "The all-entity world does not by itself establish an unresolved-entity increment.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
