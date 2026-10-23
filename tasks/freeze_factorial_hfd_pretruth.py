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

    study_manifest = pd.read_csv(args.index_dir / "study_contrast_manifest.tsv", sep="\t")
    retrieval = pd.read_csv(args.retrieval_dir / "retrieval_scores.tsv.gz", sep="\t")
    if len(study_manifest) != 15 or study_manifest.study_id.nunique() != 3:
        raise RuntimeError("expected 15 response objects from three studies")
    expected_rows = 15 * 10 * 3
    if len(retrieval) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} three-world candidate scores, observed {len(retrieval)}")
    if set(retrieval.evidence_world.astype(str)) != {"all", "known", "unresolved"}:
        raise RuntimeError("retrieval worlds are incomplete")
    if int(reports["null"].get("permutations", 0)) != 1999:
        raise RuntimeError("the frozen null must contain 1999 permutations")
    if int(reports["jackknife"].get("response_objects", 0)) != 15:
        raise RuntimeError("jackknife response-object count differs from the index")

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
    }
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
