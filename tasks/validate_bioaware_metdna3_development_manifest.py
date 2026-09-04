#!/usr/bin/env python
"""Validate scope, hashes, and 30/70 rotations of the MetDNA3 dev manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    report_path = directory / "report.json"
    truth_path = directory / "development_level1.csv.gz"
    split_path = directory / "identity_splits.csv.gz"
    download_path = directory / "download_manifest.json"
    for path in [report_path, truth_path, split_path, download_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_metdna3_development_manifest_complete":
        raise RuntimeError("unexpected development report status")
    if report.get("internal_validation_opened") or report.get("external_test_opened"):
        raise RuntimeError("development manifest claims non-development outcomes were opened")
    for key, path in [
        ("truth_sha256", truth_path),
        ("split_sha256", split_path),
        ("download_manifest_sha256", download_path),
    ]:
        if report["provenance"][key] != sha256(path):
            raise RuntimeError(f"development artifact hash mismatch: {path}")
    truth = pd.read_csv(truth_path)
    if len(truth) != 751 or set(truth["LC-MS"]) != {"HILIC-MS(+)", "HILIC-MS(-)"}:
        raise RuntimeError("truth file escaped frozen HILIC scope")
    if set(truth["confidence_level"]) != {"level1"}:
        raise RuntimeError("truth file contains non-Level-1 annotations")
    split = pd.read_csv(split_path)
    if split["fold"].nunique() != 10 or split["ik14"].nunique() != 484:
        raise RuntimeError("unexpected split dimensions")
    role_counts = split.groupby("ik14")["role"].value_counts().unstack(fill_value=0)
    if not ((role_counts["seed"] == 3) & (role_counts["heldout"] == 7)).all():
        raise RuntimeError("every identity must be seed 3 times and held out 7 times")
    download = json.loads(download_path.read_text(encoding="utf-8"))["files"]
    if len(download) != 16:
        raise RuntimeError("download manifest must contain exactly 16 files")
    if any(
        "NIST_urine_hilic/" not in row["filepath"]
        or ("_pos_" not in row["filepath"] and "_neg_" not in row["filepath"])
        for row in download
    ):
        raise RuntimeError("download manifest escaped frozen development panel")
    if sum(int(row["bytes"]) for row in download) != 902_191_840:
        raise RuntimeError("development download byte total changed")
    print(
        "[validate_bioaware_metdna3_development_manifest] PASS "
        "rows=751 identities=484 files=16",
        flush=True,
    )


if __name__ == "__main__":
    main()
