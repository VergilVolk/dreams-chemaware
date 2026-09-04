#!/usr/bin/env python3
"""Fail-closed validator for the MTBLS13729 biology Package A release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/mtbls13729/biology_package_a_release_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    for name in ("report.json", "biology_claim_ledger_v1.csv", "artifact_manifest_v1.csv", "README.md"):
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(args.output_dir / name)
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_biology_package_a_release_v1_complete" or report.get("formal") is not True:
        raise RuntimeError("release status mismatch")
    if report.get("package_A_ready") is not True or report.get("package_B_ready") is not False or report.get("package_C_ready") is not False:
        raise RuntimeError("package readiness mismatch")
    claims = pd.read_csv(args.output_dir / "biology_claim_ledger_v1.csv")
    if claims["claim_id"].tolist() != [f"B{i:02d}" for i in range(1, 11)]:
        raise RuntimeError("claim ledger changed")
    expected = {"PASS_DISCOVERY": 2, "PASS_CONTEXT": 2, "NEGATIVE_RESULT": 3, "MISSING": 3}
    if claims["status"].value_counts().to_dict() != expected:
        raise RuntimeError("claim status counts changed")
    manifest = pd.read_csv(args.output_dir / "artifact_manifest_v1.csv")
    if len(manifest) != 14 or manifest["artifact"].duplicated().any():
        raise RuntimeError("artifact manifest changed")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != row.sha256:
            raise RuntimeError(f"artifact drift: {row.artifact}")
    if sha256(args.output_dir / "artifact_manifest_v1.csv") != report["manifest_sha256"]:
        raise RuntimeError("manifest digest mismatch")
    if sha256(args.output_dir / "biology_claim_ledger_v1.csv") != report["claim_ledger_sha256"]:
        raise RuntimeError("claim ledger digest mismatch")
    if "does not provide" not in report.get("claim_limit", ""):
        raise RuntimeError("claim boundary missing")
    print("[validate_mtbls13729_biology_package_a_release_v1] PASS claims=10 artifacts=14")


if __name__ == "__main__":
    main()
