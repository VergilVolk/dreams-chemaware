#!/usr/bin/env python
"""Independent fail-closed validation of the complete B47 B1 registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    report_path = directory / "report.json"
    registry_path = directory / "artifact_registry.csv.gz"
    for path in (report_path, registry_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b47_b1_artifact_registry_complete":
        raise RuntimeError("unexpected B47 B1 registry status")
    if report.get("formal") is not True or report.get("pass_b1_provenance") is not True:
        raise RuntimeError("B47 B1 provenance registry is incomplete")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("one or more B47 B1 gates failed")
    contract = report.get("contracts", {})
    if any(contract.get(key) is not False for key in (
        "truth_opened", "phenotype_used", "performance_computed", "model_fitted",
        "reaction_network_scored", "P2b_used",
    )):
        raise RuntimeError("B47 B1 truth-blind contract changed")
    if report.get("problems") or report.get("forbidden_headers"):
        raise RuntimeError("B47 B1 complete registry contains unresolved problems")
    if report.get("provenance", {}).get("artifact_registry_sha256") != sha256_file(registry_path):
        raise RuntimeError("B47 B1 registry hash mismatch")
    registry = pd.read_csv(registry_path)
    if len(registry) != int(report["artifact_files"]):
        raise RuntimeError("B47 B1 artifact row count mismatch")
    if not registry["present"].astype(bool).all():
        raise RuntimeError("B47 B1 complete registry contains missing files")
    expected = registry["expected_sha256"].notna()
    if not registry.loc[expected, "hash_matches_expected"].astype(bool).all():
        raise RuntimeError("B47 B1 preregistered hash mismatch")
    print(
        f"[validate_bioaware_b47_b1_artifact_registry] PASS "
        f"files={len(registry)} bytes={int(report['artifact_bytes']):,} "
        f"pass_to_b2={bool(report.get('pass_to_b2_exact_event'))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
