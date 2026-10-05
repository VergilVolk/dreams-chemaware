#!/usr/bin/env python3
"""Dependency-free integrity gate for the frozen BioAware B20 action artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "direct_actions.csv.gz": "c0739bc96c45efe18c0d529b61dd5f1c2e53a6c577aabd719e7438346180585a",
    "direct_action_manifest.npz": "74fd1ba07a84e76b393081de0cb29d20bdcb0d077811e04f6421243aaf24cb69",
    "report.json": "5bdff005b4b8843574f59e33ddcca057f0fe22428186f261d7513a5d0b8a8d89",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(directory: Path) -> dict:
    missing = [name for name in EXPECTED if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"frozen B20 artifact is incomplete at {directory}; missing={missing}"
        )
    observed = {name: sha256(directory / name) for name in EXPECTED}
    mismatched = {
        name: {"expected": EXPECTED[name], "observed": observed[name]}
        for name in EXPECTED
        if observed[name] != EXPECTED[name]
    }
    if mismatched:
        raise RuntimeError(f"frozen B20 SHA256 mismatch: {mismatched}")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b20_direct_action_manifest_complete":
        raise RuntimeError("frozen B20 report has wrong status")
    if report.get("pass_to_direct_gradient_canary") is not True:
        raise RuntimeError("frozen B20 report is not authorised")
    return {
        "status": "bioaware_b20_frozen_artifact_preflight_passed",
        "directory": str(directory),
        "sha256": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))


if __name__ == "__main__":
    main()
