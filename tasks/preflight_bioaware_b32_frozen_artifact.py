#!/usr/bin/env python3
"""Dependency-free integrity gate for the frozen BioAware B32 action bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "direct_actions.csv.gz": "a4262b52258af71f4d72144a4b81aac62e802b39589ce3a11a75e1bc9112e3d3",
    "direct_action_manifest.npz": "a2ad778d8690962f7d86b3ee5f2f6fa96dbb6faef06f85e1a267bd5d4ab9e9e7",
    "report.json": "71662b4776a2121af42e89da87719d465cc26880fa2c8c8606cbdea678352755",
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
        raise FileNotFoundError(f"frozen B32 artifact incomplete at {directory}; missing={missing}")
    observed = {name: sha256(directory / name) for name in EXPECTED}
    mismatch = {
        name: {"expected": EXPECTED[name], "observed": observed[name]}
        for name in EXPECTED if observed[name] != EXPECTED[name]
    }
    if mismatch:
        raise RuntimeError(f"frozen B32 SHA256 mismatch: {mismatch}")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b32_comprehensive_action_bank_complete":
        raise RuntimeError("frozen B32 report status changed")
    if report.get("pass_to_direct_gradient_canary") is not True:
        raise RuntimeError("frozen B32 action bank is not authorised")
    return {
        "status": "bioaware_b32_frozen_artifact_preflight_passed",
        "directory": str(directory),
        "sha256": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2), flush=True)


if __name__ == "__main__":
    main()
