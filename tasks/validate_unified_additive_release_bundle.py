#!/usr/bin/env python3
"""Validate a materialized Full23 Additive decision-layer bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SCHEMA = "dreams_unified_additive_full23_v1"
CHECKPOINT_SHA256 = "7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717"
REQUIRED_RUNTIME = (
    "unified_additive_release.py",
    "enveda_unified_candidate_integrator_core.py",
    "massspecgym_embedding_readout_core.py",
    "test_unified_additive_release.py",
    "validate_unified_additive_release_bundle.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    """Hash tracked text using Git's canonical LF representation."""
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("release manifest schema mismatch")
    if manifest.get("bundle_layers", {}).get("profile") != "decision_layer":
        raise RuntimeError("release bundle profile is ambiguous")
    decision = manifest.get("decision_layer", {})
    if decision.get("arm") != "additive_linear_control":
        raise RuntimeError("release does not explicitly pin the Additive arm")
    checkpoint = bundle / str(decision.get("checkpoint", ""))
    if not checkpoint.is_file() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("release checkpoint identity mismatch")
    if decision.get("sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("manifest checkpoint identity mismatch")
    selection = manifest.get("selection_evidence", {})
    reports = selection.get("reports", [])
    if len(reports) != 4:
        raise RuntimeError("release must include the four frozen selection reports")
    for report in reports:
        path = bundle / str(report.get("path", ""))
        digest = str(report.get("sha256", "")).lower()
        if not path.is_file() or sha256_text_lf(path) != digest:
            raise RuntimeError(f"selection-evidence identity mismatch: {path}")
    frozen_sources = manifest.get("runtime_source_sha256_at_freeze", [])
    if len(frozen_sources) != 2:
        raise RuntimeError("frozen core-source registry is incomplete")
    for source in frozen_sources:
        path = bundle / str(source.get("path", ""))
        digest = str(source.get("sha256", "")).lower()
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen core-source identity mismatch: {path}")
    for name in REQUIRED_RUNTIME:
        if not (bundle / "tasks" / name).is_file():
            raise RuntimeError(f"release runtime source is missing: {name}")
    if not (bundle / "ARCHITECTURE_AND_METRICS.md").is_file():
        raise RuntimeError("release architecture/metrics report is missing")
    status_path = bundle / "model_status.json"
    if not status_path.is_file():
        raise RuntimeError("release model status is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("schema") != "unified_additive_full23_model_status_v1"
        or status.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or status.get("decision_arm") != "additive_linear_control"
        or status.get("successor_experiments", {})
        .get("full23_v11_bounded_multireference", {})
        .get("status") != "not_promoted"
    ):
        raise RuntimeError("release model status drifted from the frozen decision")

    checksum_path = bundle / "SHA256SUMS.txt"
    registered: set[str] = set()
    for raw in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = raw.split("  ", 1)
        if relative in registered:
            raise RuntimeError(f"duplicate checksum entry: {relative}")
        registered.add(relative)
        path = bundle / Path(relative)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"release file checksum mismatch: {relative}")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if registered != actual:
        raise RuntimeError(
            f"release checksum coverage drifted: missing={sorted(actual-registered)}, "
            f"extra={sorted(registered-actual)}"
        )

    upstream = manifest.get("upstream_assets", {})
    for name, identity in upstream.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"malformed upstream registry item: {name}")
        hashes = [value for key, value in identity.items() if key.endswith("sha256")]
        for digest in hashes:
            digest = str(digest).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RuntimeError(f"malformed upstream SHA256: {name}")

    test_environment = dict(os.environ)
    test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-u", "tasks/test_unified_additive_release.py"],
        cwd=bundle,
        env=test_environment,
        check=True,
    )
    print(json.dumps({
        "status": "UNIFIED_ADDITIVE_FULL23_RELEASE_BUNDLE_VALID",
        "bundle": str(bundle),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "registered_files": len(registered),
    }, indent=2))


if __name__ == "__main__":
    main()
