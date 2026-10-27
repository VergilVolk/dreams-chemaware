#!/usr/bin/env python3
"""Materialize the frozen Full23 Additive decision package without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "deploy" / "unified_additive_full23_v1"
ARCHITECTURE_REPORT = (
    ROOT / "docs" / "UNIFIED_ADDITIVE_FULL23_ARCHITECTURE_METRICS_RELEASE_20261027.md"
)
RUNTIME_SOURCES = (
    ROOT / "tasks" / "unified_additive_release.py",
    ROOT / "tasks" / "enveda_unified_candidate_integrator_core.py",
    ROOT / "tasks" / "massspecgym_embedding_readout_core.py",
    ROOT / "tasks" / "test_unified_additive_release.py",
    ROOT / "tasks" / "validate_unified_additive_release_bundle.py",
)
EXPECTED_CHECKPOINT_SHA256 = (
    "7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    """Hash tracked text using Git's canonical LF representation.

    The frozen report identities were registered from Linux/LF bytes.  Git may
    materialize the same JSON as CRLF on Windows, so source validation must not
    confuse a checkout convention with scientific-content drift.  Bundle file
    checksums remain raw-byte hashes below.
    """
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = SOURCE_PACKAGE / "manifest.json"
    checkpoint = SOURCE_PACKAGE / "assets" / "unified_candidate_integrator.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("tracked Full23 release sources are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dreams_unified_additive_full23_v1":
        raise RuntimeError("Full23 release manifest schema mismatch")
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Full23 tracked checkpoint hash mismatch")
    if manifest["decision_layer"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Full23 manifest/checkpoint identity mismatch")
    reports = manifest.get("selection_evidence", {}).get("reports", [])
    if len(reports) != 4:
        raise RuntimeError("Full23 source package lacks the four selection reports")
    for report in reports:
        path = SOURCE_PACKAGE / str(report.get("path", ""))
        if not path.is_file() or sha256_text_lf(path) != str(report.get("sha256", "")):
            raise RuntimeError(f"Full23 selection-evidence mismatch: {path}")
    frozen_sources = manifest.get("runtime_source_sha256_at_freeze", [])
    if len(frozen_sources) != 2:
        raise RuntimeError("Full23 frozen core-source registry is incomplete")
    for source in frozen_sources:
        path = ROOT / str(source.get("path", ""))
        if not path.is_file() or sha256(path) != str(source.get("sha256", "")):
            raise RuntimeError(f"Full23 frozen core-source drift: {path}")
    for path in RUNTIME_SOURCES:
        if not path.is_file():
            raise RuntimeError(f"missing existing runtime source: {path}")
    if not ARCHITECTURE_REPORT.is_file():
        raise RuntimeError("Full23 architecture/metrics report is missing")

    shutil.copytree(SOURCE_PACKAGE, output)
    runtime = output / "tasks"
    runtime.mkdir()
    for source in RUNTIME_SOURCES:
        shutil.copy2(source, runtime / source.name)
    shutil.copy2(ARCHITECTURE_REPORT, output / "ARCHITECTURE_AND_METRICS.md")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksums = output / "SHA256SUMS.txt"
    checksums.write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in files
        ) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "UNIFIED_ADDITIVE_FULL23_RELEASE_BUNDLE_COMPLETE",
        "schema": manifest["schema"],
        "profile": "decision_layer",
        "output": str(output),
        "decision_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "files": len(files) + 2,
        "upstream_assets_embedded": False,
        "upstream_asset_policy": (
            "large frozen encoders/rerankers remain external and are all registered by "
            "path, role and SHA256 in manifest.json"
        ),
    }
    report_path = output / "bundle_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with checksums.open("a", encoding="utf-8") as handle:
        handle.write(f"{sha256(report_path)}  bundle_report.json\n")

    archive_path = None
    if args.archive:
        archive_path = output.with_suffix(".tar.gz")
        if archive_path.exists():
            raise FileExistsError(archive_path)
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(output, arcname=output.name)
        report["archive"] = str(archive_path)
        report["archive_sha256"] = sha256(archive_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
