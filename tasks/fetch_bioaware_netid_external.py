#!/usr/bin/env python
"""Download, verify, and atomically install the frozen NetID v1.0 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


RECORD_ID = "5508337"
DOI = "10.5281/zenodo.5508337"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/5508337/files/"
    "LiChenPU%2FNetID-v1.0.zip/content"
)
ARCHIVE_SIZE = 155_278_201
ARCHIVE_MD5 = "066cc59ad4b8fd9b1542c529a6bdcadd"

# These are the minimum immutable inputs needed for the external benchmark.
# Suffix matching is intentional: Zenodo/GitHub may wrap the release in one
# top-level directory, but the scientific relative paths must remain exact.
REQUIRED_SUFFIXES = (
    "LICENSE",
    "FDR_example/manual_curate.csv",
    "FDR_example/raw_data.csv",
    "FDR_example/known_library.csv",
    "Sc_neg/raw_data.csv",
    "Sc_neg/NetID_output.csv",
    "Sc_neg/cyto_nodes.csv",
    "Sc_neg/cyto_edges.csv",
    "dependent/known_library.csv",
)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    if "\\" in name:
        raise RuntimeError(f"unsafe archive member uses backslash: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe archive member: {name}")
    if ":" in path.parts[0]:
        raise RuntimeError(f"unsafe archive member drive prefix: {name}")
    return path


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_SIZE:
        raise RuntimeError(f"partial archive exceeds expected size: {partial}")
    headers = {"User-Agent": "DreaMS-BioAware-NetID-lock/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                current = handle.tell()
                if current % (25 * 1024 * 1024) < len(chunk):
                    print(f"[download] {current:,}/{ARCHIVE_SIZE:,} bytes", flush=True)
    observed_size = partial.stat().st_size
    if observed_size != ARCHIVE_SIZE:
        raise RuntimeError(
            f"download size mismatch: {observed_size} != {ARCHIVE_SIZE}; "
            "the .part file was retained for a safe resume"
        )
    observed_md5 = file_hash(partial, "md5")
    if observed_md5 != ARCHIVE_MD5:
        raise RuntimeError(f"archive MD5 mismatch: {observed_md5} != {ARCHIVE_MD5}")
    os.replace(partial, destination)


def _locate_required(root: Path) -> dict[str, Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    located: dict[str, Path] = {}
    for suffix in REQUIRED_SUFFIXES:
        normalized = suffix.replace("\\", "/")
        matches = [
            path
            for path in files
            if path.relative_to(root).as_posix() == normalized
            or path.relative_to(root).as_posix().endswith("/" + normalized)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"required release member {suffix!r} has {len(matches)} matches"
            )
        located[suffix] = matches[0]
    return located


def _validate_existing(output: Path, archive: Path) -> dict:
    manifest_path = output / "bioaware_netid_source_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"existing output lacks source manifest: {output}")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_netid_source_installed":
        raise RuntimeError("existing NetID source manifest has unexpected status")
    if report.get("archive_md5") != ARCHIVE_MD5:
        raise RuntimeError("existing NetID source manifest has unexpected archive MD5")
    located = _locate_required(output)
    expected = report.get("required_files", {})
    for suffix, path in located.items():
        observed = file_hash(path)
        if expected.get(suffix, {}).get("sha256") != observed:
            raise RuntimeError(f"existing installed file hash mismatch: {suffix}")
    if archive.exists() and file_hash(archive, "md5") != ARCHIVE_MD5:
        raise RuntimeError(f"existing archive MD5 mismatch: {archive}")
    print(f"[reuse] verified frozen NetID source: {output}", flush=True)
    return report


def install(archive: Path, output: Path) -> dict:
    if output.exists():
        return _validate_existing(output, archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".netid_v1_", dir=output.parent))
    try:
        seen: set[str] = set()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = _safe_member_path(member.filename)
                normalized = member_path.as_posix()
                if normalized in seen:
                    raise RuntimeError(f"duplicate archive member: {normalized}")
                seen.add(normalized)
                # Unix symlink mode in the upper 16 bits is forbidden.
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise RuntimeError(f"archive symlink is forbidden: {normalized}")
                target = temporary.joinpath(*member_path.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        located = _locate_required(temporary)
        required = {
            suffix: {
                "relative_path": str(path.relative_to(temporary).as_posix()),
                "bytes": int(path.stat().st_size),
                "sha256": file_hash(path),
            }
            for suffix, path in located.items()
        }
        report = {
            "status": "bioaware_netid_source_installed",
            "formal": True,
            "record_id": RECORD_ID,
            "doi": DOI,
            "archive_url": ARCHIVE_URL,
            "archive_bytes": int(archive.stat().st_size),
            "archive_md5": file_hash(archive, "md5"),
            "archive_sha256": file_hash(archive),
            "archive_members": len(seen),
            "required_files": required,
            "license_notice": (
                "The bundled LICENSE is authoritative; inspect it before redistribution."
            ),
        }
        (temporary / "bioaware_netid_source_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, output)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/external/netid_v1/NetID-v1.0.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/netid_v1/source"),
    )
    parser.add_argument("--url", default=ARCHIVE_URL)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output_dir.resolve()
    if not archive.exists():
        _download(args.url, archive)
    else:
        if archive.stat().st_size != ARCHIVE_SIZE:
            raise RuntimeError(f"archive size mismatch: {archive}")
        if file_hash(archive, "md5") != ARCHIVE_MD5:
            raise RuntimeError(f"archive MD5 mismatch: {archive}")
        print(f"[reuse] verified archive: {archive}", flush=True)
    install(archive, output)


if __name__ == "__main__":
    main()
