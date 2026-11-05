#!/usr/bin/env python3
"""Build and validate the deterministic standalone CALMS v1.0 release ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


ARCHIVE_ROOT = PurePosixPath("calms_v1")
FIXED_ZIP_TIME = (2026, 11, 5, 0, 0, 0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("deploy/calms_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables/CALMS_v1.0_20261105.zip"),
    )
    parser.add_argument(
        "--checksum-output",
        type=Path,
        default=None,
        help="Defaults to <output>.sha256.",
    )
    parser.add_argument(
        "--skip-release-test",
        action="store_true",
        help="Only for archive-builder contract tests; normal builds run the frozen test.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_payloads(source: Path) -> dict[str, bytes]:
    checksum_path = source / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    output: dict[str, bytes] = {}
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split("  ", 1)
        relative = relative.replace("\\", "/")
        if relative.startswith("/") or ".." in PurePosixPath(relative).parts:
            raise RuntimeError(f"unsafe registered path: {relative}")
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        if sha256_bytes(payload) != digest:
            raise RuntimeError(f"release checksum mismatch: {relative}")
        if relative in output:
            raise RuntimeError(f"duplicate registered path: {relative}")
        output[relative] = payload
    output["checksums.sha256"] = checksum_path.read_bytes()
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    registered = set(output) - {"checksums.sha256"}
    if registered != actual:
        missing = sorted(actual - registered)
        stale = sorted(registered - actual)
        raise RuntimeError(
            f"release checksum inventory mismatch: unregistered={missing}, stale={stale}"
        )
    return output


def zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(str(ARCHIVE_ROOT / relative), date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def write_archive(output: Path, payloads: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for relative, payload in sorted(payloads.items()):
                archive.writestr(
                    zip_info(relative),
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_archive(output: Path, payloads: dict[str, bytes]) -> None:
    expected_names = [str(ARCHIVE_ROOT / item) for item in sorted(payloads)]
    with zipfile.ZipFile(output, "r") as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != expected_names:
            raise RuntimeError("release archive membership or order drifted")
        if len(infos) != len({info.filename for info in infos}):
            raise RuntimeError("release archive contains duplicate paths")
        for info, relative in zip(infos, sorted(payloads)):
            if info.date_time != FIXED_ZIP_TIME:
                raise RuntimeError(f"release archive timestamp drifted: {relative}")
            if archive.read(info.filename) != payloads[relative]:
                raise RuntimeError(f"release archive payload drifted: {relative}")


def main() -> None:
    args = arguments()
    repository = Path(__file__).resolve().parents[1]
    source = args.source if args.source.is_absolute() else repository / args.source
    output = args.output if args.output.is_absolute() else repository / args.output
    checksum_output = args.checksum_output
    if checksum_output is None:
        checksum_output = output.with_name(output.name + ".sha256")
    elif not checksum_output.is_absolute():
        checksum_output = repository / checksum_output

    if not args.skip_release_test:
        subprocess.run(
            [sys.executable, str(source / "test_calms_frozen_release.py")],
            cwd=repository,
            check=True,
        )
    payloads = registered_payloads(source)
    write_archive(output, payloads)
    validate_archive(output, payloads)
    digest = sha256_file(output)
    checksum_output.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": "CALMS_V1_RELEASE_ARCHIVE_COMPLETE",
        "archive": str(output),
        "files": len(payloads),
        "bytes": output.stat().st_size,
        "sha256": digest,
    }, indent=2))


if __name__ == "__main__":
    main()
