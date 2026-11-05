from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import zipfile


sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_calms_v1_release_archive import (  # noqa: E402
    FIXED_ZIP_TIME,
    registered_payloads,
    validate_archive,
    write_archive,
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_deterministic_release_archive(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    payloads = {
        "README.md": b"release\n",
        "runtime/core.py": b"VALUE = 1\n",
    }
    (source / "runtime").mkdir()
    for relative, payload in payloads.items():
        path = source / relative
        path.write_bytes(payload)
    checksum_text = "".join(
        f"{digest(payload)}  {relative}\n"
        for relative, payload in sorted(payloads.items())
    )
    (source / "checksums.sha256").write_text(checksum_text, encoding="utf-8")

    registered = registered_payloads(source)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_archive(first, registered)
    write_archive(second, registered)
    validate_archive(first, registered)
    validate_archive(second, registered)
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first, "r") as archive:
        assert all(info.date_time == FIXED_ZIP_TIME for info in archive.infolist())


def test_unregistered_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    payload = b"registered\n"
    (source / "registered.txt").write_bytes(payload)
    (source / "extra.txt").write_text("extra\n", encoding="utf-8")
    (source / "checksums.sha256").write_text(
        f"{digest(payload)}  registered.txt\n",
        encoding="utf-8",
    )

    try:
        registered_payloads(source)
    except RuntimeError as error:
        assert "inventory" in str(error)
    else:
        raise AssertionError("unregistered release file was accepted")
