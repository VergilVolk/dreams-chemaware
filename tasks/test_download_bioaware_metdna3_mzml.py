#!/usr/bin/env python
"""Offline tests for frozen-manifest download and resume semantics."""
from __future__ import annotations

import tempfile
from pathlib import Path

from download_bioaware_metdna3_development_mzml import download, sha256


def main() -> None:
    payload = b"bioaware-frozen-download-test"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.bin"
        source.write_bytes(payload)
        row = {"bytes": len(payload), "url": source.as_uri()}
        destination = root / "destination.bin"
        download(row, destination)
        assert destination.read_bytes() == payload
        first_hash = sha256(destination)
        download(row, destination)
        assert sha256(destination) == first_hash
        bad = root / "bad.bin"
        bad.write_bytes(b"bad")
        try:
            download(row, bad)
        except RuntimeError as error:
            assert "size mismatch" in str(error)
        else:
            raise AssertionError("mismatched existing file did not fail closed")
    print("[test_download_bioaware_metdna3_mzml] PASS")


if __name__ == "__main__":
    main()
