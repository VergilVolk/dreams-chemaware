#!/usr/bin/env python
"""Download and checksum one file from a Zenodo record.

The downloader is intentionally data-only: it never imports a model or opens
benchmark labels for performance analysis. Partial downloads are resumed when
the server honours HTTP Range; the completed file is atomically renamed only
after its deposited checksum passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-benchmark-fetch/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="21346580")
    parser.add_argument("--file", default="enveda-180-filtered.mgf.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    metadata = api_json(f"https://zenodo.org/api/records/{args.record}")
    matches = [item for item in metadata.get("files", []) if item.get("key") == args.file]
    if len(matches) != 1:
        available = sorted(item.get("key", "") for item in metadata.get("files", []))
        raise RuntimeError(f"Zenodo file {args.file!r} not unique; available={available}")
    item = matches[0]
    checksum = str(item["checksum"])
    algorithm, expected = checksum.split(":", 1)
    expected_size = int(item["size"])
    url = item["links"]["self"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema": "zenodo_file_snapshot_v1",
        "record": str(args.record),
        "record_created": metadata.get("created"),
        "record_updated": metadata.get("updated"),
        "doi": metadata.get("doi"),
        "title": metadata.get("metadata", {}).get("title"),
        "file": args.file,
        "size": expected_size,
        "checksum": checksum,
        "url": url,
        "performance_scores_opened": False,
    }
    (args.out_dir / "zenodo_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    if args.metadata_only:
        print(json.dumps(snapshot, indent=2))
        return

    target = args.out_dir / args.file
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if target.stat().st_size == expected_size and digest_file(target, algorithm) == expected:
            print(f"ZENODO_FILE_ALREADY_VERIFIED: {target}")
            return
        raise RuntimeError(f"existing target fails deposited size/checksum: {target}")

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DreaMS-benchmark-fetch/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            partial.unlink()
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as writer:
            while True:
                block = response.read(8 << 20)
                if not block:
                    break
                writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
                print(f"downloaded={writer.tell():,}/{expected_size:,}", flush=True)

    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"download size mismatch: {partial.stat().st_size} != {expected_size}")
    actual = digest_file(partial, algorithm)
    if actual != expected:
        raise RuntimeError(f"download checksum mismatch: {actual} != {expected}")
    partial.replace(target)
    print(f"ZENODO_FILE_VERIFIED: {target}")


if __name__ == "__main__":
    main()
