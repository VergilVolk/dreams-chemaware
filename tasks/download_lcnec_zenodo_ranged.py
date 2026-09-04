"""Reliably download the LCNEC proteogenomic archive with verified HTTP ranges.

Zenodo's file endpoint may restart or throttle open-ended/concurrent transfers.
This downloader uses finite byte ranges, validates every Content-Range and size,
and assembles the archive only after all chunks are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


URL = "https://zenodo.org/api/records/20922299/files/LCNEC_2026-SA.rar/content"
TOTAL_BYTES = 130_002_347
EXPECTED_MD5 = "1c3cb3dd041b6b23ccb5a84f25cd7714"
RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def digest(path: Path, algorithm: str = "md5") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_range(url: str, start: int, end: int, target: Path, retries: int) -> None:
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        print(f"[range] reuse {target.name} bytes={expected:,}", flush=True)
        return

    temporary = target.with_suffix(target.suffix + ".download")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "DreaMS-LCNEC-audit/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, received {response.status}")
                content_range = response.headers.get("Content-Range", "")
                match = RANGE_RE.fullmatch(content_range.strip())
                observed = tuple(map(int, match.groups())) if match else None
                if observed != (start, end, TOTAL_BYTES):
                    raise RuntimeError(
                        f"unexpected Content-Range {content_range!r}; "
                        f"expected bytes {start}-{end}/{TOTAL_BYTES}"
                    )
                with temporary.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            size = temporary.stat().st_size
            if size != expected:
                raise RuntimeError(f"short range: {size:,} != {expected:,}")
            os.replace(temporary, target)
            print(f"[range] complete {target.name} bytes={size:,}", flush=True)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"range {start}-{end} failed after {retries} attempts"
                ) from exc
            delay = min(60, 2**attempt)
            print(
                f"[range] retry {attempt}/{retries} for {start}-{end}: {exc}; "
                f"waiting {delay}s",
                flush=True,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/LCNEC_proteogenomic_2026"),
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/LCNEC_2026-SA.rar.partial"
        ),
    )
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.resolve()
    if not prefix.is_file():
        raise FileNotFoundError(prefix)
    prefix_size = prefix.stat().st_size
    if not 0 < prefix_size < TOTAL_BYTES:
        raise RuntimeError(f"invalid prefix size {prefix_size:,}")

    chunk_bytes = args.chunk_mib * 1024 * 1024
    chunks: list[Path] = []
    for index, start in enumerate(range(prefix_size, TOTAL_BYTES, chunk_bytes)):
        end = min(start + chunk_bytes - 1, TOTAL_BYTES - 1)
        target = args.output_dir / f"LCNEC_2026-SA.range_{index:02d}_{start}_{end}"
        download_range(args.url, start, end, target, args.retries)
        chunks.append(target)

    output = args.output_dir / "LCNEC_2026-SA.rar"
    temporary_output = output.with_suffix(".rar.assembling")
    with temporary_output.open("wb") as destination:
        for source in [prefix, *chunks]:
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    destination.write(block)
    if temporary_output.stat().st_size != TOTAL_BYTES:
        raise RuntimeError(
            f"assembled size {temporary_output.stat().st_size:,} != {TOTAL_BYTES:,}"
        )
    observed_md5 = digest(temporary_output)
    if observed_md5 != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {observed_md5} != {EXPECTED_MD5}")
    os.replace(temporary_output, output)
    print(f"[complete] {output.resolve()}", flush=True)
    print(f"[complete] bytes={output.stat().st_size:,} md5={observed_md5}", flush=True)


if __name__ == "__main__":
    main()
