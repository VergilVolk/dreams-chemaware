"""Sequentially download the public MTBLS1905 positive-mode patient mzML set.

This intentionally downloads only the 62 biological C/M/N samples.  QC and
blanks are already available separately.  Files are written to ``.part`` first
and atomically renamed only after HTTP transfer finishes; pre-existing final
files are preserved and skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


def download_python(url: str, destination: Path) -> int:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "ChemAware-MTBLS1905-reanalysis/1.0"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    return destination.stat().st_size


def download_curl(url: str, destination: Path) -> int:
    """Download through Windows curl when the Python/OpenSSL certificate store fails.

    ``--ssl-no-revoke`` avoids a known Windows certificate-store parsing failure in
    this environment; curl still validates the server certificate.  Downloading to
    a temporary suffix and renaming atomically preserves the original safety rule.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    command = [
        "curl.exe", "--fail", "--location", "--retry", "3", "--retry-all-errors",
        "--connect-timeout", "60", "--ssl-no-revoke", "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    partial.replace(destination)
    return destination.stat().st_size


def download(url: str, destination: Path, backend: str) -> int:
    if backend == "curl":
        return download_curl(url, destination)
    if backend == "python":
        return download_python(url, destination)
    try:
        return download_python(url, destination)
    except Exception as error:
        print(f"  Python download failed ({type(error).__name__}); retrying with curl.exe", flush=True)
        return download_curl(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_patient_download_manifest.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/external/MTBLS1905/positive_patients"))
    parser.add_argument("--backend", choices=("auto", "python", "curl"), default="auto")
    parser.add_argument("--attempts-per-file", type=int, default=3,
                        help="Retry transient DNS/HTTP failures without discarding completed files.")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    started = time.time()
    for ordinal, row in enumerate(manifest.itertuples(index=False), 1):
        destination = args.out_dir / row.file_name
        if destination.is_file() and destination.stat().st_size > 1024:
            status, size = "skipped_existing", destination.stat().st_size
        else:
            print(f"[{ordinal}/{len(manifest)}] {row.file_name}", flush=True)
            error = None
            size = 0
            for attempt in range(1, args.attempts_per_file + 1):
                try:
                    size = download(row.url, destination, args.backend)
                    status = "downloaded"
                    error = None
                    break
                except Exception as exc:  # retain the cohort download after transient network faults
                    error = repr(exc)
                    partial = destination.with_suffix(destination.suffix + ".part")
                    if partial.exists() and partial.stat().st_size == 0:
                        partial.unlink()
                    if attempt < args.attempts_per_file:
                        wait = 10 * attempt
                        print(f"  transfer failed (attempt {attempt}/{args.attempts_per_file}); retrying in {wait}s", flush=True)
                        time.sleep(wait)
            if error is not None:
                status = "failed"
                print(f"  failed after {args.attempts_per_file} attempts: {error}", flush=True)
        report.append({
            "sample_name": row.sample_name,
            "tissue_type": getattr(row, "tissue_type", None),
            "sample_role": getattr(row, "sample_role", None),
            "file_name": row.file_name,
            "bytes": int(size),
            "status": status,
            "error": error if status == "failed" else None,
        })
        print(f"  {status}: {size / 1024**2:.1f} MiB", flush=True)
    output = args.out_dir / "download_report.json"
    failures = [record for record in report if record["status"] == "failed"]
    output.write_text(json.dumps({"study": "MTBLS1905", "files": report, "seconds": time.time() - started}, indent=2), encoding="utf-8")
    print(f"Completed {len(report) - len(failures)}/{len(report)} downloads: {output}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} files still failed; rerun the same command later to resume only missing files.")


if __name__ == "__main__":
    main()
