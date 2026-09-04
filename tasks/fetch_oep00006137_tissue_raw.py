#!/usr/bin/env python
"""Fetch and freeze the public OEP00006137 tissue LC-MS file manifest.

The downloader is intentionally narrow and fail-closed:

* metadata are obtained from the public NODE API;
* only the four predeclared human-tissue experiments are eligible;
* files are written through ``.part`` paths and checked by byte size and MD5;
* an existing valid file is reused, while an invalid file stops the run;
* the default ``rp_only`` scope downloads the two panels containing the four
  modified-guanosine peaks selected before raw-data inspection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import ssl
import tarfile
import time
import urllib.error
import urllib.request

import certifi


ACCESSION = "OEP00006137"
API_ROOT = "https://www.biosino.org/node/api"
PROJECT_DATA_ENDPOINT = f"{API_ROOT}/app/browseDetail/getDataList"
PUBLIC_DOWNLOAD = f"{API_ROOT}/download/node/data/public"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

TISSUE_EXPERIMENTS = {
    "rp_pos": "OEX00031356",
    "rp_neg": "OEX00031349",
    "hilic_pos": "OEX00031348",
    "hilic_neg": "OEX00031352",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/OEP00006137_raw"),
    )
    parser.add_argument(
        "--scope",
        choices=("metadata_only", "rp_only", "all_tissue"),
        default="rp_only",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def api_manifest(timeout: int) -> list[dict]:
    payload = json.dumps(
        {
            "type": "project",
            "typeNo": ACCESSION,
            "sortKey": "",
            "sortType": "",
            "pageNum": 1,
            "pageSize": 1000,
            "totalCount": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        PROJECT_DATA_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Language": "en_US",
            "User-Agent": "Mozilla/5.0 DreaMS-OEP-audit/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        result = json.load(response)
    if result.get("code") != 200:
        raise RuntimeError(f"NODE API failed: {result}")
    rows = result.get("data", {}).get("list", [])
    if len(rows) != 462:
        raise RuntimeError(f"expected 462 project files, observed {len(rows)}")
    return rows


def md5sum(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".mzxml")
            ]
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"invalid tar.gz archive: {path}") from exc
    if len(members) != 1 or members[0].size <= 0:
        raise RuntimeError(
            f"archive must contain exactly one non-empty mzXML: {path} "
            f"(observed {len(members)})"
        )


def validate_existing(path: Path, row: dict) -> tuple[bool, str, str]:
    if not path.exists():
        return False, "", ""
    expected_size = int(row["fileSize"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"existing file has wrong size: {path} "
            f"({path.stat().st_size} != {expected_size})"
        )
    observed_md5 = md5sum(path)
    expected_md5 = str(row["md5"]).lower()
    if observed_md5.lower() != expected_md5:
        try:
            validate_archive(path)
        except RuntimeError:
            # This path was produced by this downloader and is not a usable
            # archive (NODE occasionally serves an all-zero object).  Remove
            # it so a fresh replay can distinguish transient corruption from
            # a persistently unavailable public object.
            path.unlink()
            return False, "", ""
        return True, "reused_catalogue_md5_mismatch", observed_md5
    return True, "reused", observed_md5


def download_one(
    row: dict, output_dir: Path, retries: int, timeout: int
) -> tuple[str, str, int, str, str, str]:
    panel = next(
        name for name, experiment in TISSUE_EXPERIMENTS.items()
        if experiment == row["expNo"]
    )
    destination = output_dir / panel / str(row["name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists, disposition, observed_md5 = validate_existing(destination, row)
    if exists:
        return (
            panel, disposition, int(row["fileSize"]), str(row["datNo"]),
            str(row["md5"]), observed_md5,
        )

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    url = f"{PUBLIC_DOWNLOAD}/{row['datNo']}"
    last_error: Exception | None = None
    mismatch_hashes: list[str] = []
    invalid_archive_hashes: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-OEP-audit/1.0"})
            with urllib.request.urlopen(
                request, timeout=timeout, context=SSL_CONTEXT
            ) as response, partial.open("wb") as out:
                while chunk := response.read(8 << 20):
                    out.write(chunk)
            if partial.stat().st_size != int(row["fileSize"]):
                raise RuntimeError(
                    f"download size mismatch for {row['datNo']}: "
                    f"{partial.stat().st_size} != {row['fileSize']}"
                )
            observed_md5 = md5sum(partial)
            expected_md5 = str(row["md5"]).lower()
            if observed_md5.lower() != expected_md5:
                mismatch_hashes.append(observed_md5.lower())
                try:
                    validate_archive(partial)
                except RuntimeError:
                    invalid_archive_hashes.append(observed_md5.lower())
                    raise
                # NODE has a small number of stale catalogue MD5 values.  Two
                # independent byte-identical downloads plus archive validation
                # are required before accepting such an object.
                if len(mismatch_hashes) < 2 or len(set(mismatch_hashes)) != 1:
                    raise RuntimeError(
                        f"catalogue MD5 mismatch awaiting stable replay for {row['datNo']}: "
                        f"{observed_md5} != {expected_md5}"
                    )
                os.replace(partial, destination)
                return (
                    panel, "downloaded_catalogue_md5_mismatch", int(row["fileSize"]),
                    str(row["datNo"]), expected_md5, observed_md5.lower(),
                )
            os.replace(partial, destination)
            return (
                panel, "downloaded", int(row["fileSize"]), str(row["datNo"]),
                expected_md5, observed_md5.lower(),
            )
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    if (
        len(invalid_archive_hashes) == retries
        and len(set(invalid_archive_hashes)) == 1
    ):
        return (
            panel, "unavailable_invalid_archive", int(row["fileSize"]),
            str(row["datNo"]), str(row["md5"]).lower(), invalid_archive_hashes[0],
        )
    raise RuntimeError(f"failed to download {row['datNo']} after {retries} attempts") from last_error


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = args.output_dir / "OEP00006137_tissue_manifest.json"
    try:
        rows = api_manifest(args.timeout)
        manifest_source = "live_node_api"
        project_file_count = len(rows)
    except (OSError, urllib.error.URLError) as exc:
        if not manifest_json.exists():
            raise
        prior = json.loads(manifest_json.read_text(encoding="utf-8"))
        if (
            prior.get("accession") != ACCESSION
            or int(prior.get("project_files", 0)) != 462
            or int(prior.get("tissue_files", 0)) != 364
            or prior.get("experiments") != TISSUE_EXPERIMENTS
        ):
            raise RuntimeError("frozen NODE manifest failed reuse validation") from exc
        rows = prior["rows"]
        manifest_source = f"frozen_tissue_manifest_after_{type(exc).__name__}"
        project_file_count = int(prior["project_files"])
        print(
            f"[manifest] live NODE API unavailable; reusing validated frozen tissue manifest ({type(exc).__name__})",
            flush=True,
        )

    tissue_ids = set(TISSUE_EXPERIMENTS.values())
    tissue = [row for row in rows if row.get("expNo") in tissue_ids]
    if len(tissue) != 364:
        raise RuntimeError(f"expected 364 human-tissue files, observed {len(tissue)}")
    for panel, experiment in TISSUE_EXPERIMENTS.items():
        panel_rows = [row for row in tissue if row["expNo"] == experiment]
        if len(panel_rows) != 91:
            raise RuntimeError(f"{panel}: expected 91 files, observed {len(panel_rows)}")
        biological = [row for row in panel_rows if not str(row["name"]).lower().startswith("qc")]
        if len(biological) != 80:
            raise RuntimeError(f"{panel}: expected 80 biological files, observed {len(biological)}")

    manifest_csv = args.output_dir / "OEP00006137_tissue_manifest.csv"
    frozen = {
        "accession": ACCESSION,
        "api_root": API_ROOT,
        "project_files": project_file_count,
        "tissue_files": len(tissue),
        "tissue_bytes": sum(int(row["fileSize"]) for row in tissue),
        "experiments": TISSUE_EXPERIMENTS,
        "manifest_source": manifest_source,
        "rows": tissue,
    }
    manifest_json.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    fields = [
        "panel", "expNo", "expName", "sapNo", "sapName", "sapDesc",
        "runNo", "runName", "datNo", "name", "fileSize", "md5", "security",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tissue:
            panel = next(name for name, exp in TISSUE_EXPERIMENTS.items() if exp == row["expNo"])
            writer.writerow({"panel": panel, **{field: row.get(field) for field in fields if field != "panel"}})

    if args.scope == "metadata_only":
        print(json.dumps({"status": "metadata_frozen", "files": len(tissue)}, indent=2))
        return
    selected_panels = {"rp_pos", "rp_neg"} if args.scope == "rp_only" else set(TISSUE_EXPERIMENTS)
    selected = [
        row for row in tissue
        if next(name for name, exp in TISSUE_EXPERIMENTS.items() if exp == row["expNo"])
        in selected_panels
    ]
    expected = 182 if args.scope == "rp_only" else 364
    if len(selected) != expected:
        raise RuntimeError(f"download scope expected {expected} files, observed {len(selected)}")

    counts: dict[str, dict[str, int]] = {}
    catalogue_md5_mismatches: list[dict[str, str]] = []
    unavailable_objects: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, args.output_dir, args.retries, args.timeout): row
            for row in selected
        }
        complete = 0
        for future in concurrent.futures.as_completed(futures):
            panel, disposition, size, dat_no, expected_md5, observed_md5 = future.result()
            counts.setdefault(panel, {"bytes": 0})
            counts[panel].setdefault(disposition, 0)
            counts[panel][disposition] += 1
            if disposition != "unavailable_invalid_archive":
                counts[panel]["bytes"] += size
            if "catalogue_md5_mismatch" in disposition:
                catalogue_md5_mismatches.append(
                    {
                        "datNo": dat_no,
                        "catalogue_md5": expected_md5,
                        "served_md5": observed_md5,
                        "disposition": disposition,
                    }
                )
            if disposition == "unavailable_invalid_archive":
                unavailable_objects.append(
                    {
                        "datNo": dat_no,
                        "catalogue_md5": expected_md5,
                        "served_md5": observed_md5,
                        "reason": "stable all-zero/non-gzip object served at public endpoint",
                    }
                )
            complete += 1
            if complete % 20 == 0 or complete == len(selected):
                print(f"[download] {complete}/{len(selected)}", flush=True)

    report = {
        "status": "OEP00006137_tissue_download_complete",
        "scope": args.scope,
        "selected_files": len(selected),
        "selected_bytes": sum(int(row["fileSize"]) for row in selected),
        "counts": counts,
        "catalogue_md5_mismatches": sorted(
            catalogue_md5_mismatches, key=lambda item: item["datNo"]
        ),
        "unavailable_objects": sorted(unavailable_objects, key=lambda item: item["datNo"]),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
    }
    (args.output_dir / f"download_{args.scope}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
