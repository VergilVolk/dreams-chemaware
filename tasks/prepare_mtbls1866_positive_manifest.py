#!/usr/bin/env python3
"""Freeze the official 109-file MTBLS1866 primary positive-DDA universe.

Only ISA sample/assay metadata and remote byte sizes are read.  No abundance,
MS1/MS2 payload, response effect, or cross-study score is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd


BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1866/"
SAMPLE_FILE = "s_MTBLS1866.txt"
ASSAY_FILE = "a_MTBLS1866_LC-MS_positive_reverse-phase_metabolite_profiling.txt"
PRIMARY_GROUPS = {"COVID-19": 83, "HEALTHY": 26}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(relative: str) -> bytes:
    request = urllib.request.Request(
        BASE + quote(relative, safe="/"),
        headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def remote_size(url: str) -> int:
    error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = response.headers.get("Content-Length")
                if value is None:
                    raise RuntimeError("HEAD response lacks Content-Length")
                return int(value)
        except Exception as exc:
            error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"unable to read remote size: {url}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/mtbls1866_positive_manifest_v1_20261022"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between one and four")

    sample_bytes = fetch(SAMPLE_FILE)
    assay_bytes = fetch(ASSAY_FILE)
    samples = pd.read_csv(
        io.BytesIO(sample_bytes), sep="\t", dtype=str, keep_default_na=False
    )
    assays = pd.read_csv(
        io.BytesIO(assay_bytes), sep="\t", dtype=str, keep_default_na=False
    )
    if samples["Sample Name"].duplicated().any() or assays["Sample Name"].duplicated().any():
        raise RuntimeError("official sample or positive assay table has duplicate Sample Name")
    design = samples[
        ["Source Name", "Sample Name", "Comment[Patient status]", "Factor Value[Medical case]"]
    ].merge(
        assays[["Sample Name", "Derived Spectral Data File"]],
        on="Sample Name",
        how="inner",
        validate="one_to_one",
    )
    primary = design.loc[
        design["Factor Value[Medical case]"].isin(PRIMARY_GROUPS)
    ].copy()
    observed = primary.groupby("Factor Value[Medical case]").size().astype(int).to_dict()
    if observed != PRIMARY_GROUPS or len(primary) != 109:
        raise RuntimeError(f"primary topology drift: {observed}")
    if primary["Source Name"].nunique() != 109 or primary["Sample Name"].nunique() != 109:
        raise RuntimeError("primary source/sample names are not one-to-one biological units")
    if primary["Derived Spectral Data File"].duplicated().any():
        raise RuntimeError("primary positive mzML paths are not unique")

    primary["official_url"] = primary["Derived Spectral Data File"].map(
        lambda value: BASE + quote(str(value), safe="/")
    )
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(remote_size, url): url for url in primary.official_url}
        for number, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            sizes[url] = future.result()
            if number % 10 == 0 or number == len(futures):
                print(f"[size {number}/{len(futures)}]", flush=True)
    primary["expected_bytes"] = primary.official_url.map(sizes).astype("int64")
    if (primary.expected_bytes <= 0).any():
        raise RuntimeError("non-positive official mzML size")

    design_ledger = primary.rename(
        columns={
            "Factor Value[Medical case]": "group",
            "Comment[Patient status]": "patient_status",
            "Derived Spectral Data File": "relative_path",
        }
    )[
        [
            "Source Name", "Sample Name", "group", "patient_status",
            "relative_path", "official_url", "expected_bytes",
        ]
    ].sort_values(["group", "Sample Name"])
    acquisition = design_ledger[
        ["Sample Name", "relative_path", "official_url", "expected_bytes"]
    ].rename(columns={"Sample Name": "assay_name"}).sort_values("assay_name")
    acquisition["linkage_visibility"] = "phenotype_blind"

    args.output_dir.mkdir(parents=True)
    design_path = args.output_dir / "design_ledger.tsv"
    acquisition_path = args.output_dir / "phenotype_blind_acquisition_manifest.tsv"
    design_ledger.to_csv(design_path, sep="\t", index=False)
    acquisition.to_csv(acquisition_path, sep="\t", index=False)
    report = {
        "schema": "mtbls1866_positive_primary_manifest_v1",
        "status": "PRIMARY_POSITIVE_DDA_UNIVERSE_FROZEN",
        "evidence_role": "development_consumed_acquisition_only",
        "denominators": {
            "official_samples": len(samples),
            "official_positive_assays": len(assays),
            "primary_biological_units": len(primary),
            "noncritical_covid": int(observed["COVID-19"]),
            "healthy": int(observed["HEALTHY"]),
            "primary_mzml_files": len(acquisition),
            "primary_mzml_bytes": int(acquisition.expected_bytes.sum()),
        },
        "excluded_groups": {
            "ICU-COVID-19": 18,
            "NON-COVID-19": 16,
            "ICU": 12,
        },
        "provenance": {
            "sample_table_sha256": sha256_bytes(sample_bytes),
            "positive_assay_table_sha256": sha256_bytes(assay_bytes),
        },
        "leakage_status": (
            "ISA design and file sizes only; no mzML payload, abundance matrix, "
            "response effect, identity result, or retrieval score opened"
        ),
        "claim_boundary": [
            "The frozen contrast is non-critical COVID versus healthy plasma, not ICU severity or treatment response.",
            "File presence does not establish an all-feature response until phenotype-blind feature processing and QC complete.",
            "Shared COVID naming with another study is not signed response truth.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
