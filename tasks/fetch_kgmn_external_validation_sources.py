#!/usr/bin/env python3
"""Fetch and checksum the public KGMN external-validation supplements.

The files are author-owned immutable inputs.  This script does not interpret
outcomes; it only materializes a checksum-bound source bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


RECORD_ID = 7089991
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
FILES = {
    "Supplementary data1.xlsx": {
        "md5": "8eadc3821d6e6973cc81cb3596ef414b",
        "role": "manual 3451-peak identity and ion-form truth plus author outputs",
    },
    "Supplementary data2.xlsx": {
        "md5": "9a047288772908c6bb0d34573bb3b2f8",
        "role": "46STD known/expanded metabolite universe and reaction pairs",
    },
    "Supplementary data3.xlsx": {
        "md5": "3e936cbbb22863371213ff8825c9f006",
        "role": "46STD/S9 peak tables, KGMN networks and orthogonal validation",
    },
}


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-KGMN-validation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference/kgmn_zenodo_7089991"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, metadata in FILES.items():
        path = args.output_dir / name
        encoded_name = urllib.parse.quote(name, safe="")
        url = f"{BASE_URL}/{encoded_name}/content"
        if path.exists() and digest(path, "md5") != metadata["md5"]:
            raise RuntimeError(f"existing source has wrong MD5: {path}")
        if not path.exists():
            print(f"[download] {name}", flush=True)
            download(url, path)
        observed_md5 = digest(path, "md5")
        if observed_md5 != metadata["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path}: {observed_md5}")
        rows.append(
            {
                "name": name,
                "role": metadata["role"],
                "bytes": path.stat().st_size,
                "md5": observed_md5,
                "sha256": digest(path, "sha256"),
                "url": url,
            }
        )

    report = {
        "status": "kgmn_external_validation_sources_fetched",
        "formal": True,
        "zenodo_record": RECORD_ID,
        "files": rows,
        "raw_accessions_required_for_full_replay": {
            "OEP003284": "46STD/S9 raw LC-MS for propagation validation",
            "OEP003157": "NIST urine, NIST plasma and BV2 raw LC-MS",
            "MTBLS601_MTBLS606": "mouse-liver datasets represented in manual peak truth",
            "MTBLS612_MTBLS615": "fruit-fly datasets represented in manual peak truth",
        },
        "claim_limit": "Immutable public inputs only; no model outcome or performance claim.",
    }
    manifest = args.output_dir / "source_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
