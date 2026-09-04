#!/usr/bin/env python
"""Seal MetDNA3 source metadata and supplements before outcome inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


DATASET = "MSV000097913"
SUPPLEMENT_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
    "PMC12398597/supplementaryFiles"
)
PARAMS_URL = (
    "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
    "file=f.MSV000097913/ccms_parameters/params.xml"
)
EXPECTED_OPEN_FILES = 239
EXPECTED_OPEN_BYTES = 10_700_428_871
BIOLOGICAL_SAMPLES = (
    "BV2cell",
    "Mouse_brain",
    "Mouse_liver",
    "NIST_plasma",
    "NIST_urine",
)
SEPARATIONS = ("hilic", "rplc")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists():
        if path.stat().st_size == 0:
            raise RuntimeError(f"existing download is empty: {path}")
        print(f"[reuse] existing download: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-BioAware/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if partial.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")
    os.replace(partial, path)


def query_inventory() -> list[dict]:
    sql = (
        "select filepath,collection,size from filename "
        f"where dataset='{DATASET}' and collection='ccms_peak' order by filepath"
    )
    url = (
        "https://datasetcache.gnps2.org/datasette/database.json?sql="
        + urllib.parse.quote(sql)
    )
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-BioAware/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        rows = json.load(response)["rows"]
    return [
        {"filepath": str(row[0]), "collection": str(row[1]), "bytes": int(row[2])}
        for row in rows
    ]


def safe_zip_inventory(path: Path) -> list[dict]:
    records: list[dict] = []
    with zipfile.ZipFile(path) as bundle:
        seen: set[str] = set()
        for member in bundle.infolist():
            if "\\" in member.filename:
                raise RuntimeError(f"unsafe supplementary member: {member.filename}")
            pure = PurePosixPath(member.filename)
            if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
                raise RuntimeError(f"unsafe supplementary member: {member.filename}")
            name = pure.as_posix()
            if name in seen:
                raise RuntimeError(f"duplicate supplementary member: {name}")
            seen.add(name)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError(f"supplementary symlink forbidden: {name}")
            if not member.is_dir():
                records.append(
                    {
                        "relative_path": name,
                        "bytes": int(member.file_size),
                        "crc32": f"{member.CRC:08x}",
                        "content_read": False,
                    }
                )
    return records


def classify_panel(filepath: str) -> tuple[str, str, str] | None:
    lower = filepath.lower()
    sample = next((value for value in BIOLOGICAL_SAMPLES if value.lower() in lower), None)
    separation = next((value for value in SEPARATIONS if f"_{value}/" in lower), None)
    polarity = "negative" if "_neg_" in lower else "positive" if "_pos_" in lower else None
    if sample and separation and polarity:
        return sample, separation, polarity
    return None


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/external/metdna3_2025")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_external_contract_v1"),
    )
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    supplement = source / "PMC12398597_supplementaryFiles.zip"
    params = source / "MSV000097913_params.xml"
    download(SUPPLEMENT_URL, supplement)
    download(PARAMS_URL, params)
    inventory = query_inventory()
    if len(inventory) != EXPECTED_OPEN_FILES:
        raise RuntimeError(f"MassIVE open-file count changed: {len(inventory)}")
    if sum(row["bytes"] for row in inventory) != EXPECTED_OPEN_BYTES:
        raise RuntimeError("MassIVE open-file byte total changed")
    panels = Counter(filter(None, (classify_panel(row["filepath"]) for row in inventory)))
    expected_panels = {
        (sample, separation, polarity)
        for sample in BIOLOGICAL_SAMPLES
        for separation in SEPARATIONS
        for polarity in ("negative", "positive")
    }
    if set(panels) != expected_panels:
        raise RuntimeError(
            f"expected 20 targeted-MS2 panels, got missing={sorted(expected_panels-set(panels))} "
            f"extra={sorted(set(panels)-expected_panels)}"
        )
    supplementary = safe_zip_inventory(supplement)
    if not any(row["relative_path"].lower().endswith(".xlsx") for row in supplementary):
        raise RuntimeError("supplementary bundle contains no xlsx outcome tables")
    source.mkdir(parents=True, exist_ok=True)
    inventory_path = source / "MSV000097913_ccms_peak_inventory.json"
    write_atomic(inventory_path, {"dataset": DATASET, "files": inventory})
    split = {
        "development": ["NIST_urine|hilic|negative", "NIST_urine|hilic|positive"],
        "internal_validation": ["NIST_urine|rplc|negative", "NIST_urine|rplc|positive"],
        "untouched_external_test": sorted(
            "|".join(panel) for panel in expected_panels if panel[0] != "NIST_urine"
        ),
    }
    report = {
        "status": "bioaware_metdna3_external_contract_frozen",
        "formal": True,
        "article": "10.1038/s41467-025-63536-6",
        "dataset": DATASET,
        "source_data_opened": False,
        "massive": {
            "open_mzml_files": len(inventory),
            "open_mzml_bytes": sum(row["bytes"] for row in inventory),
            "inventory_sha256": file_hash(inventory_path),
            "params_sha256": file_hash(params),
        },
        "supplement": {
            "url": SUPPLEMENT_URL,
            "bytes": supplement.stat().st_size,
            "sha256": file_hash(supplement),
            "members": supplementary,
        },
        "panel_counts": {"|".join(key): int(value) for key, value in sorted(panels.items())},
        "frozen_split": split,
        "published_protocol_reproduced": {
            "level1_seed_fraction": 0.30,
            "held_out_validation_fraction": 0.70,
            "folds": 10,
            "seed_definition": "MS1 + RT + MS2 matched to chemical standards",
        },
        "fixed_bioaware_increment": {
            "knowledge_layer": "Rhea reaction hypergraph",
            "data_layer": "DreaMS plus classical MS2 feature similarity",
            "candidate_gate": "MS1/formula/adduct plus cross-layer pre-mapping",
            "network_support": "dependency-corrected reaction-context support",
            "inference": "low-margin override or abstention; no forced propagation",
        },
        "forbidden": [
            "open supplementary xlsx values before this contract is written",
            "tune thresholds on internal-validation or external-test outcomes",
            "use held-out Level-1 identities as seeds",
            "use MetDNA3 output labels as BioAware model features",
            "use phenotype or differential abundance in identity ranking",
        ],
        "primary_endpoints": {
            "ranking": ["Recall@1", "MRR", "corrected", "introduced"],
            "coverage": ["seed-only Level-1", "network-supported held-out candidates"],
            "statistics": [
                "metabolite-cluster bootstrap CI",
                "panel-stratified effects",
                "degree-preserving graph decoys",
            ],
        },
        "success_gate": {
            "development": "mechanism and implementation only; no final claim",
            "internal_validation": "corrected > introduced and no panel degrades",
            "external_test": (
                "pooled Recall@1 CI lower bound > 0, MRR nonnegative, corrected > "
                "introduced, and at least 12/16 panels nonnegative"
            ),
        },
        "claim_limit": (
            "This benchmark evaluates a downstream two-layer annotation expert. "
            "It does not establish a new shared DreaMS embedding or biological flux."
        ),
    }
    destination = output / "contract.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError(f"fail-closed: existing MetDNA3 contract differs: {destination}")
        print(f"[reuse] verified MetDNA3 external contract: {destination}", flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"fail-closed: non-empty contract directory: {output}")
        write_atomic(destination, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
