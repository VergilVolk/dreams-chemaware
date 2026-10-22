#!/usr/bin/env python3
"""Census public GNPS FBMN jobs as a response-transport substrate.

This is deliberately outcome blind.  It asks only whether a literature-linked
GNPS task is a completed FBMN job with a readable quantitative feature table
and representative MS/MS spectra.  It does not download full artifacts or read
phenotype labels, feature effects, annotations, or cross-study truth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import requests


DETAILS = "https://gnps.ucsd.edu/ProteoSAFe/QueryTaskDetails"
RESULT = "https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile"
ACCESSION = re.compile(r"MSV\d{9}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(task_id: str, result_path: str, expected: bytes) -> tuple[bool, int, str]:
    try:
        with requests.get(
            RESULT,
            params={"task": task_id, "file": result_path, "block": "main"},
            headers={"Range": "bytes=0-8191"},
            stream=True,
            timeout=30,
        ) as response:
            status = int(response.status_code)
            prefix = next(response.iter_content(8192), b"")
        text = prefix[:200].decode("utf-8", errors="replace")
        valid = status == 200 and expected in prefix and b"<html" not in prefix.lower()
        return valid, status, text
    except requests.RequestException as error:
        return False, 0, type(error).__name__


def inspect_task(task_id: str) -> dict[str, object]:
    try:
        details_response = requests.get(DETAILS, params={"task": task_id}, timeout=30)
        details_response.raise_for_status()
        details = details_response.json()
    except (requests.RequestException, ValueError) as error:
        return {
            "task_id": task_id, "workflow": "", "task_status": "UNREADABLE",
            "quantification_readable": False, "spectra_readable": False,
            "metadata_readable": False, "probe_error": type(error).__name__,
        }
    workflow = str(details.get("workflow", ""))
    status = str(details.get("status", ""))
    row: dict[str, object] = {
        "task_id": task_id, "workflow": workflow, "task_status": status,
        "description": str(details.get("description", "")),
        "quantification_readable": False, "spectra_readable": False,
        "metadata_readable": False, "probe_error": "",
    }
    if workflow != "FEATURE-BASED-MOLECULAR-NETWORKING" or status != "DONE":
        return row
    quant, quant_status, quant_prefix = probe(task_id, "quantification_table_reformatted/", b",")
    spectra, spectra_status, _ = probe(task_id, "spectra_reformatted/", b"BEGIN IONS")
    metadata, metadata_status, _ = probe(task_id, "metadata_merged/", b"filename")
    row.update({
        "quantification_readable": quant,
        "spectra_readable": spectra,
        "metadata_readable": metadata,
        "quantification_http_status": quant_status,
        "spectra_http_status": spectra_status,
        "metadata_http_status": metadata_status,
        "quantification_header_prefix": quant_prefix.replace("\r", " ").replace("\n", " "),
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    mapping = pd.read_csv(args.task_mapping, dtype=str, keep_default_na=False)
    required = {"task_id", "massive_ids", "pmid"}
    if missing := required - set(mapping):
        raise ValueError(f"task mapping lacks {sorted(missing)}")
    task_map = mapping.groupby("task_id", sort=True).agg({
        "massive_ids": lambda x: " | ".join(sorted(set(x))),
        "pmid": lambda x: " | ".join(sorted(set(x))),
    }).reset_index()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(executor.map(inspect_task, task_map.task_id))
    census = task_map.merge(pd.DataFrame(inspected), on="task_id", validate="one_to_one")
    census["accessions"] = census.massive_ids.map(
        lambda value: " | ".join(sorted(set(ACCESSION.findall(value))))
    )
    census["accession_count"] = census.accessions.map(lambda value: len(value.split(" | ")) if value else 0)
    census["response_substrate_ready"] = (
        census.workflow.eq("FEATURE-BASED-MOLECULAR-NETWORKING")
        & census.task_status.eq("DONE")
        & census.quantification_readable.astype(bool)
        & census.spectra_readable.astype(bool)
    )
    args.output_dir.mkdir(parents=True)
    census.to_csv(args.output_dir / "task_census.tsv", sep="\t", index=False)
    ready = census.loc[census.response_substrate_ready]
    accessions = sorted({item for value in ready.accessions for item in value.split(" | ") if item})
    report = {
        "schema": "public_fbmn_response_substrate_census_v1",
        "status": "outcome_blind_artifact_census_no_biological_result",
        "denominators": {
            "mapped_tasks": int(len(census)),
            "completed_fbmn_tasks": int((census.workflow.eq("FEATURE-BASED-MOLECULAR-NETWORKING") & census.task_status.eq("DONE")).sum()),
            "response_substrate_ready_tasks": int(len(ready)),
            "ready_tasks_with_embedded_metadata": int(ready.metadata_readable.astype(bool).sum()),
            "ready_single_accession_tasks": int(ready.accession_count.eq(1).sum()),
            "ready_multi_accession_tasks": int(ready.accession_count.gt(1).sum()),
            "unique_mapped_accessions_in_ready_tasks": len(accessions),
        },
        "qualification": "completed GNPS1 FBMN plus readable standardized quantification table and representative MGF",
        "exclusions": "non-FBMN, deleted/failed/unreadable tasks, or missing quantitative/MS2 artifact",
        "leakage_status": "no phenotype labels, feature effects, annotations, or cross-study truth read",
        "provenance": {
            "task_mapping": str(args.task_mapping),
            "task_mapping_sha256": sha256(args.task_mapping),
        },
        "claim_boundary": [
            "Task readiness is not study readiness; sample columns must be joined to repository metadata.",
            "Multi-accession tasks must be split and source-family duplicates adjudicated.",
            "A public FBMN result may reflect author-selected preprocessing and is not equivalent to uniform raw reprocessing.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
