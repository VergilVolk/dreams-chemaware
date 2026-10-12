#!/usr/bin/env python
"""Download and hash the four public ST003325 quantitative result tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

NAMES = [f"ST003325_AN{analysis}_Results.txt" for analysis in ("005444", "005445", "005446", "005447")]
BASE = "https://www.metabolomicsworkbench.org/studydownload/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        destination = args.output / name
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".partial")
            with requests.get(BASE + name, stream=True, timeout=180) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        if block:
                            handle.write(block)
            partial.replace(destination)
        with destination.open("rt", encoding="utf-8-sig", errors="strict") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            rows_count = sum(1 for _ in handle)
        if not header or header[0] != "id_number" or rows_count < 1:
            raise RuntimeError(f"malformed result table: {destination}")
        rows.append({
            "name": name, "bytes": destination.stat().st_size, "sha256": sha256(destination),
            "features": rows_count, "injection_columns": len(header) - 1,
        })
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "status": "ST003325_RESULT_TABLES_ACQUIRED",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "sealed": False,
        "tables": rows,
        "claim_limit": "Quantitative table acquisition only; no QC, MS2 linkage, perturbation effect, or biology.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
