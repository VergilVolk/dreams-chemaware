"""Audit the public MTBLS1905 QC DDA spectra before annotation.

This is deliberately an acquisition/data-quality audit, not an annotation
workflow.  It establishes the exact MS2 universe and its precursor/peak
properties so that any subsequent claim of "additional annotations" has a
fixed denominator and an auditable input set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyteomics import mzml


def precursor_mz(spec: dict) -> float | None:
    try:
        precursor = spec["precursorList"]["precursor"][0]
        selected = precursor["selectedIonList"]["selectedIon"][0]
        return float(selected["selected ion m/z"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def scan_rt_min(spec: dict) -> float | None:
    try:
        return float(spec["scanList"]["scan"][0]["scan start time"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "median", "p95", "max", "mean")}
    x = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(x)), "p05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)), "mean": float(np.mean(x)),
    }


def audit_one(path: Path) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    ms1 = 0
    for spec in mzml.read(str(path)):
        if spec.get("ms level") == 1:
            ms1 += 1
            continue
        if spec.get("ms level") != 2:
            continue
        peaks = len(spec.get("m/z array", []))
        rows.append({
            "source_file": path.name,
            "spectrum_id": str(spec.get("id", "")),
            "precursor_mz": precursor_mz(spec),
            "rt_min": scan_rt_min(spec),
            "n_peaks": peaks,
            "tic": float(spec.get("total ion current", 0.0)),
            "collision_energy": (
                spec.get("precursorList", {}).get("precursor", [{}])[0]
                .get("activation", {}).get("collision energy")
            ),
        })
    valid_precursors = [r["precursor_mz"] for r in rows if r["precursor_mz"] is not None]
    valid_rt = [r["rt_min"] for r in rows if r["rt_min"] is not None]
    return {
        "file": path.name, "n_ms1": ms1, "n_ms2": len(rows),
        "n_missing_precursor": len(rows) - len(valid_precursors),
        "precursor_mz": quantiles(valid_precursors),
        "rt_min": quantiles(valid_rt),
        "peak_count": quantiles([r["n_peaks"] for r in rows]),
        "tic": quantiles([r["tic"] for r in rows]),
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit"))
    args = parser.parse_args()
    sources = sorted(args.input_dir.glob("QC*_MSMS_*.mzML"))
    sources = [p for p in sources if not p.name.endswith("270_1050.mzML")]
    if not sources:
        raise FileNotFoundError("No validated QC mzML files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    files: list[dict] = []
    for source in sources:
        summary, rows = audit_one(source)
        files.append(summary)
        all_rows.extend(rows)
        print(f"{source.name}: MS2={summary['n_ms2']}, missing_precursor={summary['n_missing_precursor']}")
    with (args.output_dir / "qc_ms2_scan_inventory.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "study": "MTBLS1905",
        "purpose": "pre-annotation QC DDA input audit; not annotation evidence",
        "source_files": [p.name for p in sources],
        "n_ms2_total": len(all_rows),
        "file_summaries": files,
        "combined": {
            "precursor_mz": quantiles([r["precursor_mz"] for r in all_rows if r["precursor_mz"] is not None]),
            "rt_min": quantiles([r["rt_min"] for r in all_rows if r["rt_min"] is not None]),
            "peak_count": quantiles([r["n_peaks"] for r in all_rows]),
        },
    }
    (args.output_dir / "qc_ms2_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_ms2_total": report["n_ms2_total"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
