"""Quantify the complete phenotype-blind LCNEC HSST3n QC-qualified universe.

The target universe is fixed exclusively by the pooled-QC, blank and dilution
ledger produced by ``audit_lcnec_hsst3n_qc_headroom.py``.  This script extracts
MS1 EIC signals for all 263 families from all 85 injections.  It does not test
tumour/adjacent effects and must not be replaced by the historical 221-target
author-unmatched matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from quantify_lcnec_hsst3n_dark_eic import classify, quantify_file, sha256


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip", type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip"),
    )
    parser.add_argument(
        "--overview", type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"),
    )
    parser.add_argument(
        "--targets", type=Path,
        default=Path("data/validation/lcnec_hsst3n_qc_headroom_gate/precursor_family_ledger.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/lcnec_hsst3n_all_qc_eic_v1"),
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (args.zip, args.overview, args.targets)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LCNEC_ALL_QC_EIC_PREFLIGHT_MISSING: {missing}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with args.targets.open("r", encoding="utf-8", newline="") as handle:
        target_rows = [row for row in csv.DictReader(handle) if truthy(row.get("passes_all"))]
    if len(target_rows) != 263:
        raise RuntimeError(f"expected frozen 263 QC-qualified families, found {len(target_rows)}")
    targets = [
        {
            **row,
            "family_id": int(row["family_id"]),
            "mz_median": float(row["mz_median"]),
            "rt_median_sec": float(row["rt_median_sec"]),
        }
        for row in target_rows
    ]
    targets.sort(key=lambda row: (float(row["rt_median_sec"]), int(row["family_id"])))
    target_rts = [float(row["rt_median_sec"]) for row in targets]

    with args.overview.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    if len(ledger) != 85:
        raise RuntimeError(f"expected 85 injections, found {len(ledger)}")
    file_class = [classify(row["NOTE"]) for row in ledger]
    if {name: file_class.count(name) for name in set(file_class)} != {
        "study": 68, "pooled_qc": 9, "blank": 2, "qc_dilution": 6,
    }:
        raise RuntimeError("unexpected LCNEC injection-class counts")

    pair_rows: dict[str, set[str]] = {}
    for row, kind in zip(ledger, file_class, strict=True):
        if kind == "study":
            pair_rows.setdefault(row["SAMPLE_CODE"], set()).add(row["GROUP_CODE"])
    if len(pair_rows) != 34 or any(groups != {"TU", "NG"} for groups in pair_rows.values()):
        raise RuntimeError("study ledger is not exactly 34 complete TU/NG pairs")

    area = np.zeros((len(ledger), len(targets)), dtype=np.float64)
    maximum = np.zeros_like(area)
    scan_count = np.zeros_like(area, dtype=np.int32)
    import zipfile

    with zipfile.ZipFile(args.zip) as archive:
        members = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if info.filename.lower().endswith(".mzml")
        }
        absent = [row["mzML_FILE_NAME"] for row in ledger if row["mzML_FILE_NAME"] not in members]
        if absent:
            raise RuntimeError(f"mzML members missing from archive: {absent[:5]}")
        for file_index, row in enumerate(ledger):
            with archive.open(members[row["mzML_FILE_NAME"]]) as handle:
                values = quantify_file(handle, targets, target_rts, args.ppm, args.rt_sec)
            area[file_index], maximum[file_index], scan_count[file_index] = values
            print(
                f"[all-263 EIC] {file_index + 1}/85 {row['SAMPLE_ID']} "
                f"detected={int((area[file_index] > 0).sum())}",
                flush=True,
            )

    study_index = np.asarray([i for i, kind in enumerate(file_class) if kind == "study"])
    study_detection = np.mean(area[study_index] > 0, axis=0)
    if not np.isfinite(area).all() or np.any(area < 0):
        raise RuntimeError("non-finite or negative EIC values")

    matrix_path = args.output_dir / "all_qc_family_eic_matrix.npz"
    np.savez_compressed(
        matrix_path,
        area=area,
        maximum=maximum,
        scan_count=scan_count,
        family_id=np.asarray([row["family_id"] for row in targets], dtype=np.int64),
        mz=np.asarray([row["mz_median"] for row in targets], dtype=np.float64),
        rt_sec=np.asarray([row["rt_median_sec"] for row in targets], dtype=np.float64),
        sample_id=np.asarray([row["SAMPLE_ID"] for row in ledger]),
        sample_code=np.asarray([row["SAMPLE_CODE"] for row in ledger]),
        group_code=np.asarray([row["GROUP_CODE"] for row in ledger]),
        file_class=np.asarray(file_class),
    )
    with (args.output_dir / "family_detection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("family_id", "mz", "rt_sec", "study_detection_fraction")
        )
        writer.writeheader()
        for row, detection in zip(targets, study_detection, strict=True):
            writer.writerow({
                "family_id": row["family_id"], "mz": row["mz_median"],
                "rt_sec": row["rt_median_sec"], "study_detection_fraction": float(detection),
            })

    report = {
        "status": "lcnec_hsst3n_all_qc_eic_complete",
        "formal": True,
        "phenotype_used_for_target_selection": False,
        "families": len(targets),
        "injections": len(ledger),
        "complete_pairs": len(pair_rows),
        "study_detection_median": float(np.median(study_detection)),
        "study_detection_min": float(np.min(study_detection)),
        "parameters": {"ppm": args.ppm, "rt_sec": args.rt_sec},
        "provenance": {
            "zip_sha256": sha256(args.zip),
            "overview_sha256": sha256(args.overview),
            "targets_sha256": sha256(args.targets),
            "matrix_sha256": sha256(matrix_path),
        },
        "claim_limit": (
            "Complete phenotype-blind QC-qualified abundance matrix only; no differential, "
            "family, transformation, cross-platform, dark-increment or biological result."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
