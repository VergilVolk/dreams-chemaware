#!/usr/bin/env python
"""Raw MS1 audit of the frozen HILIC methyl-donor/purine targets in OEP00006137."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

from audit_oep00006137_modified_guanosine_raw import (
    SAMPLE_PATTERN,
    extract_file,
    paired_summary,
    published_column,
    sha256sum,
)


PANEL_DIR = {("HILIC", "POS"): "hilic_pos", ("HILIC", "NEG"): "hilic_neg"}
TARGET_NAMES = {
    "Methionine",
    "Guanosine",
    "S-Adenosylmethionine",
    "S-Adenosylhomocysteine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/external/OEP00006137_raw")
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path(
            "data/external/OEP00006137_support/modified_guanosine_level1_rows.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/hilic_methyl_purine_raw_reextraction_v1"
        ),
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-half-window-sec", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def freeze_targets(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["rows"]
        if row["Compound name"] in TARGET_NAMES and row["LC type"] == "HILIC"
    ]
    if {row["Compound name"] for row in rows} != TARGET_NAMES or len(rows) != 4:
        raise RuntimeError("frozen HILIC target set is incomplete or duplicated")
    return [
        {
            "target_id": row["Peak name"],
            "lc_type": row["LC type"],
            "polarity": row["Polarity"],
            "adduct": row["Adduct"],
            "mz": float(row["mz_detected"]),
            "rt_sec": float(row["RT_detected"]),
            "assignments": [row["Compound name"]],
            "supplement_rows": [row],
        }
        for row in sorted(rows, key=lambda item: (item["Polarity"], item["RT_detected"]))
    ]


def main() -> None:
    args = parse_args()
    targets = freeze_targets(args.supplement)
    target_by_panel: dict[str, list[dict]] = {}
    for target in targets:
        panel = PANEL_DIR[(target["lc_type"], target["polarity"])]
        target_by_panel.setdefault(panel, []).append(target)

    manifest_path = args.raw_dir / "OEP00006137_tissue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    download_path = args.raw_dir / "download_all_tissue_report.json"
    download = json.loads(download_path.read_text(encoding="utf-8"))
    unavailable = {item["datNo"] for item in download["unavailable_objects"]}
    tasks = []
    missing = []
    for row in manifest["rows"]:
        panel = next(
            name for name, experiment in manifest["experiments"].items()
            if experiment == row["expNo"]
        )
        if panel not in target_by_panel:
            continue
        path = args.raw_dir / panel / row["name"]
        if not path.exists():
            missing.append(
                {
                    "datNo": row["datNo"],
                    "name": row["name"],
                    "known_unavailable": row["datNo"] in unavailable,
                }
            )
            continue
        tasks.append(
            (panel, path, target_by_panel[panel], args.ppm, args.rt_half_window_sec)
        )
    if len(tasks) != 180:
        raise RuntimeError(f"expected 180 usable HILIC archives, got {len(tasks)}")
    if len(missing) != 2 or not all(item["known_unavailable"] for item in missing):
        raise RuntimeError(f"unexpected unavailable HILIC objects: {missing}")

    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(extract_file, tasks), start=1):
            records.extend(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[HILIC EIC] {index}/{len(tasks)} archives", flush=True)
    frame = pd.DataFrame(records)
    parsed = frame["sample"].str.extract(SAMPLE_PATTERN)
    frame["subtype"] = parsed[0]
    frame["tissue"] = parsed[1]
    frame["patient"] = pd.to_numeric(parsed[2], errors="coerce")
    frame["published_column"] = frame["sample"].map(published_column)

    supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
    published_by_target = {
        target["target_id"]: target["supplement_rows"][0] for target in targets
    }
    correlations = {}
    summaries = {}
    biological = frame.loc[frame["published_column"].notna()]
    for target_id, group in biological.groupby("target_id"):
        published = published_by_target[target_id]
        observed, expected = [], []
        for row in group.itertuples(index=False):
            value = published.get(row.published_column)
            if value is not None and float(value) > 0 and row.area > 0:
                observed.append(float(row.area))
                expected.append(float(value))
        if len(observed) >= 3:
            rho, rho_p = stats.spearmanr(np.log1p(observed), np.log1p(expected))
            pearson, pearson_p = stats.pearsonr(np.log1p(observed), np.log1p(expected))
        else:
            rho = rho_p = pearson = pearson_p = float("nan")
        correlations[target_id] = {
            "n": len(observed),
            "spearman_rho_log_area": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r_log_area": float(pearson),
            "pearson_p": float(pearson_p),
        }
        summaries[target_id] = {
            "assignment": group["assignments"].iloc[0],
            "panel": group["panel"].iloc[0],
            "target_mz": float(group["target_mz"].iloc[0]),
            "target_rt_sec": float(group["target_rt_sec"].iloc[0]),
            "detected_biological_samples": int((group["area"] > 0).sum()),
            "median_apex_rt_sec": float(
                group.loc[group["area"] > 0, "apex_rt_sec"].median()
            ),
            "median_apex_ppm": float(
                group.loc[group["area"] > 0, "apex_ppm"].median()
            ),
            "MSI-H": paired_summary(group, "MSI-H"),
            "MSS": paired_summary(group, "MSS"),
        }

    qc = frame.loc[frame["sample"].str.lower().str.startswith("qc")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eic_path = args.output_dir / "target_eic.csv.gz"
    frame.to_csv(eic_path, index=False, compression="gzip")
    report = {
        "status": "OEP00006137_hilic_methyl_purine_raw_reextraction_complete",
        "formal": True,
        "targets": summaries,
        "published_vs_reextracted": correlations,
        "raw_archives": {
            "usable": len(tasks),
            "missing_public_objects": missing,
            "biological_usable_per_panel": {
                panel: int(
                    sum(
                        1 for task in tasks
                        if task[0] == panel
                        and not task[1].name.lower().startswith("qc")
                    )
                )
                for panel in sorted(target_by_panel)
            },
            "pooled_qc_archives": int(qc["archive"].nunique()),
            "deposited_qc_ms2_scans": int(
                qc.groupby(["panel", "archive"])["ms2_scans"].max().sum()
            ),
        },
        "parameters": {
            "ppm": args.ppm,
            "rt_half_window_sec": args.rt_half_window_sec,
            "integration": "trapezoidal centroided MS1 EIC area",
            "target_selection": "four names frozen before raw inspection",
        },
        "provenance": {
            "supplement_sha256": sha256sum(args.supplement),
            "manifest_sha256": sha256sum(manifest_path),
            "download_report_sha256": sha256sum(download_path),
            "eic_sha256": sha256sum(eic_path),
        },
        "claim_limit": (
            "This is a raw-MS1 abundance reproducibility test of frozen HILIC peaks. "
            "It does not infer methylation flux, enzyme activity, or causal coupling "
            "between SAM/SAH and modified guanosines."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
