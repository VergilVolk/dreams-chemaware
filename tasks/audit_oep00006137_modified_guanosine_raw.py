#!/usr/bin/env python
"""Raw-data re-extraction of the frozen modified-guanosine panel in OEP00006137.

This script does not use phenotype labels to select peaks or retention-time
windows.  The four unique peaks are frozen from the authors' Level-1
supplement, and the identity family was selected previously from MTBLS13729.
Phenotypes are used only after extraction for paired effect estimation.

The deposited biological and pooled-QC archives are MS1-only mzXML files even
though the article reports ddMS2 acquisition on QC samples.  We therefore test
abundance reproducibility and ion-family observability here; we do not claim an
independent raw-MS2 identity confirmation from OEP00006137.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
from pyteomics import mzxml
from scipy import stats


PANEL_DIR = {("RPLC", "POS"): "rp_pos", ("RPLC", "NEG"): "rp_neg"}
SAMPLE_PATTERN = re.compile(r"^ZZ_(MSI-H|MSS)_([NT])(\d+)$")


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
            "data/external/OEP00006137_raw/modified_guanosine_raw_reextraction_v1"
        ),
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-half-window-sec", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_targets(supplement: Path) -> list[dict]:
    payload = json.loads(supplement.read_text(encoding="utf-8"))
    selected = [
        row for row in payload["rows"]
        if "guanosine" in str(row["Compound name"]).lower()
        and str(row["Compound name"]).lower() != "guanosine"
        and row["LC type"] == "RPLC"
    ]
    groups: dict[tuple, dict] = {}
    for row in selected:
        key = (
            row["LC type"], row["Polarity"], row["Adduct"],
            float(row["mz_detected"]), float(row["RT_detected"]),
        )
        record = groups.setdefault(
            key,
            {
                "target_id": re.sub(r"_[ab]$", "", row["Peak name"]),
                "lc_type": row["LC type"],
                "polarity": row["Polarity"],
                "adduct": row["Adduct"],
                "mz": float(row["mz_detected"]),
                "rt_sec": float(row["RT_detected"]),
                "assignments": [],
                "supplement_rows": [],
            },
        )
        record["assignments"].append(row["Compound name"])
        record["supplement_rows"].append(row)
    targets = sorted(groups.values(), key=lambda item: (item["polarity"], item["rt_sec"]))
    if len(targets) != 4:
        raise RuntimeError(f"expected four unique RPLC modified-guanosine peaks, got {len(targets)}")
    if sorted(len(target["assignments"]) for target in targets) != [1, 1, 2, 2]:
        raise RuntimeError("unexpected isomer-assignment multiplicity in frozen targets")
    return targets


def archive_mzxml(path: Path):
    archive = tarfile.open(path, "r:gz")
    members = [
        member for member in archive.getmembers()
        if member.isfile() and member.name.lower().endswith(".mzxml")
    ]
    if len(members) != 1:
        archive.close()
        raise RuntimeError(f"expected one mzXML in {path}, observed {len(members)}")
    handle = archive.extractfile(members[0])
    if handle is None:
        archive.close()
        raise RuntimeError(f"cannot open mzXML member in {path}")
    return archive, handle


def extract_file(task: tuple[str, Path, list[dict], float, float]) -> list[dict]:
    panel, path, targets, ppm, rt_half_window = task
    archive, handle = archive_mzxml(path)
    traces = {target["target_id"]: [] for target in targets}
    ms_levels: dict[int, int] = {}
    try:
        reader = mzxml.MzXML(handle, use_index=False)
        for scan in reader:
            level = int(scan.get("msLevel", 1))
            ms_levels[level] = ms_levels.get(level, 0) + 1
            if level != 1:
                continue
            rt_sec = float(scan["retentionTime"]) * 60.0
            mz = np.asarray(scan["m/z array"], dtype=np.float64)
            intensity = np.asarray(scan["intensity array"], dtype=np.float64)
            for target in targets:
                if abs(rt_sec - target["rt_sec"]) > rt_half_window:
                    continue
                tolerance = target["mz"] * ppm * 1e-6
                matched = np.abs(mz - target["mz"]) <= tolerance
                if np.any(matched):
                    chosen = np.flatnonzero(matched)[int(np.argmax(intensity[matched]))]
                    traces[target["target_id"]].append(
                        (rt_sec, float(intensity[chosen]), float(mz[chosen]))
                    )
                else:
                    traces[target["target_id"]].append((rt_sec, 0.0, float("nan")))
    finally:
        handle.close()
        archive.close()

    stem = path.name.removesuffix(".mzXML.tar.gz")
    results = []
    for target in targets:
        values = traces[target["target_id"]]
        if values:
            rt = np.asarray([value[0] for value in values], dtype=float)
            abundance = np.asarray([value[1] for value in values], dtype=float)
            observed_mz = np.asarray([value[2] for value in values], dtype=float)
            order = np.argsort(rt)
            rt, abundance, observed_mz = rt[order], abundance[order], observed_mz[order]
            area = float(np.trapz(abundance, rt)) if len(rt) > 1 else float(abundance.sum())
            apex_index = int(np.argmax(abundance))
            apex_rt = float(rt[apex_index])
            apex_mz = float(observed_mz[apex_index])
            peak_height = float(abundance[apex_index])
        else:
            area = peak_height = 0.0
            apex_rt = apex_mz = float("nan")
        results.append(
            {
                "panel": panel,
                "archive": path.name,
                "sample": stem,
                "target_id": target["target_id"],
                "assignments": ";".join(target["assignments"]),
                "adduct": target["adduct"],
                "target_mz": target["mz"],
                "target_rt_sec": target["rt_sec"],
                "area": area,
                "peak_height": peak_height,
                "apex_rt_sec": apex_rt,
                "apex_mz": apex_mz,
                "apex_ppm": (
                    (apex_mz - target["mz"]) / target["mz"] * 1e6
                    if np.isfinite(apex_mz) else float("nan")
                ),
                "ms1_scans": ms_levels.get(1, 0),
                "ms2_scans": ms_levels.get(2, 0),
            }
        )
    return results


def published_column(sample: str) -> str | None:
    match = SAMPLE_PATTERN.match(sample)
    if not match:
        return None
    subtype, tissue, number = match.groups()
    if subtype == "MSI-H":
        return f"MSI_N{number}" if tissue == "N" else f"MSI.H_T{number}"
    return f"MSS_{tissue}{number}"


def paired_summary(frame: pd.DataFrame, subtype: str) -> dict:
    subset = frame.loc[frame["subtype"] == subtype].copy()
    pivot = subset.pivot(index="patient", columns="tissue", values="area").dropna()
    pivot = pivot.loc[(pivot["N"] > 0) & (pivot["T"] > 0)]
    delta = np.log2(pivot["T"].to_numpy() / pivot["N"].to_numpy())
    if len(delta) == 0:
        return {"n": 0}
    ttest = stats.ttest_1samp(delta, 0.0)
    try:
        wilcoxon = stats.wilcoxon(delta)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    positive = int(np.sum(delta > 0))
    nonzero = int(np.sum(delta != 0))
    sign_p = float(stats.binomtest(positive, nonzero, 0.5).pvalue) if nonzero else 1.0
    return {
        "n": int(len(delta)),
        "mean_log2fc": float(np.mean(delta)),
        "median_log2fc": float(np.median(delta)),
        "positive_pairs": positive,
        "negative_pairs": int(np.sum(delta < 0)),
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    targets = freeze_targets(args.supplement)
    target_by_panel: dict[str, list[dict]] = {}
    for target in targets:
        panel = PANEL_DIR[(target["lc_type"], target["polarity"])]
        target_by_panel.setdefault(panel, []).append(target)

    manifest_path = args.raw_dir / "OEP00006137_tissue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unavailable = {
        item["datNo"]
        for item in json.loads(
            (args.raw_dir / "download_rp_only_report.json").read_text(encoding="utf-8")
        ).get("unavailable_objects", [])
    }
    tasks = []
    missing = []
    for row in manifest["rows"]:
        panel = next(
            name for name, exp in manifest["experiments"].items()
            if exp == row["expNo"]
        )
        if panel not in target_by_panel:
            continue
        path = args.raw_dir / panel / row["name"]
        if not path.exists():
            missing.append({"datNo": row["datNo"], "name": row["name"], "known_unavailable": row["datNo"] in unavailable})
            continue
        tasks.append((panel, path, target_by_panel[panel], args.ppm, args.rt_half_window_sec))
    if len(tasks) != 180:
        raise RuntimeError(f"expected 180 usable RP archives, observed {len(tasks)}; missing={missing}")
    if len(missing) != 2 or not all(item["known_unavailable"] for item in missing):
        raise RuntimeError(f"unexpected missing raw archives: {missing}")

    records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(extract_file, tasks), start=1):
            records.extend(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[EIC] {index}/{len(tasks)} archives", flush=True)
    frame = pd.DataFrame(records)
    parsed = frame["sample"].str.extract(SAMPLE_PATTERN)
    frame["subtype"] = parsed[0]
    frame["tissue"] = parsed[1]
    frame["patient"] = pd.to_numeric(parsed[2], errors="coerce")
    frame["published_column"] = frame["sample"].map(published_column)

    supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
    unique_rows: dict[str, dict] = {}
    for target in targets:
        unique_rows[target["target_id"]] = target["supplement_rows"][0]
    correlations = {}
    summaries = {}
    for target_id, group in frame.loc[frame["published_column"].notna()].groupby("target_id"):
        published = unique_rows[target_id]
        observed = []
        expected = []
        for row in group.itertuples(index=False):
            value = published.get(row.published_column)
            if value is not None and float(value) > 0 and row.area > 0:
                observed.append(float(row.area))
                expected.append(float(value))
        rho, rho_p = stats.spearmanr(np.log1p(observed), np.log1p(expected))
        pearson, pearson_p = stats.pearsonr(np.log1p(observed), np.log1p(expected))
        correlations[target_id] = {
            "n": len(observed),
            "spearman_rho_log_area": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r_log_area": float(pearson),
            "pearson_p": float(pearson_p),
        }
        summaries[target_id] = {
            "assignments": group["assignments"].iloc[0].split(";"),
            "panel": group["panel"].iloc[0],
            "target_mz": float(group["target_mz"].iloc[0]),
            "target_rt_sec": float(group["target_rt_sec"].iloc[0]),
            "detected_biological_samples": int((group["area"] > 0).sum()),
            "median_apex_rt_sec": float(group.loc[group["area"] > 0, "apex_rt_sec"].median()),
            "median_apex_ppm": float(group.loc[group["area"] > 0, "apex_ppm"].median()),
            "MSI-H": paired_summary(group, "MSI-H"),
            "MSS": paired_summary(group, "MSS"),
        }

    qc = frame.loc[frame["sample"].str.lower().str.startswith("qc")]
    ms2_scans = int(qc.groupby(["panel", "archive"])["ms2_scans"].max().sum())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eic_path = args.output_dir / "target_eic.csv.gz"
    frame.to_csv(eic_path, index=False, compression="gzip")
    report = {
        "status": "OEP00006137_modified_guanosine_raw_reextraction_complete",
        "formal": True,
        "targets": summaries,
        "published_vs_reextracted": correlations,
        "raw_archives": {
            "usable": len(tasks),
            "missing_public_objects": missing,
            "biological_expected_per_panel": 80,
            "biological_usable_per_panel": {
                panel: int(sum(1 for task in tasks if task[0] == panel and not task[1].name.lower().startswith("qc")))
                for panel in sorted(target_by_panel)
            },
            "pooled_qc_archives": int(qc["archive"].nunique()),
            "deposited_qc_ms2_scans": ms2_scans,
        },
        "parameters": {
            "ppm": args.ppm,
            "rt_half_window_sec": args.rt_half_window_sec,
            "integration": "trapezoidal centroided MS1 EIC area",
        },
        "provenance": {
            "supplement_sha256": sha256sum(args.supplement),
            "manifest_sha256": sha256sum(manifest_path),
            "download_report_sha256": sha256sum(args.raw_dir / "download_rp_only_report.json"),
            "eic_sha256": sha256sum(eic_path),
        },
        "claim_limit": (
            "This independently re-extracts the frozen Level-1 peak coordinates and paired "
            "abundance directions from deposited MS1 data. The deposit contains no QC ddMS2 "
            "scans, so it cannot independently re-confirm positional isomers or the reported "
            "experimental-library MS2 matches."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
