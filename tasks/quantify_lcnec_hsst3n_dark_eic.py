"""Targeted MS1 EIC re-quantification and paired screen of LCNEC dark features.

Targets are frozen by phenotype-blind QC/blank/dilution and author-overlap
gates.  This script streams the same public mzML ZIP, quantifies MS1 signal in
5-ppm/15-second windows, re-applies phenotype-blind quality filters, and only
then tests 34 tumor/adjacent pairs.  It does not annotate chemical identity.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon


MS_LEVEL = "MS:1000511"
SCAN_START_TIME = "MS:1000016"
MZ_ARRAY = "MS:1000514"
INTENSITY_ARRAY = "MS:1000515"
ZLIB_COMPRESSION = "MS:1000574"
FLOAT64 = "MS:1000523"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    output = np.full(len(values), np.nan)
    indices = np.where(finite)[0]
    if not len(indices):
        return output.tolist()
    order = indices[np.argsort(values[indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(ranked, 1.0)
    return output.tolist()


def cv_params(element: ET.Element) -> dict[str, tuple[str | None, str | None]]:
    values: dict[str, tuple[str | None, str | None]] = {}
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] != "cvParam":
            continue
        accession = child.attrib.get("accession")
        value = child.attrib.get("value")
        if accession and value not in (None, ""):
            values[accession] = (value, child.attrib.get("unitName"))
    return values


def decode_array(binary_array: ET.Element) -> tuple[str | None, np.ndarray]:
    accessions = {
        child.attrib.get("accession")
        for child in binary_array.iter()
        if child.tag.rsplit("}", 1)[-1] == "cvParam"
    }
    kind = "mz" if MZ_ARRAY in accessions else "intensity" if INTENSITY_ARRAY in accessions else None
    if kind is None or FLOAT64 not in accessions or ZLIB_COMPRESSION not in accessions:
        return kind, np.empty(0)
    binary = next(
        (child.text for child in binary_array if child.tag.rsplit("}", 1)[-1] == "binary"), None
    )
    if not binary:
        return kind, np.empty(0)
    return kind, np.frombuffer(zlib.decompress(base64.b64decode(binary)), dtype="<f8")


def quantify_file(
    handle: BinaryIO, targets: list[dict[str, object]], target_rts: list[float], ppm: float, rt_window: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros(len(targets), dtype=float)
    maxima = np.zeros(len(targets), dtype=float)
    scans = np.zeros(len(targets), dtype=int)
    for _event, element in ET.iterparse(handle, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "spectrum":
            continue
        params = cv_params(element)
        try:
            level = int(float(params.get(MS_LEVEL, ("nan", None))[0]))
        except ValueError:
            level = -1
        if level != 1:
            element.clear()
            continue
        rt_value, rt_unit = params.get(SCAN_START_TIME, (None, None))
        try:
            rt = float(rt_value) if rt_value is not None else math.nan
        except ValueError:
            rt = math.nan
        if rt_unit and "minute" in rt_unit.lower():
            rt *= 60.0
        if not math.isfinite(rt):
            element.clear()
            continue
        left = bisect.bisect_left(target_rts, rt - rt_window)
        right = bisect.bisect_right(target_rts, rt + rt_window)
        if left == right:
            element.clear()
            continue
        arrays: dict[str, np.ndarray] = {}
        for child in element.iter():
            if child.tag.rsplit("}", 1)[-1] == "binaryDataArray":
                kind, values = decode_array(child)
                if kind:
                    arrays[kind] = values
        mz_values = arrays.get("mz", np.empty(0))
        intensity_values = arrays.get("intensity", np.empty(0))
        if len(mz_values) != len(intensity_values) or not len(mz_values):
            element.clear()
            continue
        for target_index in range(left, right):
            target_mz = float(targets[target_index]["mz_median"])
            tolerance = target_mz * ppm * 1e-6
            lo = np.searchsorted(mz_values, target_mz - tolerance, side="left")
            hi = np.searchsorted(mz_values, target_mz + tolerance, side="right")
            if hi <= lo:
                continue
            signal = float(np.max(intensity_values[lo:hi]))
            sums[target_index] += signal
            maxima[target_index] = max(maxima[target_index], signal)
            scans[target_index] += 1
        element.clear()
    return sums, maxima, scans


def classify(note: str) -> str:
    if note == "Study sample":
        return "study"
    if note == "QC sample":
        return "pooled_qc"
    if note == "Method blank":
        return "blank"
    if note.startswith("Serial dilution"):
        return "qc_dilution"
    return "unknown"


def dilution_fraction(sample_id: str) -> float:
    return {
        "SD-A-0": 0.0,
        "SD-A-1-16": 1 / 16,
        "SD-A-1-8": 1 / 8,
        "SD-A-1-4": 1 / 4,
        "SD-A-1-2": 1 / 2,
        "SD-A-1": 1.0,
    }.get(sample_id, math.nan)


def main() -> None:
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
        default=Path("data/validation/lcnec_hsst3n_author_overlap_gate/qualified_family_author_overlap.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_eic_gate")
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    args = parser.parse_args()

    with args.targets.open("r", encoding="utf-8", newline="") as handle:
        targets = [row for row in csv.DictReader(handle) if row["author_matched"].lower() == "false"]
    targets.sort(key=lambda row: float(row["rt_median_sec"]))
    if len(targets) != 221:
        raise RuntimeError(f"expected frozen 221 author-unmatched targets, found {len(targets)}")
    target_rts = [float(row["rt_median_sec"]) for row in targets]

    with args.overview.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    if len(ledger) != 85:
        raise RuntimeError(f"expected 85 injections, found {len(ledger)}")

    area_matrix = np.zeros((len(ledger), len(targets)), dtype=float)
    max_matrix = np.zeros_like(area_matrix)
    scan_matrix = np.zeros_like(area_matrix, dtype=int)
    with zipfile.ZipFile(args.zip) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if info.filename.lower().endswith(".mzml")}
        for file_index, row in enumerate(ledger):
            name = row["mzML_FILE_NAME"]
            with archive.open(members[name]) as handle:
                areas, maxima, scans = quantify_file(handle, targets, target_rts, args.ppm, args.rt_sec)
            area_matrix[file_index] = areas
            max_matrix[file_index] = maxima
            scan_matrix[file_index] = scans
            print(f"[dark EIC] {file_index + 1}/85 {row['SAMPLE_ID']} detected={(areas > 0).sum()}", flush=True)

    file_class = [classify(row["NOTE"]) for row in ledger]
    qc_idx = [i for i, value in enumerate(file_class) if value == "pooled_qc"]
    blank_idx = [i for i, value in enumerate(file_class) if value == "blank"]
    dilution_idx = [i for i, value in enumerate(file_class) if value == "qc_dilution"]
    study_idx = [i for i, value in enumerate(file_class) if value == "study"]
    dilution_levels = [dilution_fraction(ledger[i]["SAMPLE_ID"]) for i in dilution_idx]

    pair_rows: dict[str, dict[str, int]] = {}
    for index in study_idx:
        code = ledger[index]["SAMPLE_CODE"]
        group = ledger[index]["GROUP_CODE"]
        pair_rows.setdefault(code, {})[group] = index
    if len(pair_rows) != 34 or any(set(value) != {"TU", "NG"} for value in pair_rows.values()):
        raise RuntimeError("study ledger is not exactly 34 complete TU/NG pairs")

    results: list[dict[str, object]] = []
    quality_indices: list[int] = []
    raw_t_p: list[float] = []
    raw_w_p: list[float] = []
    for target_index, target in enumerate(targets):
        qc = area_matrix[qc_idx, target_index]
        blank = area_matrix[blank_idx, target_index]
        dilution = area_matrix[dilution_idx, target_index]
        study = area_matrix[study_idx, target_index]
        qc_mean = float(np.mean(qc))
        qc_cv = float(np.std(qc, ddof=1) / qc_mean) if qc_mean > 0 else math.inf
        blank_ratio = float(np.max(blank) / np.median(qc)) if np.median(qc) > 0 else math.inf
        rho = float(spearmanr(dilution_levels, np.log1p(dilution)).statistic)
        if not math.isfinite(rho):
            rho = 0.0
        detection = float(np.mean(study > 0))
        quality = qc_cv <= 0.30 and blank_ratio <= 0.20 and rho >= 0.70 and detection >= 0.80

        positives = study[study > 0]
        pseudocount = float(np.min(positives) / 2) if len(positives) else 1.0
        tu = np.asarray([area_matrix[value["TU"], target_index] for value in pair_rows.values()])
        ng = np.asarray([area_matrix[value["NG"], target_index] for value in pair_rows.values()])
        delta = np.log2(tu + pseudocount) - np.log2(ng + pseudocount)
        t_p = float(ttest_rel(np.log2(tu + pseudocount), np.log2(ng + pseudocount)).pvalue)
        try:
            w_p = float(wilcoxon(delta).pvalue)
        except ValueError:
            w_p = 1.0
        raw_t_p.append(t_p)
        raw_w_p.append(w_p)
        if quality:
            quality_indices.append(target_index)
        results.append(
            {
                "family_id": target["family_id"],
                "mz": target["mz_median"],
                "rt_sec": target["rt_median_sec"],
                "qc_cv": qc_cv,
                "blank_to_qc_ratio": blank_ratio,
                "dilution_spearman_rho": rho,
                "study_detection_fraction": detection,
                "quality_pass": quality,
                "mean_log2fc_tu_vs_ng": float(np.mean(delta)),
                "direction_concordance": float(max(np.mean(delta > 0), np.mean(delta < 0))),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            }
        )

    q_all = bh_adjust(raw_t_p)
    quality_p = [raw_t_p[index] for index in quality_indices]
    quality_q = bh_adjust(quality_p)
    quality_q_by_index = {index: value for index, value in zip(quality_indices, quality_q, strict=True)}
    for index, result in enumerate(results):
        result["paired_t_q_all_221"] = q_all[index]
        result["paired_t_q_quality_set"] = quality_q_by_index.get(index, math.nan)
        result["robust_discovery"] = bool(
            result["quality_pass"]
            and float(result["paired_t_q_quality_set"]) <= 0.10
            and float(result["wilcoxon_p"]) <= 0.05
            and abs(float(result["mean_log2fc_tu_vs_ng"])) >= 0.50
            and float(result["direction_concordance"]) >= 0.65
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dark_feature_paired_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    np.savez_compressed(
        args.output_dir / "dark_feature_eic_matrix.npz",
        area=area_matrix,
        maximum=max_matrix,
        scan_count=scan_matrix,
        family_id=np.asarray([int(target["family_id"]) for target in targets]),
        sample_id=np.asarray([row["SAMPLE_ID"] for row in ledger]),
    )

    robust = [row for row in results if row["robust_discovery"]]
    report = {
        "status": "lcnec_hsst3n_dark_eic_gate_complete",
        "formal": True,
        "targets": len(targets),
        "complete_pairs": len(pair_rows),
        "phenotype_blind_quality_pass": len(quality_indices),
        "robust_dark_discoveries": len(robust),
        "top_robust": sorted(robust, key=lambda row: float(row["paired_t_q_quality_set"]))[:20],
        "gates": {
            "quality_pass_ge_100": len(quality_indices) >= 100,
            "robust_dark_discoveries_ge_5": len(robust) >= 5,
        },
        "pass_to_annotation_and_mechanism_module": len(robust) >= 5,
        "parameters": {
            "ppm": args.ppm, "rt_sec": args.rt_sec, "qc_cv_max": 0.30,
            "blank_ratio_max": 0.20, "dilution_rho_min": 0.70, "study_detection_min": 0.80,
            "quality_set_fdr_max": 0.10, "absolute_log2fc_min": 0.50, "direction_concordance_min": 0.65,
        },
        "provenance": {
            "zip_sha256": sha256(args.zip), "overview_sha256": sha256(args.overview),
            "targets_sha256": sha256(args.targets),
        },
        "claim_limit": "MS1 abundance features absent from the author HSST3n ledger; chemical identities and mechanisms remain unassigned.",
    }
    (args.output_dir / "dark_eic_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
