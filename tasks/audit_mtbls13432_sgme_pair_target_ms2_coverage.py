#!/usr/bin/env python
"""Low-cost, fail-closed MS2 coverage gate for the SgME-HCC pivot.

The source study's processed table is not a direct map from every selected
feature to an observable precursor in the public tissue mzML files.  This
audit therefore separates true annotation gaps from already assigned records,
and separates directly observable precursor m/z values from neutral-mass-only
features before streaming a small, paired patient panel.

No binary arrays are decoded and no phenotype is used for annotation.  This is
an acquisition-coverage audit, not an identification result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyreadr
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
NS = "{http://psi.hupo.org/ms/mzml}"
UNANNOTATED = {"", "nan", "none", "unannotated", "no matching", "no match"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peaks",
        type=Path,
        default=ROOT / "data/external/mtbls13432_sgme_hcc/source_snapshot/data/lcms/lcms_peaks.Rdata",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=ROOT / "data/validation/mtbls13432_sgme_preflight/positive_mode_paired_pilot_files.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/validation/mtbls13432_sgme_coverage_probe",
    )
    parser.add_argument("--patients", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--rt-tolerance-sec", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_annotation(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text


def is_unannotated(value: object) -> bool:
    return clean_annotation(value).lower() in UNANNOTATED


def mz_id_coordinates(value: object) -> tuple[float, float, str]:
    text = str(value)
    match = re.match(r"^([0-9.]+)_([0-9.]+)(m/z|n)_([A-Z]+)$", text)
    if not match:
        raise RuntimeError(f"unsupported mz_id: {text}")
    return float(match.group(1)), float(match.group(2)), match.group(3)


def build_target_ledger(peaks: Path) -> pd.DataFrame:
    objects = pyreadr.read_r(str(peaks))
    raw = objects["lcms_raw_peaks"].copy()
    retained = objects["lcms_hit_peaks"].copy()
    retained_ids = set(retained["mz_id"].astype(str))

    retained_gap = retained[retained["MSMS_annotation"].map(is_unannotated)].copy()
    retained_gap["source_class"] = "retained_unannotated"

    selected = raw[
        raw["is_tested"].fillna(False).astype(bool)
        & ~raw["mz_id"].astype(str).isin(retained_ids)
    ].copy()
    selected["source_class"] = np.where(
        selected["MSMS_annotation"].map(is_unannotated),
        "selected_unannotated_not_retained",
        "selected_assigned_not_retained",
    )
    ledger = pd.concat([retained_gap, selected], ignore_index=True, sort=False)

    coordinates = ledger["mz_id"].map(mz_id_coordinates)
    ledger["target_rt_sec"] = coordinates.map(lambda item: item[0] * 60.0)
    ledger["mz_id_value"] = coordinates.map(lambda item: item[1])
    ledger["mz_id_kind"] = coordinates.map(lambda item: item[2])
    observed = pd.to_numeric(ledger.get("mz"), errors="coerce")
    direct_from_id = ledger["mz_id_value"].where(ledger["mz_id_kind"].eq("m/z"))
    ledger["target_precursor_mz"] = observed.where(observed.notna(), direct_from_id)
    ledger["precursor_source"] = np.select(
        [observed.notna(), direct_from_id.notna()],
        ["author_observed_mz", "mz_id_observed_mz"],
        default="neutral_mass_only",
    )
    ledger["is_true_gap"] = ~ledger["source_class"].eq("selected_assigned_not_retained")
    ledger["directly_matchable"] = ledger["is_true_gap"] & ledger["target_precursor_mz"].notna()
    ledger["target_id"] = np.where(
        ledger.get("peak_id", pd.Series(index=ledger.index, dtype=object)).fillna("").astype(str).str.len() > 0,
        ledger.get("peak_id").fillna("").astype(str),
        ledger["mz_id"].astype(str),
    )
    return ledger


def number(value: str | None) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def stream_ms2(url: str, timeout_sec: int):
    request = Request(url, headers={"User-Agent": "DreaMS-SgME-coverage-audit/1.0"})
    with urlopen(request, timeout=timeout_sec) as response:
        context = etree.iterparse(
            response, events=("end",), tag=f"{NS}spectrum", huge_tree=True
        )
        try:
            for _, spectrum in context:
                level = 0
                rt_sec = math.nan
                precursor_mz = math.nan
                collision_energy = math.nan
                for cv in spectrum.iterfind(f".//{NS}cvParam"):
                    accession = cv.get("accession")
                    value = cv.get("value")
                    if accession == "MS:1000511":
                        level = int(number(value))
                    elif accession == "MS:1000016":
                        rt_sec = number(value)
                        if cv.get("unitAccession") == "UO:0000031":
                            rt_sec *= 60.0
                    elif accession == "MS:1000744":
                        precursor_mz = number(value)
                    elif accession == "MS:1000045":
                        collision_energy = number(value)
                if level == 2:
                    yield {
                        "spectrum_id": str(spectrum.get("id", "")),
                        "rt_sec": rt_sec,
                        "precursor_mz": precursor_mz,
                        "collision_energy": collision_energy,
                        "n_peaks": int(spectrum.get("defaultArrayLength", "0") or 0),
                    }
                spectrum.clear()
                while spectrum.getprevious() is not None:
                    del spectrum.getparent()[0]
        finally:
            del context


def audit_file(item: dict, mode_targets: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    match_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    ms2_count = 0
    finite_count = 0
    for spectrum in stream_ms2(str(item["url"]), args.timeout_sec):
        ms2_count += 1
        precursor = float(spectrum["precursor_mz"])
        rt_sec = float(spectrum["rt_sec"])
        spectrum_rows.append(
            {
                "case_id": item["case_id"],
                "section_id": item["section_id"],
                "tissue_role": item["tissue_role"],
                "mode": item["mode"],
                "filename": item["filename"],
                **spectrum,
            }
        )
        if not (math.isfinite(precursor) and math.isfinite(rt_sec)):
            continue
        finite_count += 1
        for target in mode_targets.itertuples(index=False):
            ppm_error = abs(precursor - float(target.target_precursor_mz)) / float(target.target_precursor_mz) * 1e6
            rt_error = abs(rt_sec - float(target.target_rt_sec))
            if ppm_error > args.ppm or rt_error > args.rt_tolerance_sec:
                continue
            match_rows.append(
                {
                    "case_id": item["case_id"],
                    "section_id": item["section_id"],
                    "tissue_role": item["tissue_role"],
                    "mode": item["mode"],
                    "filename": item["filename"],
                    "target_id": target.target_id,
                    "source_class": target.source_class,
                    "mz_id": target.mz_id,
                    "precursor_source": target.precursor_source,
                    "target_precursor_mz": target.target_precursor_mz,
                    "observed_precursor_mz": precursor,
                    "ppm_error": ppm_error,
                    "target_rt_sec": target.target_rt_sec,
                    "observed_rt_sec": rt_sec,
                    "rt_error_sec": rt_error,
                    **spectrum,
                }
            )
    file_row = {
        "case_id": item["case_id"],
        "section_id": item["section_id"],
        "tissue_role": item["tissue_role"],
        "mode": item["mode"],
        "filename": item["filename"],
        "ms2_spectra": ms2_count,
        "ms2_with_finite_precursor_rt": finite_count,
        "target_matches": len(match_rows),
    }
    return match_rows, spectrum_rows, file_row


def main() -> None:
    args = parse_args()
    if args.patients < 1:
        raise RuntimeError("--patients must be positive")
    ledger = build_target_ledger(args.peaks)
    pilot = pd.read_csv(args.pilot_manifest)
    patient_order = list(dict.fromkeys(pilot["case_id"].astype(str)))[: args.patients]
    files = pilot[pilot["case_id"].astype(str).isin(patient_order)].copy()
    files = files.sort_values(["case_id", "section_id", "mode", "filename"])
    if args.max_files:
        files = files.head(args.max_files)
    if files.empty:
        raise RuntimeError("no pilot files selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output_dir / "target_ledger.csv", index=False)
    files.to_csv(args.output_dir / "streamed_files.csv", index=False)

    match_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    file_rows: list[dict] = []
    target_seen: dict[str, set[str]] = defaultdict(set)
    target_spectra: dict[str, int] = defaultdict(int)
    target_rows = ledger[ledger["directly_matchable"]].copy()

    file_records = files.to_dict("records")
    with ThreadPoolExecutor(max_workers=min(args.workers, len(file_records))) as executor:
        futures = {
            executor.submit(
                audit_file,
                item,
                target_rows[target_rows["lcms_mode"].eq(item["mode"])],
                args,
            ): item
            for item in file_records
        }
        for position, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            current_matches, current_spectra, current_file = future.result()
            match_rows.extend(current_matches)
            spectrum_rows.extend(current_spectra)
            file_rows.append(current_file)
            for row in current_matches:
                target_seen[str(row["target_id"])].add(str(row["case_id"]))
                target_spectra[str(row["target_id"])] += 1
            print(
                f"[stream {position}/{len(file_records)}] {item['filename']} "
                f"ms2={current_file['ms2_spectra']} target_matches={current_file['target_matches']}",
                flush=True,
            )

    matches = pd.DataFrame(match_rows)
    matches.to_csv(args.output_dir / "target_ms2_matches.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(args.output_dir / "ms2_spectra.csv", index=False)
    file_report = pd.DataFrame(file_rows)
    file_report.to_csv(args.output_dir / "file_coverage.csv", index=False)

    true_gap = ledger[ledger["is_true_gap"]]
    directly_matchable = true_gap[true_gap["directly_matchable"]]
    summary_by_class = (
        ledger.groupby(["source_class", "lcms_mode", "precursor_source"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict("records")
    )
    report = {
        "status": "mtbls13432_sgme_pair_target_ms2_coverage_complete",
        "formal": False,
        "purpose": "low-cost acquisition-coverage rejection gate before an 88-file pilot",
        "target_ledger": {
            "records_in_old_59_upper_bound": int(len(ledger)),
            "true_unannotated_gap_records": int(true_gap.shape[0]),
            "already_assigned_not_retained": int((~ledger["is_true_gap"]).sum()),
            "true_gaps_directly_matchable_to_tissue_dda": int(directly_matchable.shape[0]),
            "true_gaps_neutral_mass_only": int((true_gap["precursor_source"] == "neutral_mass_only").sum()),
            "by_class_mode_precursor_source": summary_by_class,
        },
        "probe": {
            "patients": int(files["case_id"].nunique()),
            "sections": int(files[["case_id", "section_id"]].drop_duplicates().shape[0]),
            "files": int(len(files)),
            "ms2_spectra": int(file_report["ms2_spectra"].sum()),
            "unique_targets_matched": int(len(target_seen)),
            "targets_seen_in_three_patients": int(sum(len(value) >= 3 for value in target_seen.values())),
            "target_spectrum_matches": int(sum(target_spectra.values())),
        },
        "screening_interpretation": {
            "coverage_fraction_of_direct_gaps": (
                float(len(target_seen) / len(directly_matchable)) if len(directly_matchable) else 0.0
            ),
            "not_a_full_gate": True,
            "rule": "A near-zero result among high-abundance targets in three paired patients is evidence against the cost-effectiveness of the 88-file pilot, but cannot prove absence from all 109 sections.",
        },
        "parameters": {
            "ppm": args.ppm,
            "rt_tolerance_sec": args.rt_tolerance_sec,
            "patient_limit": args.patients,
            "max_files": args.max_files,
            "workers": args.workers,
        },
        "provenance": {
            "peaks_sha256": sha256(args.peaks),
            "pilot_manifest_sha256": sha256(args.pilot_manifest),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "This audit measures public tissue-DDA coverage only; matches are not metabolite identifications.",
    }
    output = args.output_dir / "coverage_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
