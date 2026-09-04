#!/usr/bin/env python
"""Freeze the ST001154 CSH-negative external BioAware readiness contract.

This stage aligns author MZ_MSMS annotations to raw negative DDA scans and
checks candidate-library coverage.  It deliberately does not score DreaMS or
BioAware, so the cohort remains outcome-unopened.
"""

from __future__ import annotations

import argparse
import glob
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile


def checksum(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_concatenated_json(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    position = 0
    records: list[dict] = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        record, position = decoder.raw_decode(text, position)
        records.append(record)
    return records


def analysis_record(records: list[dict], analysis_id: str) -> dict:
    matches = [
        record
        for record in records
        if record.get("METABOLOMICS WORKBENCH", {}).get("ANALYSIS_ID") == analysis_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {analysis_id} record; found {len(matches)}")
    return matches[0]


def truth_rows(record: dict) -> pd.DataFrame:
    rows = pd.DataFrame(record["MS_METABOLITE_DATA"]["Metabolites"])
    required = {
        "Metabolite",
        "Type",
        "Adduct",
        "AnnotationApproach",
        "retention times",
        "m/z",
        "InChiKey",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"author truth table lacks columns: {missing}")
    rows["ik14"] = rows["InChiKey"].fillna("").astype(str).str[:14]
    rows["target_mz"] = pd.to_numeric(rows["m/z"], errors="coerce")
    rows["target_rt_sec"] = (
        pd.to_numeric(rows["retention times"], errors="coerce") * 60.0
    )
    return rows


def load_ms2(path: Path) -> tuple[pd.DataFrame, dict]:
    experiment = MSExperiment()
    MzMLFile().load(str(path), experiment)
    levels: dict[int, int] = {}
    records = []
    for spectrum in experiment:
        level = int(spectrum.getMSLevel())
        levels[level] = levels.get(level, 0) + 1
        if level != 2 or not spectrum.getPrecursors():
            continue
        precursor = spectrum.getPrecursors()[0]
        records.append(
            {
                "native_id": spectrum.getNativeID(),
                "observed_rt_sec": float(spectrum.getRT()),
                "observed_precursor_mz": float(precursor.getMZ()),
                "precursor_charge": int(precursor.getCharge()),
                "polarity_code": int(spectrum.getInstrumentSettings().getPolarity()),
                "peak_count": int(spectrum.size()),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("converted ST001154 pilot contains no precursor-bearing MS2")
    if frame["native_id"].duplicated().any():
        raise RuntimeError("duplicate MS2 native IDs in converted pilot")
    return frame, {str(key): value for key, value in sorted(levels.items())}


def align_targets(
    targets: pd.DataFrame, ms2: pd.DataFrame, ppm: float, rt_seconds: float
) -> pd.DataFrame:
    mz = ms2["observed_precursor_mz"].to_numpy(float)
    rt = ms2["observed_rt_sec"].to_numpy(float)
    aligned = []
    for row in targets.itertuples(index=False):
        target_mz = float(row.target_mz)
        target_rt = float(row.target_rt_sec)
        ppm_error = np.abs(mz - target_mz) / target_mz * 1e6
        rt_error = np.abs(rt - target_rt)
        candidates = np.flatnonzero((ppm_error <= ppm) & (rt_error <= rt_seconds))
        if len(candidates) == 0:
            continue
        best = min(candidates, key=lambda index: (ppm_error[index], rt_error[index]))
        record = row._asdict()
        record.update(ms2.iloc[int(best)].to_dict())
        record["absolute_ppm_error"] = float(ppm_error[best])
        record["absolute_rt_error_sec"] = float(rt_error[best])
        record["matched_scans_in_window"] = int(len(candidates))
        aligned.append(record)
    return pd.DataFrame(aligned)


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mwtab",
        type=Path,
        default=Path(
            "data/reference/bioaware_public_cohort_probe_20260901/"
            "ST001154__mwtab__json"
        ),
    )
    parser.add_argument(
        "--mzml",
        type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/mzml/"
            "Flenniken198_negCSH_345321_152.mzML"
        ),
    )
    parser.add_argument(
        "--sample-workbook",
        type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/"
            "KOMP_All_AssaysSampleDetails.xlsx"
        ),
    )
    parser.add_argument(
        "--library-integrity",
        type=Path,
        default=Path(
            "data/validation/mona_negative_library_chemical_integrity_v1/"
            "library_row_integrity.csv.gz"
        ),
    )
    parser.add_argument(
        "--development-units",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_units_v2"),
    )
    parser.add_argument("--analysis-id", default="AN001943")
    parser.add_argument("--primary-ppm", type=float, default=5.0)
    parser.add_argument("--primary-rt-sec", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_st001154_external_readiness_v1"),
    )
    args = parser.parse_args()

    for path in (args.mwtab, args.mzml, args.sample_workbook, args.library_integrity):
        if not path.is_file():
            raise FileNotFoundError(path)

    record = analysis_record(read_concatenated_json(args.mwtab), args.analysis_id)
    all_truth = truth_rows(record)
    exact_truth = all_truth.loc[
        (all_truth["Type"] == "TargetCPD")
        & (all_truth["AnnotationApproach"] == "MZ_MSMS")
        & (all_truth["Adduct"] == "[M-H]-")
        & all_truth["target_mz"].notna()
        & all_truth["target_rt_sec"].notna()
        & (all_truth["ik14"].str.len() == 14)
    ].copy()
    if exact_truth["ik14"].duplicated().any():
        raise RuntimeError("exact [M-H]- truth identities are not unique")

    ms2, level_counts = load_ms2(args.mzml)
    polarity_counts = {
        str(key): int(value)
        for key, value in ms2["polarity_code"].value_counts().sort_index().items()
    }
    if set(ms2["polarity_code"].unique()) != {2}:
        raise RuntimeError(f"pilot is not uniformly negative-polarity MS2: {polarity_counts}")

    cshneg = pd.read_excel(args.sample_workbook, sheet_name="CSHNEG")
    expected_raw = args.mzml.stem + ".raw"
    sample_rows = cshneg.loc[cshneg["FileName"].astype(str) == expected_raw]
    if len(sample_rows) != 1 or str(sample_rows.iloc[0]["SAMPLETYPE"]) != "Study Sample":
        raise RuntimeError("converted RAW is not a uniquely registered CSHNEG Study Sample")

    library = pd.read_csv(args.library_integrity)
    approved = library.loc[bool_series(library["approved_m_h_reference"])].copy()
    approved["ik14"] = approved["inchikey"].fillna("").astype(str).str[:14]
    formula_per_full_key = (
        approved[["inchikey", "calculated_formula"]]
        .drop_duplicates()
        .groupby("inchikey")["calculated_formula"]
        .agg(list)
    )
    if (formula_per_full_key.map(len) != 1).any():
        raise RuntimeError("approved MONA full InChIKeys map to multiple formulas")
    full_key_formula = formula_per_full_key.map(lambda values: values[0]).to_dict()
    formula_per_ik14 = (
        approved[["ik14", "calculated_formula"]]
        .drop_duplicates()
        .groupby("ik14")["calculated_formula"]
        .agg(list)
    )
    unambiguous_ik14_formula = {
        ik14: values[0] for ik14, values in formula_per_ik14.items() if len(values) == 1
    }
    approved_ik14 = set(approved["ik14"])
    formula_candidates = approved.groupby("calculated_formula")["ik14"].nunique().to_dict()

    development_paths = glob.glob(str(args.development_units / "*" / "queries.csv.gz"))
    if not development_paths:
        raise RuntimeError("no frozen BioAware development query ledgers found")
    development_ids = set(
        pd.concat(
            [pd.read_csv(path, usecols=["truth_ik14"]) for path in development_paths],
            ignore_index=True,
        )["truth_ik14"].astype(str)
    )

    sensitivity = {}
    primary = None
    for ppm in (5.0, 10.0):
        for rt_seconds in (6.0, 12.0, 30.0):
            aligned = align_targets(exact_truth, ms2, ppm, rt_seconds)
            sensitivity[f"{ppm:g}ppm_{rt_seconds:g}sec"] = {
                "aligned_truth_identities": int(aligned["ik14"].nunique()),
                "matched_scans_in_windows": int(aligned["matched_scans_in_window"].sum()),
            }
            if ppm == args.primary_ppm and rt_seconds == args.primary_rt_sec:
                primary = aligned
    if primary is None:
        primary = align_targets(exact_truth, ms2, args.primary_ppm, args.primary_rt_sec)

    primary["approved_mona_truth"] = primary["ik14"].isin(approved_ik14)
    primary["truth_formula"] = primary["InChiKey"].map(full_key_formula)
    fallback = primary["truth_formula"].isna()
    primary.loc[fallback, "truth_formula"] = primary.loc[fallback, "ik14"].map(
        unambiguous_ik14_formula
    )
    primary["formula_mapping"] = np.where(
        primary["InChiKey"].isin(full_key_formula),
        "exact_full_inchikey",
        np.where(
            primary["ik14"].isin(unambiguous_ik14_formula),
            "unambiguous_ik14_fallback",
            "unresolved",
        ),
    )
    primary["formula_candidate_identities"] = (
        primary["truth_formula"].map(formula_candidates).fillna(0).astype(int)
    )
    primary["development_identity_overlap"] = primary["ik14"].isin(development_ids)
    primary["formula_ambiguous"] = primary["formula_candidate_identities"] >= 2

    exact_ids = set(exact_truth["ik14"])
    exact_library_ids = exact_ids & approved_ik14
    exact_full_keys = dict(zip(exact_truth["ik14"], exact_truth["InChiKey"]))
    exact_formula = {
        ik14: full_key_formula.get(
            exact_full_keys[ik14], unambiguous_ik14_formula.get(ik14)
        )
        for ik14 in exact_library_ids
    }
    exact_ambiguous = {
        ik14
        for ik14 in exact_library_ids
        if exact_formula[ik14] is not None and formula_candidates[exact_formula[ik14]] >= 2
    }
    primary_ids = set(primary["ik14"])
    primary_library = primary.loc[primary["approved_mona_truth"]]
    primary_ambiguous = primary_library.loc[primary_library["formula_ambiguous"]]
    primary_dev_overlap = primary_ids & development_ids

    gates = {
        "author_exact_m_h_mz_msms_identities_ge_50": len(exact_ids) >= 50,
        "strict_raw_aligned_identities_ge_40": len(primary_ids) >= 40,
        "strict_aligned_approved_mona_truth_ge_25": int(primary_library["ik14"].nunique()) >= 25,
        "strict_aligned_formula_ambiguous_ge_15": int(primary_ambiguous["ik14"].nunique()) >= 15,
        "development_identity_overlap_fraction_le_0_20": (
            len(primary_dev_overlap) / max(len(primary_ids), 1) <= 0.20
        ),
        "uniform_negative_polarity_ms2": set(ms2["polarity_code"].unique()) == {2},
        "biological_study_sample_verified": True,
    }
    report = {
        "status": "bioaware_st001154_external_readiness_complete",
        "formal": True,
        "outcome_status": "unopened; no DreaMS or BioAware scores computed",
        "study": {
            "study_id": "ST001154",
            "analysis_id": args.analysis_id,
            "sample_file": expected_raw,
            "sample_label": str(sample_rows.iloc[0]["KOMPLABEL"]),
            "sample_type": str(sample_rows.iloc[0]["SAMPLETYPE"]),
            "chromatography": "CSH reversed phase",
            "polarity": "negative",
        },
        "raw_scan_audit": {
            "spectra_by_ms_level": level_counts,
            "precursor_bearing_ms2": int(len(ms2)),
            "ms2_polarity_codes": polarity_counts,
            "rt_range_sec": [float(ms2["observed_rt_sec"].min()), float(ms2["observed_rt_sec"].max())],
            "precursor_mz_range": [float(ms2["observed_precursor_mz"].min()), float(ms2["observed_precursor_mz"].max())],
        },
        "author_truth": {
            "all_annotation_rows": int(len(all_truth)),
            "target_mz_msms_rows": int(
                ((all_truth["Type"] == "TargetCPD") & (all_truth["AnnotationApproach"] == "MZ_MSMS")).sum()
            ),
            "exact_m_h_target_mz_msms_identities": len(exact_ids),
            "unsupported_acetate_adduct_rows": int(all_truth["Adduct"].isin(["[M+Hac-H]-", "[M+HAc-H]-"]).sum()),
            "unsupported_formate_adduct_rows": int((all_truth["Adduct"] == "[M+FA-H]-").sum()),
            "exact_truth_in_approved_mona": len(exact_library_ids),
            "exact_truth_formula_ambiguous_in_approved_mona": len(exact_ambiguous),
        },
        "strict_primary_alignment": {
            "ppm": args.primary_ppm,
            "rt_seconds": args.primary_rt_sec,
            "aligned_truth_identities": len(primary_ids),
            "aligned_approved_mona_truth_identities": int(primary_library["ik14"].nunique()),
            "aligned_formula_ambiguous_identities": int(primary_ambiguous["ik14"].nunique()),
            "development_identity_overlap": len(primary_dev_overlap),
            "development_identity_overlap_fraction": len(primary_dev_overlap) / max(len(primary_ids), 1),
        },
        "alignment_sensitivity": sensitivity,
        "gates": gates,
        "pass_to_external_manifest": all(gates.values()),
        "provenance": {
            "mwtab_sha256": checksum(args.mwtab),
            "mzml_sha256": checksum(args.mzml),
            "sample_workbook_sha256": checksum(args.sample_workbook),
            "library_integrity_sha256": checksum(args.library_integrity),
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": (
            "Readiness only. Author MZ_MSMS annotations aligned to raw scans are independent author "
            "structure-resolved targets, not locally reinjected MSI Level 1 standards. No retrieval, "
            "network-context, biological-mechanism, or SOTA claim is tested here."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "strict_aligned_exact_m_h_targets.csv.gz"
    report_path = args.output_dir / "report.json"
    primary.sort_values(["target_rt_sec", "target_mz", "ik14"]).to_csv(
        aligned_path, index=False, compression="gzip"
    )
    report["provenance"]["aligned_targets_sha256"] = checksum(aligned_path)
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_external_manifest"]:
        raise RuntimeError("ST001154 external readiness gates failed")


if __name__ == "__main__":
    main()
