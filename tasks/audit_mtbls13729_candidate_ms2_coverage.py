#!/usr/bin/env python
"""Audit raw DDA coverage for the frozen MTBLS13729 biology candidates.

This audit deliberately separates acquisition coverage from structural
identification.  A precursor/RT match only establishes that an MS2 spectrum
was acquired near an MS1 feature.  It is not an annotation by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "data/mtbls13729/biology_closure_analysis_v1/"
            "candidate_identity_and_abundance.csv"
        ),
    )
    parser.add_argument(
        "--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_rp")
    )
    parser.add_argument(
        "--peak-resolved-dir",
        type=Path,
        default=Path("data/mtbls13729/biology_closure_eic_v1/per_sample"),
    )
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fixed-rt-window-sec", type=float, default=15.0)
    parser.add_argument("--peak-boundary-pad-sec", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/frozen_candidate_ms2_coverage_v1"),
    )
    args = parser.parse_args()

    try:
        import pyopenms as oms
    except ImportError as exc:
        raise RuntimeError("pyopenms is required to read mzML") from exc

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates).copy()
    required = {"feature_id", "biology_label", "mz", "rt_sec"}
    missing = required.difference(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate table is missing columns: {sorted(missing)}")
    candidates = candidates.sort_values("feature_id").reset_index(drop=True)
    target_mz = candidates["mz"].to_numpy(float)
    target_rt = candidates["rt_sec"].to_numpy(float)

    paths = sorted(args.mzml_dir.glob("*.mzML"))
    if not paths:
        raise FileNotFoundError(f"no mzML files under {args.mzml_dir}")

    matches: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for number, path in enumerate(paths, start=1):
        detected = np.zeros(len(candidates), dtype=bool)
        left = np.full(len(candidates), np.nan, dtype=float)
        right = np.full(len(candidates), np.nan, dtype=float)
        eic_path = args.peak_resolved_dir / f"pos_rp__{path.stem}__eic.csv.gz"
        if eic_path.exists():
            eic = pd.read_csv(eic_path).set_index("feature_id")
            aligned = eic.reindex(candidates["feature_id"].to_numpy(int))
            detected = aligned["detected_eic"].fillna(False).to_numpy(bool)
            left = (
                aligned["local_peak_left_rt"].to_numpy(float)
                - args.peak_boundary_pad_sec
            )
            right = (
                aligned["local_peak_right_rt"].to_numpy(float)
                + args.peak_boundary_pad_sec
            )

        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        try:
            loader.load(str(path), experiment)
        except Exception as exc:
            files.append(
                {
                    "sample": path.stem,
                    "status": "failed",
                    "eic_available": eic_path.exists(),
                    "error": repr(exc),
                }
            )
            continue

        sample_matches = 0
        for spectrum in experiment:
            precursors = spectrum.getPrecursors()
            if not precursors:
                continue
            precursor = precursors[0]
            precursor_mz = float(precursor.getMZ())
            rt_sec = float(spectrum.getRT())
            ppm_error = (precursor_mz - target_mz) / target_mz * 1e6
            eligible = np.flatnonzero(
                (np.abs(ppm_error) <= args.precursor_ppm)
                & (np.abs(rt_sec - target_rt) <= args.fixed_rt_window_sec)
            )
            if not len(eligible):
                continue
            fragment_mz, fragment_intensity = spectrum.get_peaks()
            for index in eligible:
                peak_resolved = bool(
                    detected[index]
                    and np.isfinite(left[index])
                    and np.isfinite(right[index])
                    and left[index] <= rt_sec <= right[index]
                )
                matches.append(
                    {
                        "feature_id": int(candidates.iloc[index].feature_id),
                        "biology_label": str(candidates.iloc[index].biology_label),
                        "sample": path.stem,
                        "native_id": spectrum.getNativeID(),
                        "precursor_mz": precursor_mz,
                        "precursor_ppm_error": float(ppm_error[index]),
                        "ms2_rt_sec": rt_sec,
                        "rt_error_sec": float(rt_sec - target_rt[index]),
                        "n_fragment_peaks": int(len(fragment_mz)),
                        "fragment_tic": float(np.sum(fragment_intensity)),
                        "eic_available": bool(eic_path.exists()),
                        "eic_detected": bool(detected[index]),
                        "peak_resolved_match": peak_resolved,
                    }
                )
                sample_matches += 1
        files.append(
            {
                "sample": path.stem,
                "status": "complete",
                "eic_available": eic_path.exists(),
                "n_fixed_window_matches": sample_matches,
            }
        )
        if number % 10 == 0 or number == len(paths):
            print(f"[candidate MS2] {number}/{len(paths)} files", flush=True)

    detail = pd.DataFrame(matches)
    if detail.empty:
        raise RuntimeError("no candidate had a precursor/RT-matched MS2 spectrum")

    summary_rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        subset = detail[detail.feature_id == int(candidate.feature_id)]
        resolved = subset[subset.peak_resolved_match]
        summary_rows.append(
            {
                "feature_id": int(candidate.feature_id),
                "biology_label": str(candidate.biology_label),
                "mz": float(candidate.mz),
                "rt_sec": float(candidate.rt_sec),
                "fixed_window_ms2_spectra": int(len(subset)),
                "fixed_window_samples": int(subset["sample"].nunique()),
                "peak_resolved_ms2_spectra": int(len(resolved)),
                "peak_resolved_samples": int(resolved["sample"].nunique()),
                "median_abs_ppm_error": (
                    float(np.median(np.abs(resolved.precursor_ppm_error)))
                    if len(resolved)
                    else np.nan
                ),
                "median_abs_rt_error_sec": (
                    float(np.median(np.abs(resolved.rt_error_sec)))
                    if len(resolved)
                    else np.nan
                ),
                "median_fragment_peaks": (
                    float(np.median(resolved.n_fragment_peaks))
                    if len(resolved)
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    detail_path = output / "candidate_ms2_matches.csv.gz"
    summary_path = output / "candidate_ms2_coverage.csv"
    files_path = output / "mzml_file_audit.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(files).to_csv(files_path, index=False)
    payload = {
        "status": "mtbls13729_frozen_candidate_ms2_coverage_complete",
        "formal": True,
        "candidates": int(len(candidates)),
        "mzml_files": int(len(paths)),
        "failed_mzml_files": int(sum(row["status"] == "failed" for row in files)),
        "samples_with_peak_resolved_eic": int(
            sum(bool(row["eic_available"]) for row in files)
        ),
        "candidates_with_fixed_window_ms2": int(
            (summary.fixed_window_ms2_spectra > 0).sum()
        ),
        "candidates_with_peak_resolved_ms2": int(
            (summary.peak_resolved_ms2_spectra > 0).sum()
        ),
        "parameters": {
            "precursor_ppm": args.precursor_ppm,
            "fixed_rt_window_sec": args.fixed_rt_window_sec,
            "peak_boundary_pad_sec": args.peak_boundary_pad_sec,
        },
        "outputs": {
            "summary": str(summary_path),
            "detail": str(detail_path),
            "file_audit": str(files_path),
        },
        "claim_limit": (
            "Precursor/RT linkage establishes DDA acquisition coverage only. "
            "It does not establish a molecular structure, adduct, or MSI Level 2 identity."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
