"""Find acquired MS2 spectra that can support technically gated MS1 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/mtbls13729/biology_candidates/candidates_for_ms2_annotation.csv"),
    )
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-sec", type=float, default=20.0)
    parser.add_argument(
        "--peak-resolved-dir",
        type=Path,
        default=None,
        help="Optional per-sample peak-resolved EIC directory; when supplied, link MS2 only inside local peak bounds.",
    )
    parser.add_argument("--peak-boundary-pad-sec", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/biology_candidates"))
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    links: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    for panel in args.panels:
        panel_candidates = candidates[candidates.panel == panel].copy()
        if panel_candidates.empty:
            continue
        target_mz = panel_candidates.mz.to_numpy(float)
        target_rt = panel_candidates.rt_sec.to_numpy(float)
        for path in sorted((args.mzml_root / panel).glob("*.mzML")):
            local_left = target_rt - args.rt_sec
            local_right = target_rt + args.rt_sec
            local_detected = np.ones(len(panel_candidates), dtype=bool)
            if args.peak_resolved_dir is not None:
                peak_path = args.peak_resolved_dir / "per_sample" / f"{panel}__{path.stem}__eic.csv.gz"
                if not peak_path.exists():
                    file_rows.append({
                        "panel": panel,
                        "sample_name": path.stem,
                        # A raw mzML can legitimately be absent from the
                        # upstream EIC sample manifest (P06-Ltu in pos_rp is
                        # the known MTBLS13729 example).  In that case there
                        # is no chromatographic peak boundary against which an
                        # MS2 spectrum can be linked.  This is an explicit
                        # exclusion, not a parser/execution failure.
                        "status": "excluded_no_peak_resolved_eic",
                        "reason": f"Missing peak-resolved EIC: {peak_path}",
                    })
                    continue
                peaks = pd.read_csv(peak_path).set_index("feature_id")
                aligned = peaks.reindex(panel_candidates.feature_id.to_numpy(int))
                local_detected = aligned["detected_eic"].fillna(False).to_numpy(bool)
                local_left = aligned["local_peak_left_rt"].to_numpy(float) - args.peak_boundary_pad_sec
                local_right = aligned["local_peak_right_rt"].to_numpy(float) + args.peak_boundary_pad_sec
            exp = oms.MSExperiment()
            loader = oms.MzMLFile()
            options = loader.getOptions()
            options.setMSLevels([2])
            loader.setOptions(options)
            try:
                loader.load(str(path), exp)
            except Exception as exc:
                file_rows.append({"panel": panel, "sample_name": path.stem, "status": "failed", "error": repr(exc)})
                continue
            n_links = 0
            for scan_index, spectrum in enumerate(exp):
                precursors = spectrum.getPrecursors()
                if not precursors:
                    continue
                precursor = precursors[0]
                precursor_mz = float(precursor.getMZ())
                rt = float(spectrum.getRT())
                ppm_error = np.abs(target_mz - precursor_mz) / target_mz * 1e6
                eligible = np.flatnonzero(
                    (ppm_error <= args.ppm)
                    & local_detected
                    & np.isfinite(local_left)
                    & np.isfinite(local_right)
                    & (rt >= local_left)
                    & (rt <= local_right)
                )
                if not len(eligible):
                    continue
                mzs, intensities = spectrum.get_peaks()
                for index in eligible:
                    candidate = panel_candidates.iloc[int(index)]
                    links.append(
                        {
                            "panel": panel,
                            "feature_id": int(candidate.feature_id),
                            "candidate_mz": float(candidate.mz),
                            "candidate_rt_sec": float(candidate.rt_sec),
                            "sample_name": path.stem,
                            "native_id": spectrum.getNativeID(),
                            "scan_index_ms2_only": scan_index,
                            "ms2_rt_sec": rt,
                            "precursor_mz": precursor_mz,
                            "precursor_charge": int(precursor.getCharge()),
                            "collision_energy": float(precursor.getActivationEnergy()),
                            "ppm_error": float(ppm_error[index]),
                            "rt_error_sec": float(rt - target_rt[index]),
                            "n_fragment_peaks": int(len(mzs)),
                            "fragment_tic": float(np.sum(intensities)),
                        }
                    )
                    n_links += 1
            file_rows.append({"panel": panel, "sample_name": path.stem, "status": "complete", "n_candidate_links": n_links})
            print(f"{panel} {path.stem}: {n_links} candidate MS2 links", flush=True)

    link_frame = pd.DataFrame(links)
    if len(link_frame):
        coverage = (
            link_frame.groupby(["panel", "feature_id"], as_index=False)
            .agg(
                n_ms2_spectra=("native_id", "size"),
                n_samples_with_ms2=("sample_name", "nunique"),
                median_ppm_error=("ppm_error", "median"),
                median_abs_rt_error_sec=("rt_error_sec", lambda x: float(np.median(np.abs(x)))),
                median_fragment_peaks=("n_fragment_peaks", "median"),
            )
        )
        coverage = candidates.merge(coverage, on=["panel", "feature_id"], how="left")
    else:
        coverage = candidates.copy()
    for column in ("n_ms2_spectra", "n_samples_with_ms2"):
        if column not in coverage:
            coverage[column] = 0
        coverage[column] = coverage[column].fillna(0).astype(int)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    links_path = out / "candidate_ms2_links.csv.gz"
    coverage_path = out / "candidate_ms2_coverage.csv"
    files_path = out / "candidate_ms2_file_audit.csv"
    link_frame.to_csv(links_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(file_rows).to_csv(files_path, index=False)
    report = {
        "status": "complete_with_failures" if any(row["status"] == "failed" for row in file_rows) else "complete",
        "file_status_counts": pd.Series([row["status"] for row in file_rows]).value_counts().to_dict(),
        "n_candidates": int(len(candidates)),
        "n_candidates_with_ms2": int((coverage.n_ms2_spectra > 0).sum()),
        "n_ms2_links": int(len(link_frame)),
        "links": str(links_path),
        "coverage": str(coverage_path),
        "file_audit": str(files_path),
        "parameters": {
            "ppm": args.ppm,
            "rt_sec": args.rt_sec,
            "peak_resolved_dir": str(args.peak_resolved_dir) if args.peak_resolved_dir else None,
            "peak_boundary_pad_sec": args.peak_boundary_pad_sec,
        },
        "interpretation_limit": "Precursor/RT linkage identifies supporting spectra; it does not identify a structure.",
    }
    (out / "candidate_ms2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
