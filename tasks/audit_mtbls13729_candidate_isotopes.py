"""Check isotope-spacing and co-elution evidence for provisional MS1 candidates.

The output helps distinguish a plausible carbon-containing ion from an isolated
noise/interference peak and gives a first indication of charge state.  It does
not identify a metabolite and is deliberately independent of any library name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


C13_SHIFT = 1.00335483507


def intensity_at(mzs: np.ndarray, intensities: np.ndarray, target: float, ppm: float) -> float:
    tolerance = target * ppm * 1e-6
    left = int(np.searchsorted(mzs, target - tolerance, side="left"))
    right = int(np.searchsorted(mzs, target + tolerance, side="right"))
    return float(np.max(intensities[left:right])) if right > left else 0.0


def trace_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 5 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def inspect_file(
    path: Path, targets: pd.DataFrame, ppm: float, rt_window: float, max_charge: int
) -> list[dict[str, object]]:
    exp = oms.MSExperiment()
    loader = oms.MzMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    loader.load(str(path), exp)

    results: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        times: list[float] = []
        base: list[float] = []
        charges = range(1, max_charge + 1)
        isotope_traces: dict[int, list[float]] = {charge: [] for charge in charges}
        pre_traces: dict[int, list[float]] = {charge: [] for charge in charges}
        for spectrum in exp:
            rt = float(spectrum.getRT())
            if abs(rt - float(target.rt_sec)) > rt_window:
                continue
            mzs, intensities = spectrum.get_peaks()
            mzs = np.asarray(mzs, dtype=float)
            intensities = np.asarray(intensities, dtype=float)
            times.append(rt)
            base.append(intensity_at(mzs, intensities, float(target.mz), ppm))
            for charge in charges:
                shift = C13_SHIFT / charge
                isotope_traces[charge].append(intensity_at(mzs, intensities, float(target.mz) + shift, ppm))
                pre_traces[charge].append(intensity_at(mzs, intensities, float(target.mz) - shift, ppm))

        time = np.asarray(times, dtype=float)
        base_arr = np.asarray(base, dtype=float)
        if not len(base_arr) or np.max(base_arr) <= 0:
            continue
        base_apex = int(np.argmax(base_arr))
        for charge in charges:
            isotope = np.asarray(isotope_traces[charge], dtype=float)
            pre = np.asarray(pre_traces[charge], dtype=float)
            isotope_apex = int(np.argmax(isotope)) if np.max(isotope) > 0 else -1
            results.append(
                {
                    "sample_name": path.stem,
                    "feature_id": int(target.feature_id),
                    "mz": float(target.mz),
                    "rt_sec": float(target.rt_sec),
                    "charge_hypothesis": charge,
                    "n_scans": int(len(base_arr)),
                    "base_apex": float(base_arr[base_apex]),
                    "base_apex_rt": float(time[base_apex]),
                    "isotope_apex": float(np.max(isotope)),
                    "isotope_to_base_apex_ratio": float(np.max(isotope) / base_arr[base_apex]),
                    "isotope_trace_correlation": trace_correlation(base_arr, isotope),
                    "isotope_apex_rt_delta_sec": (
                        float(time[isotope_apex] - time[base_apex]) if isotope_apex >= 0 else np.nan
                    ),
                    "preceding_peak_apex_ratio": float(np.max(pre) / base_arr[base_apex]),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="neg_rp")
    parser.add_argument(
        "--priority-table",
        type=Path,
        default=Path("data/mtbls13729/ms1_paired_analysis/neg_rp__discovery_priority_features.csv"),
    )
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--samples", nargs="*", default=["P21-Rmu", "P21-RN", "P27-Rmu", "P27-RN", "P29-Rmu", "P29-RN"])
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-window-sec", type=float, default=20.0)
    parser.add_argument("--max-charge", type=int, default=6)
    parser.add_argument("--rt-min", type=float, default=None)
    parser.add_argument("--rt-max", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/discovery_candidate_audit"),
    )
    args = parser.parse_args()

    targets = pd.read_csv(args.priority_table)[["feature_id", "mz", "rt_sec"]]
    if args.rt_min is not None:
        targets = targets[targets.rt_sec >= args.rt_min]
    if args.rt_max is not None:
        targets = targets[targets.rt_sec <= args.rt_max]
    rows: list[dict[str, object]] = []
    for sample in args.samples:
        path = args.mzml_root / args.panel / f"{sample}.mzML"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Inspecting {sample}", flush=True)
        rows.extend(inspect_file(path, targets, args.ppm, args.rt_window_sec, args.max_charge))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["feature_id", "charge_hypothesis"], as_index=False)
        .agg(
            n_samples=("sample_name", "nunique"),
            median_isotope_ratio=("isotope_to_base_apex_ratio", "median"),
            median_trace_correlation=("isotope_trace_correlation", "median"),
            median_abs_isotope_rt_delta_sec=("isotope_apex_rt_delta_sec", lambda x: float(np.nanmedian(np.abs(x)))),
            median_preceding_peak_ratio=("preceding_peak_apex_ratio", "median"),
        )
    )
    summary["plausible_isotope_support"] = (
        (summary.median_isotope_ratio > 0.01)
        & (summary.median_trace_correlation > 0.7)
        & (summary.median_abs_isotope_rt_delta_sec < 3.0)
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / f"{args.panel}__candidate_isotope_details.csv"
    summary_path = out / f"{args.panel}__candidate_isotope_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "status": "complete",
        "panel": args.panel,
        "samples": args.samples,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "interpretation_limit": "Isotope co-elution supports an ion/charge hypothesis; it does not establish molecular identity.",
    }
    (out / f"{args.panel}__candidate_isotope_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
