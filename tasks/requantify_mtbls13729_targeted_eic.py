#!/usr/bin/env python
"""Uniform targeted-EIC re-quantification of MTBLS13729 consensus features.

Every consensus target is re-extracted from every raw MS1 file with identical
logic. This avoids mixing OpenMS discovery intensities with a different gap-fill
scale and prevents non-detection from being silently encoded as biological zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


SAMPLE_RE = re.compile(r"^P\d{2}-(?:Ltu|Rtu|Rmu|LN|RN)$")


def parameter_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "ppm": float(args.ppm),
        "rt_half_window_sec": float(args.rt_half_window_sec),
        "resolve_local_peaks": bool(args.resolve_local_peaks),
        "max_apex_delta_sec": float(args.max_apex_delta_sec),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_eics(
    path: Path,
    targets: pd.DataFrame,
    ppm: float,
    rt_half_window: float,
    resolve_local_peaks: bool = False,
    max_apex_delta_sec: float = 12.0,
) -> pd.DataFrame:
    experiment = oms.MSExperiment()
    loader = oms.MzMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)

    target_rt = targets["rt_sec"].to_numpy(float)
    target_mz = targets["mz"].to_numpy(float)
    order = np.argsort(target_rt)
    sorted_rt = target_rt[order]
    traces: list[list[tuple[float, float]]] = [[] for _ in range(len(targets))]

    for spectrum in experiment:
        rt = float(spectrum.getRT())
        left = int(np.searchsorted(sorted_rt, rt - rt_half_window, side="left"))
        right = int(np.searchsorted(sorted_rt, rt + rt_half_window, side="right"))
        active = order[left:right]
        if not len(active):
            continue
        mzs, intensities = spectrum.get_peaks()
        mzs = np.asarray(mzs, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        for target_idx in active:
            mz = target_mz[target_idx]
            tol = mz * ppm * 1e-6
            lo = int(np.searchsorted(mzs, mz - tol, side="left"))
            hi = int(np.searchsorted(mzs, mz + tol, side="right"))
            value = float(np.max(intensities[lo:hi])) if hi > lo else 0.0
            traces[target_idx].append((rt, value))

    rows = []
    for idx, trace in enumerate(traces):
        if not trace:
            rows.append(
                {
                    "feature_id": int(targets.iloc[idx]["feature_id"]),
                    "eic_auc": math.nan,
                    "eic_apex": math.nan,
                    "eic_apex_rt": math.nan,
                    "eic_apex_delta_sec": math.nan,
                    "eic_baseline": math.nan,
                    "eic_snr": math.nan,
                    "n_eic_scans": 0,
                    "detected_eic": False,
                }
            )
            continue
        times = np.asarray([item[0] for item in trace], dtype=float)
        signal = np.asarray([item[1] for item in trace], dtype=float)
        baseline = float(np.percentile(signal, 20))
        corrected = np.maximum(signal - baseline, 0.0)
        noise = max(float(1.4826 * np.median(np.abs(signal - np.median(signal)))), math.sqrt(max(baseline, 0.0) + 1.0), 1.0)
        local_peak_count = 1
        local_left_rt = float(times[0])
        local_right_rt = float(times[-1])
        local_prominence = math.nan
        if resolve_local_peaks and len(times) >= 5 and np.max(corrected) > 0:
            smoothed = gaussian_filter1d(corrected, sigma=1.0, mode="nearest")
            min_prominence = max(3.0 * noise, 0.01 * float(np.max(smoothed)))
            peak_indices, properties = find_peaks(smoothed, prominence=min_prominence, distance=2)
            eligible = np.flatnonzero(np.abs(times[peak_indices] - target_rt[idx]) <= max_apex_delta_sec)
            local_peak_count = int(len(peak_indices))
            if not len(eligible):
                rows.append(
                    {
                        "feature_id": int(targets.iloc[idx]["feature_id"]),
                        "eic_auc": math.nan,
                        "eic_apex": math.nan,
                        "eic_apex_rt": math.nan,
                        "eic_apex_delta_sec": math.nan,
                        "eic_baseline": baseline,
                        "eic_snr": math.nan,
                        "n_eic_scans": int(len(times)),
                        "detected_eic": False,
                        "local_peak_count": local_peak_count,
                        "local_peak_left_rt": math.nan,
                        "local_peak_right_rt": math.nan,
                        "local_peak_prominence": math.nan,
                    }
                )
                continue
            # The consensus RT defines the chromatographic identity.  Prefer the
            # nearest resolved apex; use prominence only to break near-ties.
            candidate_peaks = peak_indices[eligible]
            candidate_prominence = properties["prominences"][eligible]
            distance = np.abs(times[candidate_peaks] - target_rt[idx])
            selection_order = np.lexsort((-candidate_prominence, distance))
            chosen_position = int(eligible[selection_order[0]])
            apex_idx = int(peak_indices[chosen_position])
            local_prominence = float(properties["prominences"][chosen_position])
            widths = peak_widths(smoothed, [apex_idx], rel_height=0.95)
            left_idx = max(0, int(math.floor(float(widths[2][0]))))
            right_idx = min(len(times) - 1, int(math.ceil(float(widths[3][0]))))
            local_left_rt = float(times[left_idx])
            local_right_rt = float(times[right_idx])
            local_times = times[left_idx : right_idx + 1]
            local_corrected = corrected[left_idx : right_idx + 1]
        else:
            apex_idx = int(np.argmax(corrected))
            local_times = times
            local_corrected = corrected

        apex = float(corrected[apex_idx])
        apex_rt = float(times[apex_idx])
        snr = apex / noise
        auc = float(np.trapz(local_corrected, local_times)) if len(local_times) > 1 else 0.0
        apex_delta = apex_rt - target_rt[idx]
        allowed_delta = max_apex_delta_sec if resolve_local_peaks else rt_half_window * 0.75
        detected = bool(snr >= 3.0 and abs(apex_delta) <= allowed_delta and auc > 0)
        rows.append(
            {
                "feature_id": int(targets.iloc[idx]["feature_id"]),
                "eic_auc": auc,
                "eic_apex": apex,
                "eic_apex_rt": apex_rt,
                "eic_apex_delta_sec": apex_delta,
                "eic_baseline": baseline,
                "eic_snr": snr,
                "n_eic_scans": int(len(times)),
                "detected_eic": detected,
                "local_peak_count": local_peak_count,
                "local_peak_left_rt": local_left_rt,
                "local_peak_right_rt": local_right_rt,
                "local_peak_prominence": local_prominence,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-half-window-sec", type=float, default=20.0)
    parser.add_argument("--resolve-local-peaks", action="store_true")
    parser.add_argument("--max-apex-delta-sec", type=float, default=12.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.mzml_root.resolve()
    consensus_dir = args.consensus_dir.resolve()
    out = args.output_dir.resolve()
    per_sample = out / "per_sample"
    per_sample.mkdir(parents=True, exist_ok=True)
    full_report = {"status": "complete", "panels": {}}

    for panel in args.panels:
        targets = pd.read_csv(consensus_dir / f"{panel}__requantification_targets.csv.gz")
        samples = pd.read_csv(consensus_dir / f"{panel}__samples.csv")
        manifest = []
        if args.max_samples:
            samples = samples.iloc[: args.max_samples].copy()
        for i, row in enumerate(samples.itertuples(index=False), start=1):
            sample = str(row.sample_name)
            if not SAMPLE_RE.match(sample):
                continue
            source = root / panel / f"{sample}.mzML"
            target_path = per_sample / f"{panel}__{sample}__eic.csv.gz"
            metadata_path = per_sample / f"{panel}__{sample}__eic.meta.json"
            reusable = False
            if target_path.exists() and metadata_path.exists() and not args.force:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                reusable = (
                    metadata.get("parameters") == parameter_signature(args)
                    and metadata.get("targets_sha256")
                    == file_sha256(consensus_dir / f"{panel}__requantification_targets.csv.gz")
                    and metadata.get("source_bytes") == source.stat().st_size
                )
            if reusable:
                manifest.append({"panel": panel, "sample_name": sample, "status": "reused", "output": str(target_path)})
                print(f"[{panel} {i}/{len(samples)}] reused {sample}", flush=True)
                continue
            started = time.time()
            print(f"[{panel} {i}/{len(samples)}] extracting {len(targets)} targets from {sample}", flush=True)
            try:
                values = extract_eics(
                    source,
                    targets,
                    args.ppm,
                    args.rt_half_window_sec,
                    resolve_local_peaks=args.resolve_local_peaks,
                    max_apex_delta_sec=args.max_apex_delta_sec,
                )
                values.insert(0, "sample_name", sample)
                values.insert(0, "panel", panel)
                values.to_csv(target_path, index=False)
                metadata_path.write_text(
                    json.dumps(
                        {
                            "panel": panel,
                            "sample_name": sample,
                            "parameters": parameter_signature(args),
                            "targets_sha256": file_sha256(
                                consensus_dir / f"{panel}__requantification_targets.csv.gz"
                            ),
                            "source_bytes": source.stat().st_size,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                manifest.append(
                    {
                        "panel": panel,
                        "sample_name": sample,
                        "status": "complete",
                        "output": str(target_path),
                        "detected_fraction": float(values["detected_eic"].mean()),
                        "elapsed_seconds": time.time() - started,
                    }
                )
            except Exception as exc:
                manifest.append({"panel": panel, "sample_name": sample, "status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started})
            pd.DataFrame(manifest).to_csv(out / f"{panel}__manifest.partial.csv", index=False)

        manifest_frame = pd.DataFrame(manifest)
        manifest_frame.to_csv(out / f"{panel}__manifest.csv", index=False)
        usable = manifest_frame[manifest_frame["status"].isin(["complete", "reused"])]
        auc_parts = []
        detect_parts = []
        snr_parts = []
        for row in usable.itertuples(index=False):
            frame = pd.read_csv(row.output)
            sample = str(row.sample_name)
            auc_parts.append(frame.set_index("feature_id")["eic_auc"].rename(sample))
            detect_parts.append(frame.set_index("feature_id")["detected_eic"].rename(sample))
            snr_parts.append(frame.set_index("feature_id")["eic_snr"].rename(sample))
        pd.concat(auc_parts, axis=1).to_csv(out / f"{panel}__eic_auc_matrix.csv.gz")
        pd.concat(detect_parts, axis=1).to_csv(out / f"{panel}__eic_detection_matrix.csv.gz")
        pd.concat(snr_parts, axis=1).to_csv(out / f"{panel}__eic_snr_matrix.csv.gz")
        full_report["panels"][panel] = {
            "n_targets": len(targets),
            "status_counts": manifest_frame["status"].value_counts().to_dict(),
            "median_detected_fraction": float(manifest_frame.get("detected_fraction", pd.Series(dtype=float)).median()),
            "resolve_local_peaks": bool(args.resolve_local_peaks),
            "max_apex_delta_sec": float(args.max_apex_delta_sec),
            "parameters": parameter_signature(args),
            "all_samples_parameter_locked": bool(
                len(usable) == len(samples)
                and all(
                    (per_sample / f"{panel}__{sample}__eic.meta.json").exists()
                    for sample in usable["sample_name"].astype(str)
                )
            ),
        }
    (out / "report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(json.dumps(full_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
