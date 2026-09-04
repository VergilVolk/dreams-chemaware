#!/usr/bin/env python
"""Phenotype-blind audit of mono-O-acetyl-Neu5Ac-like exact-mass features.

The assay deliberately separates four evidence layers:

1. exact mass at C13H21NO10 [M-H]- (m/z 350.109269),
2. phenotype-blind chromatographic peak discovery in all 60 negative-HILIC runs,
3. RT-resolved DDA-MS2 motif support, and
4. paired abundance effects evaluated only after the RT features are frozen.

Neither exact mass nor the monitored product ions distinguish 4-, 7-, 8- and
9-O-acetyl positional isomers.  Accordingly, all outputs retain the suffix
``-like`` and cannot establish an O-acetylation site without a co-eluting
authentic standard or an isomer-resolving method.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyopenms as oms
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")
TARGET_MZ = 350.109269
FORMULA = "C13H21NO10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/neg_hilic")
    )
    parser.add_argument(
        "--donor-deltas",
        type=Path,
        default=Path(
            "data/mtbls13729/sialic_donor_decoupling_v1/"
            "rmu_patient_sialic_donor_deltas.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/oacetyl_neu5ac_like_v2"),
    )
    parser.add_argument("--mz", type=float, default=TARGET_MZ)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-cluster-diameter-sec", type=float, default=8.0)
    parser.add_argument("--minimum-cluster-samples", type=int, default=12)
    parser.add_argument("--minimum-feature-separation-sec", type=float, default=18.0)
    parser.add_argument("--integration-half-window-sec", type=float, default=9.0)
    parser.add_argument("--maximum-apex-delta-sec", type=float, default=8.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sample(sample: str) -> tuple[str, str]:
    match = SAMPLE_RE.fullmatch(sample)
    if match is None:
        raise RuntimeError(f"unexpected sample identifier: {sample}")
    return match.group(1), match.group(2)


def robust_noise(signal: np.ndarray, baseline: float) -> float:
    if len(signal) > 1:
        difference = np.diff(signal)
        difference_noise = 1.4826 * float(
            np.median(np.abs(difference - np.median(difference)))
        ) / math.sqrt(2.0)
    else:
        difference_noise = 0.0
    return max(difference_noise, math.sqrt(max(baseline, 0.0) + 1.0), 1.0)


def load_ms1_trace(
    path: Path, target_mz: float, ppm: float
) -> tuple[np.ndarray, np.ndarray]:
    experiment = oms.MSExperiment()
    loader = oms.MzMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)
    tolerance = target_mz * ppm * 1e-6
    times: list[float] = []
    signal: list[float] = []
    for spectrum in experiment:
        mzs, intensities = spectrum.get_peaks()
        mzs = np.asarray(mzs, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        lo = int(np.searchsorted(mzs, target_mz - tolerance, side="left"))
        hi = int(np.searchsorted(mzs, target_mz + tolerance, side="right"))
        times.append(float(spectrum.getRT()))
        signal.append(float(np.max(intensities[lo:hi])) if hi > lo else 0.0)
    if not times:
        raise RuntimeError(f"no MS1 spectra in {path}")
    return np.asarray(times, dtype=float), np.asarray(signal, dtype=float)


def discover_sample_peaks(
    sample: str, times: np.ndarray, signal: np.ndarray
) -> list[dict[str, float | str]]:
    baseline = float(np.percentile(signal, 20))
    corrected = np.maximum(signal - baseline, 0.0)
    noise = robust_noise(signal, baseline)
    smoothed = gaussian_filter1d(corrected, sigma=1.0, mode="nearest")
    prominence = max(4.0 * noise, 0.005 * float(np.max(smoothed)))
    indices, properties = find_peaks(smoothed, prominence=prominence, distance=3)
    if len(indices) > 12:
        keep = np.argsort(properties["prominences"])[-12:]
        indices = indices[keep]
        prominences = properties["prominences"][keep]
    else:
        prominences = properties["prominences"]
    return [
        {
            "sample": sample,
            "apex_rt_sec": float(times[index]),
            "apex_intensity": float(corrected[index]),
            "prominence": float(prominence_value),
            "snr": float(corrected[index] / noise),
        }
        for index, prominence_value in zip(indices, prominences)
    ]


def discover_population_features(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    peaks: pd.DataFrame,
    support_radius_sec: float,
    minimum_samples: int,
    minimum_separation_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze non-overlapping RT features from a phenotype-blind mean trace.

    Each sample first contributes a unit-scale trace, preventing a few high-
    intensity samples from choosing the RT features.  Candidate population
    peaks are then required to have an independently detected sample peak in
    at least ``minimum_samples`` runs.
    """
    if peaks.empty:
        raise RuntimeError("no exact-mass chromatographic peaks were detected")
    grid = np.arange(
        min(float(times.min()) for times, _ in traces.values()),
        max(float(times.max()) for times, _ in traces.values()) + 0.5,
        0.5,
    )
    normalized = []
    for times, signal in traces.values():
        baseline = float(np.percentile(signal, 20))
        corrected = np.maximum(signal - baseline, 0.0)
        scale = float(np.percentile(corrected, 99.5))
        if scale <= 0:
            normalized.append(np.zeros_like(grid))
            continue
        normalized.append(np.interp(grid, times, corrected / scale, left=0.0, right=0.0))
    matrix = np.asarray(normalized, dtype=float)
    consensus = gaussian_filter1d(np.mean(matrix, axis=0), sigma=2.0, mode="nearest")
    prominence = max(0.01, 0.02 * float(np.max(consensus)))
    distance_bins = max(1, int(round(minimum_separation_sec / 0.5)))
    indices, properties = find_peaks(
        consensus, prominence=prominence, distance=distance_bins
    )
    candidates = []
    for index, population_prominence in zip(indices, properties["prominences"]):
        center = float(grid[index])
        nearby = peaks.loc[np.abs(peaks["apex_rt_sec"] - center) <= support_radius_sec]
        sample_support = int(nearby["sample"].nunique())
        if sample_support < minimum_samples:
            continue
        candidates.append(
            {
                "median_rt_sec": center,
                "samples": sample_support,
                "prevalence": sample_support / len(traces),
                "peak_calls": int(len(nearby)),
                "rt_min_sec": float(nearby["apex_rt_sec"].min()),
                "rt_max_sec": float(nearby["apex_rt_sec"].max()),
                "median_prominence": float(nearby["prominence"].median()),
                "median_snr": float(nearby["snr"].median()),
                "population_prominence": float(population_prominence),
                "population_height": float(consensus[index]),
            }
        )
    summary = pd.DataFrame(candidates)
    if summary.empty:
        raise RuntimeError(
            "no phenotype-blind RT cluster met the frozen sample-prevalence gate"
        )
    summary = summary.sort_values("median_rt_sec").reset_index(drop=True)
    summary.insert(0, "feature_id", [f"OAc-like-{i + 1:02d}" for i in range(len(summary))])
    assigned = peaks.copy()
    centers = summary["median_rt_sec"].to_numpy(float)
    nearest = np.argmin(
        np.abs(assigned["apex_rt_sec"].to_numpy(float)[:, None] - centers[None, :]),
        axis=1,
    )
    distance = np.abs(assigned["apex_rt_sec"].to_numpy(float) - centers[nearest])
    assigned["feature_id"] = np.where(
        distance <= support_radius_sec,
        summary.iloc[nearest]["feature_id"].to_numpy(),
        None,
    )
    assigned = assigned.loc[assigned["feature_id"].notna()].copy()
    consensus_frame = pd.DataFrame(
        {
            "rt_sec": grid,
            "mean_unit_scaled_intensity": consensus,
            "sample_median_unit_scaled_intensity": np.median(matrix, axis=0),
            "sample_nonzero_fraction": np.mean(matrix > 0, axis=0),
        }
    )
    return assigned, summary, consensus_frame


def quantify_feature(
    times: np.ndarray,
    signal: np.ndarray,
    center: float,
    half_window: float,
    maximum_apex_delta: float,
) -> dict[str, float | bool | int]:
    active = np.abs(times - center) <= half_window
    local_times = times[active]
    local_signal = signal[active]
    if len(local_times) < 5:
        return {
            "area": 0.0,
            "apex": 0.0,
            "apex_rt_sec": math.nan,
            "apex_delta_sec": math.nan,
            "snr": 0.0,
            "detected": False,
            "local_peak_count": 0,
            "left_rt_sec": math.nan,
            "right_rt_sec": math.nan,
        }
    baseline = float(np.percentile(local_signal, 20))
    corrected = np.maximum(local_signal - baseline, 0.0)
    noise = robust_noise(local_signal, baseline)
    smoothed = gaussian_filter1d(corrected, sigma=1.0, mode="nearest")
    prominence = max(3.0 * noise, 0.005 * float(np.max(smoothed)))
    indices, properties = find_peaks(smoothed, prominence=prominence, distance=2)
    eligible = np.flatnonzero(
        np.abs(local_times[indices] - center) <= maximum_apex_delta
    )
    if not len(eligible):
        return {
            "area": 0.0,
            "apex": float(np.max(corrected)),
            "apex_rt_sec": math.nan,
            "apex_delta_sec": math.nan,
            "snr": float(np.max(corrected) / noise),
            "detected": False,
            "local_peak_count": int(len(indices)),
            "left_rt_sec": math.nan,
            "right_rt_sec": math.nan,
        }
    candidates = indices[eligible]
    candidate_prominence = properties["prominences"][eligible]
    distances = np.abs(local_times[candidates] - center)
    chosen_position = int(np.lexsort((-candidate_prominence, distances))[0])
    peak_index = int(candidates[chosen_position])
    widths = peak_widths(smoothed, [peak_index], rel_height=0.95)
    left = max(0, int(math.floor(float(widths[2][0]))))
    right = min(len(local_times) - 1, int(math.ceil(float(widths[3][0]))))
    apex = float(corrected[peak_index])
    apex_rt = float(local_times[peak_index])
    area = float(np.trapezoid(corrected[left : right + 1], local_times[left : right + 1]))
    snr = apex / noise
    detected = bool(snr >= 3.0 and area > 0.0)
    return {
        "area": area,
        "apex": apex,
        "apex_rt_sec": apex_rt,
        "apex_delta_sec": apex_rt - center,
        "snr": snr,
        "detected": detected,
        "local_peak_count": int(len(indices)),
        "left_rt_sec": float(local_times[left]),
        "right_rt_sec": float(local_times[right]),
    }


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return math.nan
    observed = abs(float(np.mean(values)))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float(np.mean(null >= observed - 1e-12))


def bootstrap_mean_ci(
    values: np.ndarray, resamples: int, rng: np.random.Generator
) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [math.nan, math.nan]
    sampled = values[rng.integers(0, len(values), size=(resamples, len(values)))]
    means = np.mean(sampled, axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def bh_qvalues(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if not np.any(finite):
        return q.tolist()
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(ranked), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    q[np.flatnonzero(finite)] = restored
    return q.tolist()


def paired_effect(
    abundance: pd.DataFrame,
    feature_id: str,
    tumour: str,
    normal: str,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame]:
    feature = abundance.loc[abundance["feature_id"].eq(feature_id)].copy()
    detected_areas = feature.loc[feature["detected"], "area"].to_numpy(float)
    floor = 0.5 * float(np.min(detected_areas)) if len(detected_areas) else 1.0
    tumour_frame = feature.loc[feature["tissue"].eq(tumour)].rename(
        columns={"area": "tumour_area", "detected": "tumour_detected"}
    )[["patient", "tumour_area", "tumour_detected"]]
    normal_frame = feature.loc[feature["tissue"].eq(normal)].rename(
        columns={"area": "normal_area", "detected": "normal_detected"}
    )[["patient", "normal_area", "normal_detected"]]
    paired = tumour_frame.merge(normal_frame, on="patient", validate="one_to_one")
    paired["complete_detection"] = paired["tumour_detected"] & paired["normal_detected"]
    paired["paired_log2_delta_floor"] = np.log2(
        (paired["tumour_area"] + floor) / (paired["normal_area"] + floor)
    )
    complete = paired.loc[paired["complete_detection"]].copy()
    complete["paired_log2_delta"] = np.log2(
        complete["tumour_area"] / complete["normal_area"]
    )
    primary = complete["paired_log2_delta"].to_numpy(float)
    sensitivity = paired["paired_log2_delta_floor"].to_numpy(float)
    report = {
        "tumour": tumour,
        "normal": normal,
        "pairs_total": int(len(paired)),
        "pairs_complete_detection": int(len(complete)),
        "positive_complete_pairs": int(np.sum(primary > 0)),
        "mean_complete_log2_delta": float(np.mean(primary)) if len(primary) else math.nan,
        "median_complete_log2_delta": float(np.median(primary)) if len(primary) else math.nan,
        "complete_mean_bootstrap_95ci": bootstrap_mean_ci(
            primary, bootstrap_resamples, rng
        ),
        "complete_exact_sign_flip_p": exact_sign_flip_p(primary),
        "floor": floor,
        "positive_floor_pairs": int(np.sum(sensitivity > 0)),
        "mean_floor_log2_delta": float(np.mean(sensitivity)) if len(sensitivity) else math.nan,
        "floor_exact_sign_flip_p": exact_sign_flip_p(sensitivity),
    }
    paired.insert(0, "feature_id", feature_id)
    paired.insert(2, "contrast", f"{tumour}_vs_{normal}")
    return report, paired


def exact_two_group_permutation_p(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    combined = np.concatenate([first, second])
    n_first = len(first)
    observed = abs(float(np.mean(first) - np.mean(second)))
    exceed = 0
    total = 0
    for chosen in itertools.combinations(range(len(combined)), n_first):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(chosen)] = True
        difference = abs(float(np.mean(combined[mask]) - np.mean(combined[~mask])))
        exceed += int(difference >= observed - 1e-12)
        total += 1
    return float(exceed / total)


def fragment_metrics(
    mzs: np.ndarray, intensities: np.ndarray, target: float, tolerance: float
) -> tuple[bool, float]:
    lo = int(np.searchsorted(mzs, target - tolerance, side="left"))
    hi = int(np.searchsorted(mzs, target + tolerance, side="right"))
    if hi <= lo or not len(intensities):
        return False, 0.0
    relative = float(np.max(intensities[lo:hi]) / max(float(np.max(intensities)), 1.0))
    return bool(relative >= 0.01), relative


def sanitize_json(value):
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def audit_ms2(
    mzml_dir: Path,
    features: pd.DataFrame,
    target_mz: float,
    ppm: float,
    fragment_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragment_targets = {
        "frag_87_0088": 87.0088,
        "frag_128_0350": 128.0350,
        "frag_170_0459": 170.0459,
        "frag_308_0987": 308.0987,
        "frag_332_0987": 332.0987,
    }
    tolerance = target_mz * ppm * 1e-6
    rows: list[dict] = []
    for path in sorted(mzml_dir.glob("*.hdf5")):
        sample = path.stem
        patient, tissue = parse_sample(sample)
        with h5py.File(path, "r") as handle:
            precursor = handle["precursor_mz"][:]
            active = np.flatnonzero(np.abs(precursor - target_mz) <= tolerance)
            for index in active:
                rt = float(handle["RT"][index])
                distance = np.abs(features["median_rt_sec"].to_numpy(float) - rt)
                nearest = int(np.argmin(distance))
                if distance[nearest] > 14.0:
                    continue
                spectrum = np.asarray(handle["spectrum"][index], dtype=float)
                valid = spectrum[0] > 0
                mzs = spectrum[0, valid]
                intensities = spectrum[1, valid]
                order = np.argsort(mzs)
                mzs = mzs[order]
                intensities = intensities[order]
                row = {
                    "feature_id": str(features.iloc[nearest]["feature_id"]),
                    "sample": sample,
                    "patient": patient,
                    "tissue": tissue,
                    "row": int(index),
                    "rt_sec": rt,
                    "precursor_mz": float(precursor[index]),
                    "precursor_ppm": float((precursor[index] - target_mz) / target_mz * 1e6),
                    "base_peak_mz": float(mzs[int(np.argmax(intensities))]),
                }
                for name, fragment in fragment_targets.items():
                    present, relative = fragment_metrics(
                        mzs, intensities, fragment, fragment_tolerance
                    )
                    row[f"{name}_present"] = present
                    row[f"{name}_relative"] = relative
                rows.append(row)
    spectra = pd.DataFrame(rows)
    summaries: list[dict] = []
    for feature_id in features["feature_id"]:
        subset = spectra.loc[spectra["feature_id"].eq(feature_id)]
        summary = {
            "feature_id": feature_id,
            "ms2_spectra": int(len(subset)),
            "ms2_samples": int(subset["sample"].nunique()) if len(subset) else 0,
            "ms2_patients": int(subset["patient"].nunique()) if len(subset) else 0,
        }
        for name in fragment_targets:
            column = f"{name}_present"
            summary[f"{name}_spectra"] = int(subset[column].sum()) if len(subset) else 0
            summary[f"{name}_samples"] = (
                int(subset.loc[subset[column], "sample"].nunique()) if len(subset) else 0
            )
        summaries.append(summary)
    return spectra, pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    paths = sorted(args.mzml_dir.glob("*.mzML"))
    expected = 60 if not args.max_samples else args.max_samples
    if args.max_samples:
        paths = paths[: args.max_samples]
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} negative-HILIC mzML files, found {len(paths)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    peak_rows: list[dict] = []
    for index, path in enumerate(paths, 1):
        parse_sample(path.stem)
        times, signal = load_ms1_trace(path, args.mz, args.ppm)
        traces[path.stem] = (times, signal)
        peak_rows.extend(discover_sample_peaks(path.stem, times, signal))
        print(f"[OAc MS1] {index}/{len(paths)} {path.stem}", flush=True)
    peak_calls, features, consensus_trace = discover_population_features(
        traces,
        pd.DataFrame(peak_rows),
        args.rt_cluster_diameter_sec,
        args.minimum_cluster_samples,
        args.minimum_feature_separation_sec,
    )
    abundance_rows: list[dict] = []
    for feature in features.itertuples(index=False):
        for sample, (times, signal) in traces.items():
            patient, tissue = parse_sample(sample)
            metrics = quantify_feature(
                times,
                signal,
                float(feature.median_rt_sec),
                args.integration_half_window_sec,
                args.maximum_apex_delta_sec,
            )
            abundance_rows.append(
                {
                    "feature_id": feature.feature_id,
                    "feature_rt_sec": float(feature.median_rt_sec),
                    "sample": sample,
                    "patient": patient,
                    "tissue": tissue,
                    **metrics,
                }
            )
    abundance = pd.DataFrame(abundance_rows)
    ms2_spectra, ms2_summary = audit_ms2(
        args.mzml_dir,
        features,
        args.mz,
        args.ppm,
        args.fragment_tolerance_da,
    )
    features = features.merge(ms2_summary, on="feature_id", how="left", validate="one_to_one")
    rng = np.random.default_rng(args.seed)
    effect_rows: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    for feature_id in features["feature_id"]:
        for tumour, normal in [("Rmu", "RN"), ("Rtu", "RN"), ("Ltu", "LN")]:
            effect, pairs = paired_effect(
                abundance,
                feature_id,
                tumour,
                normal,
                args.bootstrap_resamples,
                rng,
            )
            effect_rows.append({"feature_id": feature_id, **effect})
            pair_frames.append(pairs)
    effects = pd.DataFrame(effect_rows)
    rmu_mask = effects["tumour"].eq("Rmu")
    effects.loc[rmu_mask, "complete_exact_sign_flip_bh_q"] = bh_qvalues(
        effects.loc[rmu_mask, "complete_exact_sign_flip_p"].tolist()
    )
    pairs = pd.concat(pair_frames, ignore_index=True)

    subtype_rows = []
    donor_correlations = []
    donor = pd.read_csv(args.donor_deltas)
    free = donor.loc[donor["node"].eq("free_neu5ac"), ["patient", "paired_log2_delta"]].rename(
        columns={"paired_log2_delta": "free_neu5ac_delta"}
    )
    for feature_id in features["feature_id"]:
        rmu = pairs.loc[
            pairs["feature_id"].eq(feature_id) & pairs["contrast"].eq("Rmu_vs_RN")
        ]
        rtu = pairs.loc[
            pairs["feature_id"].eq(feature_id) & pairs["contrast"].eq("Rtu_vs_RN")
        ]
        rmu_delta = rmu["paired_log2_delta_floor"].to_numpy(float)
        rtu_delta = rtu["paired_log2_delta_floor"].to_numpy(float)
        subtype_rows.append(
            {
                "feature_id": feature_id,
                "rmu_mean_floor_delta": float(np.mean(rmu_delta)),
                "rtu_mean_floor_delta": float(np.mean(rtu_delta)),
                "rmu_minus_rtu": float(np.mean(rmu_delta) - np.mean(rtu_delta)),
                "exact_group_permutation_p": exact_two_group_permutation_p(
                    rmu_delta, rtu_delta
                ),
            }
        )
        merged = rmu[["patient", "paired_log2_delta_floor"]].merge(
            free, on="patient", validate="one_to_one"
        )
        correlation = stats.spearmanr(
            merged["paired_log2_delta_floor"], merged["free_neu5ac_delta"]
        )
        donor_correlations.append(
            {
                "feature_id": feature_id,
                "n": int(len(merged)),
                "spearman_rho": float(correlation.statistic),
                "spearman_p": float(correlation.pvalue),
            }
        )
    subtype = pd.DataFrame(subtype_rows)
    subtype["exact_group_permutation_bh_q"] = bh_qvalues(
        subtype["exact_group_permutation_p"].tolist()
    )
    donor_correlations_frame = pd.DataFrame(donor_correlations)

    peak_calls.to_csv(args.output_dir / "phenotype_blind_peak_calls.csv", index=False)
    consensus_trace.to_csv(args.output_dir / "phenotype_blind_consensus_trace.csv.gz", index=False)
    features.to_csv(args.output_dir / "frozen_rt_features.csv", index=False)
    abundance.to_csv(args.output_dir / "per_sample_eic.csv.gz", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    pairs.to_csv(args.output_dir / "paired_patient_values.csv", index=False)
    subtype.to_csv(args.output_dir / "subtype_interactions.csv", index=False)
    donor_correlations_frame.to_csv(
        args.output_dir / "free_neu5ac_correlations.csv", index=False
    )
    ms2_spectra.to_csv(args.output_dir / "rt_resolved_ms2_spectra.csv.gz", index=False)

    primary = effects.loc[effects["tumour"].eq("Rmu")].merge(
        features, on="feature_id", validate="one_to_one"
    )
    report = {
        "status": "mtbls13729_oacetyl_neu5ac_like_audit_complete",
        "formal": False,
        "target": {
            "formula": FORMULA,
            "mz": args.mz,
            "adduct": "[M-H]-",
            "compatible_isomers": [
                "4-O-acetyl-Neu5Ac",
                "7-O-acetyl-Neu5Ac",
                "8-O-acetyl-Neu5Ac",
                "9-O-acetyl-Neu5Ac",
            ],
        },
        "phenotype_blind_discovery": {
            "samples": len(paths),
            "peak_calls": int(len(peak_calls)),
            "frozen_rt_features": int(len(features)),
            "minimum_cluster_samples": args.minimum_cluster_samples,
            "rt_cluster_diameter_sec": args.rt_cluster_diameter_sec,
        },
        "primary_rmu_results": primary.replace({np.nan: None}).to_dict("records"),
        "subtype_interactions": subtype.replace({np.nan: None}).to_dict("records"),
        "free_neu5ac_correlations": donor_correlations_frame.replace({np.nan: None}).to_dict("records"),
        "evidence_contract": {
            "rt_features_selected_without_phenotype": True,
            "primary_complete_pair_test": "two-sided exact sign-flip of mean paired log2 delta",
            "rmu_multiplicity": "BH across all phenotype-blind frozen RT features",
            "missingness_sensitivity": "half-minimum-detected-area floor, reported separately",
            "ms2": "RT-resolved motif counts; motifs do not locate the O-acetyl position",
        },
        "claim_limit": (
            "This is an exact-mass and RT-resolved discovery audit in the MTBLS13729 cohort. "
            "It cannot identify an O-acetyl positional isomer, establish a glycan carrier, "
            "replicate the finding in an independent cohort, or infer O-acetylation flux."
        ),
        "provenance": {
            "mzml_directory": str(args.mzml_dir.resolve()),
            "donor_deltas_sha256": sha256(args.donor_deltas),
            "script_sha256": sha256(Path(__file__)),
        },
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"mzml_dir", "donor_deltas", "output_dir"}
        },
    }
    report = sanitize_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
