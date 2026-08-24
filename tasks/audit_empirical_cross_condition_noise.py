"""Audit real MS/MS variation before choosing synthetic noise parameters.

The noise fine-tuning project originally used arbitrary deletion, intensity
jitter and added-peak ranges.  This script instead measures those quantities
on *real* same-identity, same-adduct spectra acquired under different
instrument or collision-energy conditions.  It makes no model forward pass.

The report is a calibration aid, not an efficacy result: it tells us what
synthetic perturbations are physically plausible for this data source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = ROOT / "data/validation/cross_condition_m3/train_pairs.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUT = ROOT / "data/validation/cross_condition_m3/empirical_noise_audit.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--mz-tolerance-da", type=float, default=0.02)
    p.add_argument("--high-intensity-threshold", type=float, default=0.2,
                   help="Relative-to-base-peak threshold used for shared main peaks.")
    return p.parse_args()


def valid_peaks(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = np.asarray(raw[0], float), np.asarray(raw[1], float)
    keep = (mz > 0) & np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    if len(intensity) and intensity.max() > 0:
        intensity = intensity / intensity.max()
    return mz, intensity


def one_to_one_matches(a_mz: np.ndarray, b_mz: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy monotonic peak matching; sufficient for an audit at 0.02 Da."""
    matches: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a_mz) and j < len(b_mz):
        delta = a_mz[i] - b_mz[j]
        if abs(delta) <= tol:
            matches.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches


def quantiles(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = manifest["cross_pairs"][: args.max_pairs]
    if not pairs:
        raise RuntimeError("No cross-condition pairs in manifest")

    shared_a: list[float] = []
    shared_b: list[float] = []
    high_shared_a: list[float] = []
    high_shared_b: list[float] = []
    unmatched_a: list[float] = []
    unmatched_b: list[float] = []
    intensity_ratio_abslog: list[float] = []
    mz_abs_delta: list[float] = []
    n_peaks_a: list[float] = []
    n_peaks_b: list[float] = []

    with h5py.File(args.data, "r") as handle:
        spectra = handle["spectrum"]
        for index, pair in enumerate(pairs, start=1):
            left, right = pair["rows"]
            a_mz, a_i = valid_peaks(spectra[left])
            b_mz, b_i = valid_peaks(spectra[right])
            matches = one_to_one_matches(a_mz, b_mz, args.mz_tolerance_da)
            ma = np.asarray([i for i, _ in matches], dtype=int)
            mb = np.asarray([j for _, j in matches], dtype=int)
            n_peaks_a.append(float(len(a_mz)))
            n_peaks_b.append(float(len(b_mz)))
            shared_a.append(len(matches) / max(1, len(a_mz)))
            shared_b.append(len(matches) / max(1, len(b_mz)))
            unmatched_a.append(1.0 - shared_a[-1])
            unmatched_b.append(1.0 - shared_b[-1])
            high_a = np.flatnonzero(a_i >= args.high_intensity_threshold)
            high_b = np.flatnonzero(b_i >= args.high_intensity_threshold)
            high_shared_a.append(len(np.intersect1d(ma, high_a)) / max(1, len(high_a)))
            high_shared_b.append(len(np.intersect1d(mb, high_b)) / max(1, len(high_b)))
            if len(matches):
                ratio = np.clip(a_i[ma] / np.maximum(b_i[mb], 1e-8), 1e-8, 1e8)
                intensity_ratio_abslog.extend(np.abs(np.log(ratio)).tolist())
                mz_abs_delta.extend(np.abs(a_mz[ma] - b_mz[mb]).tolist())
            if index % 500 == 0:
                print(f"Audited {index:,}/{len(pairs):,} real cross-condition pairs", flush=True)

    report = {
        "status": "empirical_cross_condition_noise_audit",
        "pair_definition": manifest.get("pair_definition"),
        "n_pairs": len(pairs),
        "mz_tolerance_da": args.mz_tolerance_da,
        "high_intensity_threshold_relative_to_base_peak": args.high_intensity_threshold,
        "peak_count_left": quantiles(n_peaks_a),
        "peak_count_right": quantiles(n_peaks_b),
        "matched_peak_fraction_left": quantiles(shared_a),
        "matched_peak_fraction_right": quantiles(shared_b),
        "unmatched_peak_fraction_left": quantiles(unmatched_a),
        "unmatched_peak_fraction_right": quantiles(unmatched_b),
        "shared_high_intensity_fraction_left": quantiles(high_shared_a),
        "shared_high_intensity_fraction_right": quantiles(high_shared_b),
        "matched_peak_abs_log_intensity_ratio": quantiles(intensity_ratio_abslog),
        "matched_peak_abs_mz_delta_da": quantiles(mz_abs_delta),
        "interpretation": [
            "Use the empirical unmatched-peak and intensity-ratio quantiles to bound synthetic masking and jitter.",
            "Do not model individual m/z displacement unless the matched m/z audit shows instrument-scale displacement beyond tolerance.",
            "This does not establish that synthetic augmentation improves retrieval; it only prevents an unphysical augmentation range.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
