#!/usr/bin/env python
"""Audit diagnostic MS2 evidence for the frozen C20:4 acylcarnitine anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIAGNOSTIC_FRAGMENTS = (60.0808, 85.0284, 144.1019)
NEUTRAL_LOSS = 59.0735


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_peak(
    mz: np.ndarray,
    intensity: np.ndarray,
    target: float,
    tolerance_da: float,
) -> tuple[bool, float, float]:
    if not len(mz) or not np.isfinite(intensity).any() or np.nanmax(intensity) <= 0:
        return False, np.nan, 0.0
    indices = np.flatnonzero(np.abs(mz - target) <= tolerance_da)
    if not len(indices):
        return False, np.nan, 0.0
    best = indices[np.argmax(intensity[indices])]
    relative = float(intensity[best] / np.nanmax(intensity))
    return True, float(mz[best]), relative


def spectrum_evidence(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    tolerance_da: float,
    minimum_relative_intensity: float,
) -> dict[str, object]:
    evidence: dict[str, object] = {}
    diagnostic_present = []
    for target in DIAGNOSTIC_FRAGMENTS:
        found, observed, relative = relative_peak(mz, intensity, target, tolerance_da)
        present = bool(found and relative >= minimum_relative_intensity)
        diagnostic_present.append(present)
        key = f"fragment_{target:.4f}".replace(".", "_")
        evidence[f"{key}_present"] = present
        evidence[f"{key}_observed_mz"] = observed
        evidence[f"{key}_relative_intensity"] = relative
    loss_target = precursor_mz - NEUTRAL_LOSS
    found, observed, relative = relative_peak(mz, intensity, loss_target, tolerance_da)
    loss_present = bool(found and relative >= minimum_relative_intensity)
    evidence["neutral_loss_59_0735_present"] = loss_present
    evidence["neutral_loss_59_0735_observed_mz"] = observed
    evidence["neutral_loss_59_0735_relative_intensity"] = relative
    evidence["diagnostic_motif_count"] = int(sum(diagnostic_present) + loss_present)
    evidence["strong_carnitine_motif"] = bool(
        evidence["diagnostic_motif_count"] >= 3
        and diagnostic_present[1]
        and (diagnostic_present[0] or diagnostic_present[2])
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_rp"))
    parser.add_argument("--target-mz", type=float, default=448.33946255)
    parser.add_argument("--target-rt-sec", type=float, default=631.066)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--rt-window-sec", type=float, default=15.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--minimum-relative-intensity", type=float, default=0.005)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/c20_4_anchor_ms2_audit_v1"),
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
    paths = sorted(args.mzml_dir.glob("*.mzML"))
    if not paths:
        raise FileNotFoundError(f"no mzML files under {args.mzml_dir}")

    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for number, path in enumerate(paths, start=1):
        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        try:
            loader.load(str(path), experiment)
        except Exception as exc:
            failed.append({"sample": path.stem, "error": repr(exc)})
            continue
        for spectrum in experiment:
            precursors = spectrum.getPrecursors()
            if not precursors:
                continue
            precursor_mz = float(precursors[0].getMZ())
            ppm_error = (precursor_mz - args.target_mz) / args.target_mz * 1e6
            rt_delta = float(spectrum.getRT() - args.target_rt_sec)
            if abs(ppm_error) > args.precursor_ppm or abs(rt_delta) > args.rt_window_sec:
                continue
            mz, intensity = spectrum.get_peaks()
            evidence = spectrum_evidence(
                np.asarray(mz, dtype=float),
                np.asarray(intensity, dtype=float),
                precursor_mz,
                args.fragment_tolerance_da,
                args.minimum_relative_intensity,
            )
            rows.append({
                "sample": path.stem,
                "native_id": spectrum.getNativeID(),
                "rt_sec": float(spectrum.getRT()),
                "precursor_mz": precursor_mz,
                "precursor_ppm_error": float(ppm_error),
                "rt_delta_sec": rt_delta,
                "n_peaks": int(len(mz)),
                **evidence,
            })
        if number % 10 == 0 or number == len(paths):
            print(f"[C20:4 MS2] {number}/{len(paths)} files", flush=True)

    spectra = pd.DataFrame(rows)
    spectra.to_csv(output / "matching_ms2_spectra.csv.gz", index=False)
    pd.DataFrame(failed).to_csv(output / "failed_mzml_files.csv", index=False)
    if spectra.empty:
        raise RuntimeError("no MS2 spectrum matched the frozen C20:4 precursor and RT window")
    per_sample = (
        spectra.groupby("sample", as_index=False)
        .agg(
            matching_ms2=("native_id", "nunique"),
            strongest_motif_count=("diagnostic_motif_count", "max"),
            strong_carnitine_motif=("strong_carnitine_motif", "max"),
            best_abs_precursor_ppm=("precursor_ppm_error", lambda values: float(np.min(np.abs(values)))),
            best_abs_rt_delta_sec=("rt_delta_sec", lambda values: float(np.min(np.abs(values)))),
        )
    )
    per_sample.to_csv(output / "per_sample_evidence.csv", index=False)
    payload = {
        "status": "mtbls13729_c20_4_ms2_audit_complete",
        "target": {"feature_id": 3222, "mz": args.target_mz, "rt_sec": args.target_rt_sec},
        "mzml_files": int(len(paths)),
        "failed_mzml_files": int(len(failed)),
        "matching_ms2_spectra": int(len(spectra)),
        "samples_with_matching_ms2": int(spectra["sample"].nunique()),
        "strong_motif_spectra": int(spectra["strong_carnitine_motif"].sum()),
        "samples_with_strong_motif": int(
            per_sample["strong_carnitine_motif"].astype(bool).sum()
        ),
        "parameters": {
            "precursor_ppm": args.precursor_ppm,
            "rt_window_sec": args.rt_window_sec,
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "minimum_relative_intensity": args.minimum_relative_intensity,
        },
        "provenance": {
            "mzml_directory": str(args.mzml_dir.resolve()),
            "mzml_file_hashes": {path.name: sha256(path) for path in paths},
        },
        "claim_limit": (
            "Diagnostic fragments support an acylcarnitine class assignment. They do not establish "
            "double-bond position, stereochemistry, chromatographic identity, or MSI Level 1 identity."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
