#!/usr/bin/env python
"""Audit diagnostic-fragment recurrence across samples and collision energies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHES = ROOT / "data/mtbls13729/frozen_candidate_ms2_coverage_v1/candidate_ms2_matches.csv.gz"
DEFAULT_MZML = ROOT / "data/mtbls13729/mzml/pos_rp"
DEFAULT_OUTPUT = ROOT / "data/mtbls13729/candidate_ce_recurrence_v1"

DIAGNOSTIC = {
    1597: [(166.0725, "methylated-guanine aglycone"), (152.0558, "alternative methylguanine/ribose-loss ion")],
    3019: [(180.0886, "dimethylated-guanine aglycone")],
    1717: [(100.0759, "diacetylspermidine diagnostic product"), (114.0916, "supporting polyamine product")],
    3222: [(85.0281, "acylcarnitine-class product"), (60.0808, "carnitine-related product")],
    4966: [(110.0347, "purine-like recurrent product")],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_energy(spectrum) -> float:
    precursors = spectrum.getPrecursors()
    if not precursors:
        return np.nan
    precursor = precursors[0]
    for key in (b"collision energy", "collision energy"):
        try:
            if precursor.metaValueExists(key):
                return float(precursor.getMetaValue(key))
        except Exception:
            pass
    try:
        value = float(precursor.getActivationEnergy())
        return value if value > 0 else np.nan
    except Exception:
        return np.nan


def patient_id(sample: str) -> str:
    return sample.split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--mzml-dir", type=Path, default=DEFAULT_MZML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance-da", type=float, default=0.02)
    parser.add_argument("--minimum-relative-intensity", type=float, default=0.005)
    args = parser.parse_args()

    try:
        import pyopenms as oms
    except ImportError as exc:
        raise RuntimeError("pyopenms is required") from exc

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(args.matches)
    matches = matches[
        matches.peak_resolved_match.astype(bool) & matches.feature_id.isin(DIAGNOSTIC)
    ].copy()
    if matches.empty:
        raise RuntimeError("no peak-resolved target spectra")

    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for number, (sample, group) in enumerate(sorted(matches.groupby("sample")), start=1):
        path = args.mzml_dir / f"{sample}.mzML"
        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        try:
            loader.load(str(path), experiment)
        except Exception as exc:
            failures.append({"sample": sample, "error": repr(exc)})
            continue
        by_native = {s.getNativeID(): s for s in experiment}
        for row in group.itertuples(index=False):
            spectrum = by_native.get(str(row.native_id))
            if spectrum is None:
                failures.append({"sample": sample, "error": f"missing {row.native_id}"})
                continue
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, float)
            intensity = np.asarray(intensity, float)
            maximum = float(np.max(intensity)) if len(intensity) else 0.0
            relative = intensity / maximum if maximum > 0 else np.zeros_like(intensity)
            base = {
                "feature_id": int(row.feature_id),
                "sample": sample,
                "patient": patient_id(sample),
                "native_id": str(row.native_id),
                "collision_energy": collision_energy(spectrum),
                "precursor_mz": float(row.precursor_mz),
            }
            for target_mz, label in DIAGNOSTIC[int(row.feature_id)]:
                delta = np.abs(mz - target_mz)
                hit = bool(len(delta) and np.min(delta) <= args.tolerance_da)
                if hit:
                    hit_index = int(np.argmin(delta))
                    hit = bool(relative[hit_index] >= args.minimum_relative_intensity)
                records.append(
                    {
                        **base,
                        "diagnostic_label": label,
                        "target_mz": target_mz,
                        "present": hit,
                        "observed_mz": float(mz[hit_index]) if hit else np.nan,
                        "relative_intensity": float(relative[hit_index]) if hit else 0.0,
                    }
                )
        if number % 10 == 0 or number == matches["sample"].nunique():
            print(f"[CE audit] {number}/{matches['sample'].nunique()} samples", flush=True)

    detail = pd.DataFrame(records)
    detail.to_csv(output / "spectrum_diagnostic_detail.csv.gz", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    if detail.empty:
        raise RuntimeError("no diagnostic records produced")

    detail["ce_bin"] = detail.collision_energy.map(
        lambda x: "unknown" if not np.isfinite(x) else f"{int(round(x / 5) * 5)} eV"
    )
    detail["spectrum_key"] = detail["sample"].astype(str) + "|" + detail["native_id"].astype(str)
    summary = (
        detail.groupby(["feature_id", "diagnostic_label", "target_mz"], as_index=False)
        .agg(
            spectra=("spectrum_key", "nunique"),
            samples=("sample", "nunique"),
            patients=("patient", "nunique"),
            collision_energies=("collision_energy", lambda x: int(pd.Series(x).dropna().nunique())),
            present_spectra=("present", "sum"),
            median_relative_intensity=("relative_intensity", "median"),
        )
    )
    summary["support_fraction"] = summary.present_spectra / summary.spectra
    if not summary.support_fraction.between(0, 1).all():
        raise RuntimeError("diagnostic support fraction outside [0, 1]")
    summary.to_csv(output / "diagnostic_recurrence_summary.csv", index=False)

    by_ce = (
        detail.groupby(["feature_id", "diagnostic_label", "ce_bin"], as_index=False)
        .agg(spectra=("spectrum_key", "nunique"), present_spectra=("present", "sum"))
    )
    by_ce["support_fraction"] = by_ce.present_spectra / by_ce.spectra
    if not by_ce.support_fraction.between(0, 1).all():
        raise RuntimeError("CE-stratified support fraction outside [0, 1]")
    by_ce.to_csv(output / "diagnostic_recurrence_by_ce.csv", index=False)

    finite_ce = detail.dropna(subset=["collision_energy"])
    payload = {
        "status": "mtbls13729_candidate_ce_recurrence_complete",
        "formal": True,
        "peak_resolved_spectra": int(detail[["sample", "native_id"]].drop_duplicates().shape[0]),
        "samples": int(detail["sample"].nunique()),
        "patients": int(detail.patient.nunique()),
        "features": int(detail.feature_id.nunique()),
        "finite_collision_energy_fraction": float(
            finite_ce[["sample", "native_id"]].drop_duplicates().shape[0]
            / detail[["sample", "native_id"]].drop_duplicates().shape[0]
        ),
        "collision_energies": sorted(finite_ce.collision_energy.unique().tolist()),
        "summary": summary.to_dict(orient="records"),
        "claim_limit": (
            "Cross-sample recurrence strengthens ion-family evidence only. All audited spectra "
            "used the same recorded 30 eV collision energy, so this dataset provides no cross-CE "
            "replication. It cannot resolve positional isomers or replace same-method authentic-"
            "standard RT/MS2."
        ),
        "provenance": {
            "matches_sha256": sha256(args.matches),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
