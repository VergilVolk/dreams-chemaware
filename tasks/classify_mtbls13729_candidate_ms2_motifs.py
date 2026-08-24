"""Assign conservative class-level MS/MS motifs to MTBLS13729 candidates.

These rules support chemical *class* evidence only.  They never assign an
isomer or an MSI Level 1 identity.  The initial rule set is intentionally
small and literature anchored rather than an unrestricted rule dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


AC_FRAGMENT_MZ = {
    "carnitine_60.0808": 60.0808,
    "carnitine_85.0284": 85.0284,
    "carnitine_144.1019": 144.1019,
}
AC_NEUTRAL_LOSS = 59.0735
PHOSPHOCHOLINE_MZ = 184.0733


def peak_match(mz: np.ndarray, intensity: np.ndarray, target: float, tolerance: float) -> tuple[bool, float, float]:
    eligible = np.flatnonzero(np.abs(mz - target) <= tolerance)
    if not len(eligible):
        return False, np.nan, 0.0
    index = eligible[np.argmax(intensity[eligible])]
    relative = float(intensity[index] / np.max(intensity)) if np.max(intensity) > 0 else 0.0
    return True, float(mz[index]), relative


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--links", type=Path, default=Path("data/mtbls13729/biology_candidates_peakresolved/candidate_ms2_links.csv.gz"))
    p.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    p.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    p.add_argument("--min-relative-intensity", type=float, default=0.005)
    p.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/biology_candidates_peakresolved"))
    args = p.parse_args()

    links = pd.read_csv(args.links)
    rows: list[dict] = []
    for (panel, sample), group in links.groupby(["panel", "sample_name"]):
        path = args.mzml_root / panel / f"{sample}.mzML"
        exp = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        loader.load(str(path), exp)
        spectra = {s.getNativeID(): s for s in exp}
        for link in group.itertuples(index=False):
            spectrum = spectra.get(link.native_id)
            if spectrum is None:
                continue
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            if not len(mz) or np.max(intensity) <= 0:
                continue
            motif_details: dict[str, dict[str, float | bool]] = {}
            ac_count = 0
            for name, target in AC_FRAGMENT_MZ.items():
                present, observed, relative = peak_match(mz, intensity, target, args.fragment_tolerance_da)
                accepted = bool(present and relative >= args.min_relative_intensity)
                motif_details[name] = {"present": accepted, "observed_mz": observed, "relative_intensity": relative}
                ac_count += int(accepted)
            nl_target = float(link.precursor_mz) - AC_NEUTRAL_LOSS
            present, observed, relative = peak_match(mz, intensity, nl_target, args.fragment_tolerance_da)
            nl_accepted = bool(present and relative >= args.min_relative_intensity)
            motif_details["neutral_loss_59.0735"] = {
                "present": nl_accepted,
                "observed_mz": observed,
                "relative_intensity": relative,
            }
            ac_count += int(nl_accepted)
            pc_present, pc_observed, pc_relative = peak_match(
                mz, intensity, PHOSPHOCHOLINE_MZ, args.fragment_tolerance_da
            )
            pc_accepted = bool(pc_present and pc_relative >= args.min_relative_intensity)
            rows.append({
                "panel": panel,
                "feature_id": int(link.feature_id),
                "sample_name": sample,
                "native_id": link.native_id,
                "precursor_mz": float(link.precursor_mz),
                "ms2_rt_sec": float(link.ms2_rt_sec),
                "acylcarnitine_motif_count": ac_count,
                "acylcarnitine_class_supported": bool(
                    ac_count >= 3
                    and motif_details["carnitine_85.0284"]["present"]
                    and (
                        motif_details["carnitine_60.0808"]["present"]
                        or motif_details["carnitine_144.1019"]["present"]
                    )
                ),
                "phosphocholine_184_supported": pc_accepted,
                "phosphocholine_observed_mz": pc_observed,
                "phosphocholine_relative_intensity": pc_relative,
                "motif_details_json": json.dumps(motif_details, separators=(",", ":")),
            })

    evidence = pd.DataFrame(rows)
    summary = (
        evidence.groupby(["panel", "feature_id"], as_index=False)
        .agg(
            n_ms2_spectra=("native_id", "nunique"),
            n_ms2_samples=("sample_name", "nunique"),
            best_acylcarnitine_motif_count=("acylcarnitine_motif_count", "max"),
            n_acylcarnitine_support_spectra=("acylcarnitine_class_supported", "sum"),
            n_phosphocholine_support_spectra=("phosphocholine_184_supported", "sum"),
        )
    )
    ac_samples = (
        evidence.loc[evidence.acylcarnitine_class_supported]
        .groupby(["panel", "feature_id"])["sample_name"].nunique()
        .rename("n_acylcarnitine_support_samples")
    )
    pc_samples = (
        evidence.loc[evidence.phosphocholine_184_supported]
        .groupby(["panel", "feature_id"])["sample_name"].nunique()
        .rename("n_phosphocholine_support_samples")
    )
    summary = summary.join(ac_samples, on=["panel", "feature_id"]).join(pc_samples, on=["panel", "feature_id"])
    for column in ["n_acylcarnitine_support_samples", "n_phosphocholine_support_samples"]:
        summary[column] = summary[column].fillna(0).astype(int)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence_path = out / "candidate_ms2_class_motif_evidence.csv.gz"
    summary_path = out / "candidate_ms2_class_motif_summary.csv"
    evidence.to_csv(evidence_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "status": "complete",
        "n_features": int(len(summary)),
        "n_acylcarnitine_class_supported_in_2plus_samples": int((summary.n_acylcarnitine_support_samples >= 2).sum()),
        "n_phosphocholine_supported_in_2plus_samples": int((summary.n_phosphocholine_support_samples >= 2).sum()),
        "evidence": str(evidence_path),
        "summary": str(summary_path),
        "interpretation_limit": "Diagnostic fragments support a chemical class, not a unique molecular structure or isomer.",
        "literature": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10859589/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7657438/",
            "https://pubmed.ncbi.nlm.nih.gov/27562752/",
        ],
    }
    (out / "candidate_ms2_class_motif_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
