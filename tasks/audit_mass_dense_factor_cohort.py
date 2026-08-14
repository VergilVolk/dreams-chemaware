"""Audit a mass-dense cohort for rule-free DreaMS factor validation.

Each retained molecule/adduct unit must provide:
1. a same-molecule, same-adduct positive pair with an instrument change and/or
   at least 10 eV collision-energy difference; and
2. at least one different molecule with the same adduct within 10 ppm.

The script only builds a manifest.  It does not run either DreaMS checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_factor_cohort_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", default="val")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--min-ce-difference", type=float, default=10.0)
    parser.add_argument("--max-neighbors", type=int, default=20)
    return parser.parse_args()


def decode_array(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def choose_positive_pair(group: pd.DataFrame, min_ce_difference: float):
    rows = group.index.to_numpy(dtype=np.int64)
    instruments = group["instrument"].to_numpy()
    energies = group["collision_energy"].to_numpy(dtype=float)
    best = None
    for left in range(len(group) - 1):
        for right in range(left + 1, len(group)):
            instrument_diff = instruments[left] != instruments[right]
            ce_finite = np.isfinite(energies[left]) and np.isfinite(energies[right])
            ce_difference = (
                abs(float(energies[left] - energies[right])) if ce_finite else 0.0
            )
            ce_diff = ce_difference >= min_ce_difference
            if not (instrument_diff or ce_diff):
                continue
            score = (int(instrument_diff) + int(ce_diff), ce_difference)
            if best is None or score > best[0]:
                best = (
                    score,
                    [int(rows[left]), int(rows[right])],
                    bool(instrument_diff),
                    bool(ce_diff),
                    ce_difference,
                )
    return best


def main() -> None:
    args = parse_args()
    with h5py.File(args.data, "r") as handle:
        folds = decode_array(handle["fold"][:])
        selected = np.flatnonzero(folds == args.fold)
        spectra = np.asarray(handle["spectrum"][selected])
        mzs = spectra[:, 0, :]
        intensities = spectra[:, 1, :]
        valid_peak = (
            np.isfinite(mzs) & np.isfinite(intensities)
            & (mzs > 0) & (intensities > 0)
        )
        peak_count = valid_peak.sum(axis=1)
        max_mz = np.where(valid_peak, mzs, -np.inf).max(axis=1)
        max_intensity = np.where(valid_peak, intensities, -np.inf).max(axis=1)
        min_intensity = np.where(valid_peak, intensities, np.inf).min(axis=1)
        amplitude = np.divide(
            max_intensity,
            min_intensity,
            out=np.zeros_like(max_intensity),
            where=np.isfinite(max_intensity) & np.isfinite(min_intensity)
            & (min_intensity > 0),
        )
        relative = np.divide(
            intensities,
            max_intensity[:, None],
            out=np.zeros_like(intensities),
            where=np.isfinite(max_intensity[:, None])
            & (max_intensity[:, None] > 0),
        )
        high_peak_count = ((relative > 0.1) & valid_peak).sum(axis=1)
        fold_precursor = np.asarray(handle["precursor_mz"][selected], dtype=float)
        precursor_valid = (
            np.isfinite(fold_precursor)
            & (fold_precursor > 0)
            & (fold_precursor <= 1000)
        )
        quality_mask = (
            precursor_valid
            & (peak_count >= 3)
            & (peak_count <= 128)
            & (max_mz <= 1000)
            & (amplitude >= 20)
            & (high_peak_count >= 3)
        )
        quality_audit = {
            "fold_spectra": int(len(selected)),
            "quality_valid_spectra": int(quality_mask.sum()),
            "quality_valid_fraction": float(quality_mask.mean()),
            "excluded_precursor": int((~precursor_valid).sum()),
            "excluded_peak_count": int(((peak_count < 3) | (peak_count > 128)).sum()),
            "excluded_max_mz": int((max_mz > 1000).sum()),
            "excluded_intensity_amplitude": int((amplitude < 20).sum()),
            "excluded_high_peak_count": int((high_peak_count < 3).sum()),
            "note": "Exclusion counts overlap; DataFormatA-like spectrum-level QC.",
        }
        selected = selected[quality_mask]
        inchikeys = decode_array(handle["INCHIKEY"][selected])
        frame = pd.DataFrame({
            "row": selected,
            "ik14": np.asarray([value[:14] for value in inchikeys]),
            "adduct": decode_array(handle["adduct"][selected]),
            "precursor_mz": np.asarray(handle["precursor_mz"][selected], dtype=float),
            "instrument": decode_array(handle["INSTRUMENT_TYPE"][selected]),
            "collision_energy": np.asarray(
                handle["COLLISION_ENERGY"][selected], dtype=float
            ),
        }).set_index("row", drop=True)
    frame = frame.loc[
        (frame["ik14"].str.len() == 14)
        & np.isfinite(frame["precursor_mz"])
        & (frame["precursor_mz"] > 0)
    ]

    units = []
    for (ik14, adduct), group in frame.groupby(["ik14", "adduct"], sort=False):
        if len(group) < 2:
            continue
        positive = choose_positive_pair(group, args.min_ce_difference)
        if positive is None:
            continue
        _, rows, instrument_diff, ce_diff, ce_difference = positive
        masses = frame.loc[rows, "precursor_mz"].to_numpy(dtype=float)
        units.append({
            "unit_id": len(units),
            "ik14": ik14,
            "adduct": adduct,
            "positive_rows": rows,
            "precursor_mz": float(np.mean(masses)),
            "positive_instrument_diff": instrument_diff,
            "positive_ce_diff_ge_threshold": ce_diff,
            "positive_ce_difference": ce_difference,
        })

    by_adduct: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        by_adduct.setdefault(unit["adduct"], []).append(index)
    directed_links = 0
    for adduct, indices in by_adduct.items():
        order = sorted(indices, key=lambda idx: units[idx]["precursor_mz"])
        masses = np.asarray([units[idx]["precursor_mz"] for idx in order])
        for position, unit_index in enumerate(order):
            mass = masses[position]
            tolerance = mass * args.ppm * 1e-6
            left = int(np.searchsorted(masses, mass - tolerance, side="left"))
            right = int(np.searchsorted(masses, mass + tolerance, side="right"))
            candidates = [
                order[j] for j in range(left, right)
                if order[j] != unit_index
                and units[order[j]]["ik14"] != units[unit_index]["ik14"]
            ]
            candidates.sort(
                key=lambda idx: abs(units[idx]["precursor_mz"] - mass)
            )
            candidates = candidates[: args.max_neighbors]
            units[unit_index]["negative_unit_ids"] = [
                int(units[idx]["unit_id"]) for idx in candidates
            ]
            units[unit_index]["nearest_negative_ppm"] = (
                float(
                    abs(units[candidates[0]]["precursor_mz"] - mass)
                    / mass
                    * 1e6
                )
                if candidates else None
            )
            directed_links += len(candidates)

    retained = [unit for unit in units if unit["negative_unit_ids"]]
    retained_ids = {unit["unit_id"] for unit in retained}
    for unit in retained:
        unit["negative_unit_ids"] = [
            idx for idx in unit["negative_unit_ids"] if idx in retained_ids
        ]
    retained = [unit for unit in retained if unit["negative_unit_ids"]]
    by_adduct_summary = {}
    for adduct in sorted(by_adduct):
        candidates = [unit for unit in units if unit["adduct"] == adduct]
        kept = [unit for unit in retained if unit["adduct"] == adduct]
        by_adduct_summary[adduct] = {
            "cross_condition_units": len(candidates),
            "mass_dense_units": len(kept),
            "directed_links": int(sum(len(unit["negative_unit_ids"]) for unit in kept)),
        }
    result = {
        "status": "mass_dense_factor_cohort_audit",
        "data": str(args.data),
        "fold": args.fold,
        "ppm": args.ppm,
        "min_ce_difference": args.min_ce_difference,
        "quality_control": quality_audit,
        "n_quality_valid_fold_spectra": int(len(frame)),
        "cross_condition_units_before_mass_filter": int(len(units)),
        "mass_dense_units": int(len(retained)),
        "mass_dense_unique_molecules": int(len({unit["ik14"] for unit in retained})),
        "directed_negative_links": int(
            sum(len(unit["negative_unit_ids"]) for unit in retained)
        ),
        "by_adduct": by_adduct_summary,
        "units": retained,
        "decision_note": (
            "This manifest is eligible for activation extraction only after molecule-"
            "disjoint discovery/confirmation splitting preserves enough 10 ppm links."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"fold spectra={len(frame):,}; cross-condition units={len(units):,}; "
        f"mass-dense units={len(retained):,}; links={result['directed_negative_links']:,}"
    )
    for adduct, summary in by_adduct_summary.items():
        print(adduct, summary)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
