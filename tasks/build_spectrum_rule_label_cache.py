"""Build spectrum-level 335-rule labels aligned to cached official embeddings.

Unlike the earlier molecule-level rule-vector proxy, these labels are computed
from each individual MassSpecGym spectrum.  They represent observed spectral
motifs, not guaranteed molecular substructures and not unique mechanisms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
M_H = 1.00782503223


def decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def target_hit(sorted_values: np.ndarray, target: float, tolerance: float) -> bool:
    if sorted_values.size == 0:
        return False
    position = int(np.searchsorted(sorted_values, target))
    if position < sorted_values.size and abs(float(sorted_values[position]) - target) < tolerance:
        return True
    return position > 0 and abs(float(sorted_values[position - 1]) - target) < tolerance


def range_hit(sorted_values: np.ndarray, low: float, high: float) -> bool:
    if sorted_values.size == 0:
        return False
    position = int(np.searchsorted(sorted_values, low, side="left"))
    return position < sorted_values.size and float(sorted_values[position]) <= high


def spectrum_rule_vector(mz_padded: np.ndarray, precursor: float, rules: list[dict]) -> np.ndarray:
    """Match rules on valid experimental peaks only; zero padding is excluded."""
    mz = np.sort(mz_padded[np.isfinite(mz_padded) & (mz_padded > 0)].astype(np.float64))
    if mz.size:
        diffs = np.sort(np.abs(mz[:, None] - mz[None, :]).reshape(-1))
    else:
        diffs = np.empty(0, dtype=np.float64)
    labels = np.zeros(len(rules), dtype=np.uint8)
    for index, rule in enumerate(rules):
        kind = rule.get("match_type")
        value = rule.get("value")
        if kind == "mass_diff":
            labels[index] = target_hit(diffs, float(value), 0.02)
        elif kind == "peak_mz":
            labels[index] = target_hit(mz, float(value), 0.02)
        elif kind == "mass_range":
            labels[index] = range_hit(diffs, float(value[0]), float(value[1]))
        elif kind == "hr_shift":
            hydrogen_count = float(value)
            if hydrogen_count == 0:
                eligible = diffs[diffs >= 12.0]
                labels[index] = bool(
                    eligible.size and np.any(np.abs(eligible - np.round(eligible)) < 0.02)
                )
            else:
                labels[index] = target_hit(diffs, abs(hydrogen_count) * M_H, 0.02)
        elif kind == "parity":
            labels[index] = bool(
                diffs.size and np.any((np.round(diffs).astype(np.int64) % 2) == (round(precursor) % 2))
            )
        elif kind == "mass_diff_range":
            low, high = map(float, value)
            labels[index] = bool(diffs.size and np.any((diffs > high) | (diffs < low)))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_manifest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz",
    )
    parser.add_argument("--max-spectra", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.max_spectra:
        manifest = manifest[: args.max_spectra]
    wanted = [str(row["spectrum_id"]) for row in manifest]

    with h5py.File(args.data, "r") as handle:
        identifiers = [decode(value) for value in handle["IDENTIFIER"][:]]
        index = {identifier: row for row, identifier in enumerate(identifiers)}
        missing = [identifier for identifier in wanted if identifier not in index]
        if missing:
            raise KeyError(f"{len(missing)} manifest identifiers absent from HDF5; first={missing[0]}")
        hdf_rows = np.asarray([index[identifier] for identifier in wanted], dtype=np.int64)
        order = np.argsort(hdf_rows)
        sorted_rows = hdf_rows[order]
        spectra_sorted = np.asarray(handle["spectrum"][sorted_rows], dtype=np.float32)
        precursor_sorted = np.asarray(handle["precursor_mz"][sorted_rows], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    spectra = spectra_sorted[inverse]
    precursor = precursor_sorted[inverse]

    rules = json.loads(
        (ROOT / "dreams/models/chem_aware/chem_rules_data.json").read_text(encoding="utf-8")
    )["rules"]
    labels = np.zeros((len(manifest), len(rules)), dtype=np.uint8)
    for position, (spectrum, precursor_mz) in enumerate(zip(spectra, precursor), start=1):
        labels[position - 1] = spectrum_rule_vector(spectrum[0], precursor_mz, rules)
        if position % 1000 == 0 or position == len(manifest):
            print(f"rules {position:,}/{len(manifest):,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels,
        embedding_idx=np.asarray([row["embedding_idx"] for row in manifest], dtype=np.int64),
        hdf5_row=hdf_rows,
        ik14=np.asarray([row["inchikey_14"] for row in manifest], dtype="U14"),
        spectrum_id=np.asarray(wanted, dtype="U64"),
        rule_name=np.asarray([rule["name"] for rule in rules], dtype="U160"),
        rule_category=np.asarray([rule["category"] for rule in rules], dtype="U8"),
    )
    prevalence = labels.mean(axis=0)
    report = {
        "status": "spectrum_rule_labels_complete",
        "spectra": int(len(labels)),
        "unique_molecules": int(len(set(row["inchikey_14"] for row in manifest))),
        "rules": int(labels.shape[1]),
        "mean_rule_prevalence": float(prevalence.mean()),
        "rules_prevalence_1_to_50_percent": int(((prevalence >= 0.01) & (prevalence <= 0.5)).sum()),
        "label_semantics": "observed spectral motif in this spectrum",
        "padding_policy": "zero padding excluded from every peak and peak-pair rule",
        "claim_limit": "A matched mass pattern is not a unique fragment structure or bond-breaking mechanism.",
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
