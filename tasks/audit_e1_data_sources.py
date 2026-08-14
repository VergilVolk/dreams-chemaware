"""Audit annotated01 and MassSpecGym for budget-aware E1 training decisions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", type=Path, default=ROOT / "data" / "annotated01.mgf")
    parser.add_argument("--indices", type=Path, default=ROOT / "tasks" / "_cache" / "indices.json")
    parser.add_argument("--massspecgym", type=Path,
                        default=ROOT / "data" / "models" / "MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--massspecgym-metadata", type=Path,
                        default=ROOT / "data" / "massspecgym" / "metadata.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "e1" / "dataset_audit.json")
    parser.add_argument("--skip-annotated-scan", action="store_true")
    return parser.parse_args()


def decode(values):
    return np.asarray([
        value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def quantiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def load_annotated_index(path: Path):
    print(f"Loading annotated01 index: {path}", flush=True)
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    counts = defaultdict(int)
    for key, value in raw["ik_counts"].items():
        counts[key[:14]] += int(value)
    result = {
        "unique_ik14": len(counts),
        "spectra": int(sum(counts.values())),
        "spectra_per_ik14": quantiles(list(counts.values())),
        "ik14_with_2plus_spectra": int(sum(value >= 2 for value in counts.values())),
        "ik14_with_5plus_spectra": int(sum(value >= 5 for value in counts.values())),
        "formula_annotated_ik": len({key[:14] for key in raw.get("ik_to_fm", {})}),
        "murcko_annotated_ik": len({key[:14] for key in raw.get("ik_to_murcko", {})}),
    }
    return raw, counts, result


def load_massspecgym_test_iks(path: Path):
    test = set()
    train = set()
    val = set()
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ik = row["inchikey14"][:14]
            if int(row["n_spectra_train"]) > 0:
                train.add(ik)
            if int(row["n_spectra_val"]) > 0:
                val.add(ik)
            if int(row["n_spectra_test"]) > 0:
                test.add(ik)
    return train, val, test


def audit_massspecgym(path: Path):
    print(f"Auditing MassSpecGym: {path}", flush=True)
    with h5py.File(path, "r") as handle:
        folds = decode(handle["fold"][:])
        adducts = decode(handle["adduct"][:])
        iks = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])])
        instruments = decode(handle["INSTRUMENT_TYPE"][:])
        formulas = decode(handle["FORMULA"][:])
        smiles = decode(handle["smiles"][:])
        ce = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)
        precursor = np.asarray(handle["precursor_mz"][:], dtype=np.float64)

        n_spectra = len(folds)
        peak_counts = np.zeros(n_spectra, dtype=np.int16)
        intense_counts = np.zeros(n_spectra, dtype=np.int16)
        max_fragment_mz = np.zeros(n_spectra, dtype=np.float32)
        chunk = 8192
        for start in range(0, n_spectra, chunk):
            end = min(start + chunk, n_spectra)
            spectra = np.asarray(handle["spectrum"][start:end])
            mz = spectra[:, 0, :]
            intensity = spectra[:, 1, :]
            peak_counts[start:end] = np.sum(mz > 0, axis=1)
            maximum = intensity.max(axis=1, keepdims=True)
            intense_counts[start:end] = np.sum(
                (mz > 0) & (intensity >= 0.1 * np.maximum(maximum, 1e-12)), axis=1
            )
            max_fragment_mz[start:end] = mz.max(axis=1)

    per_ik = Counter(iks)
    fold_summary = {}
    for fold in sorted(set(folds)):
        mask = folds == fold
        fold_summary[fold] = {
            "spectra": int(mask.sum()),
            "unique_ik14": int(len(set(iks[mask]))),
            "adducts": dict(Counter(adducts[mask]).most_common()),
            "instruments": dict(Counter(instruments[mask]).most_common()),
            "peak_count": quantiles(peak_counts[mask]),
            "intense_peak_count": quantiles(intense_counts[mask]),
            "precursor_mz": quantiles(precursor[mask]),
            "a_quality_proxy_count": int(np.sum(
                mask & (intense_counts >= 3) & (precursor <= 1000) & (max_fragment_mz <= 1000)
            )),
        }

    result = {
        "spectra": n_spectra,
        "unique_ik14": len(per_ik),
        "spectra_per_ik14": quantiles(list(per_ik.values())),
        "ik14_with_2plus_spectra": int(sum(value >= 2 for value in per_ik.values())),
        "adducts": dict(Counter(adducts).most_common()),
        "instruments": dict(Counter(instruments).most_common()),
        "collision_energy": quantiles(ce[np.isfinite(ce)]),
        "collision_energy_missing": int(np.sum(~np.isfinite(ce))),
        "formula_missing": int(np.sum(np.asarray([not value for value in formulas]))),
        "smiles_missing": int(np.sum(np.asarray([not value for value in smiles]))),
        "folds": fold_summary,
    }
    return iks, folds, result


def audit_annotated_mgf(path: Path):
    print(f"Streaming annotated01: {path}", flush=True)
    spectra = 0
    ion_modes = Counter()
    header_presence = Counter()
    peak_count_sample = []
    intense_count_sample = []
    precursor_sample = []
    exact = Counter()
    ik_pm_min = {}
    ik_pm_max = {}
    ik_ion_mask = defaultdict(int)

    current_ik = ""
    current_pm = math.nan
    current_ion = "UNKNOWN"
    has_formula = False
    has_smiles = False
    peaks_mz = []
    peaks_intensity = []

    def finish():
        nonlocal spectra, current_ik, current_pm, current_ion, has_formula, has_smiles
        nonlocal peaks_mz, peaks_intensity
        if not current_ik or not peaks_mz:
            return
        spectra += 1
        ion_modes[current_ion] += 1
        header_presence["formula"] += int(has_formula)
        header_presence["smiles"] += int(has_smiles)
        header_presence["precursor"] += int(math.isfinite(current_pm))
        # annotated01 intentionally has no ADduct field, but retain the counter
        # so this absence is explicit in the report.
        intensity = np.asarray(peaks_intensity, dtype=np.float64)
        maximum = float(intensity.max())
        intense = int(np.sum(intensity >= 0.1 * maximum)) if maximum > 0 else 0
        n_peaks = len(peaks_mz)
        max_mz = max(peaks_mz)
        high_quality = (
            intense >= 3 and math.isfinite(current_pm)
            and current_pm <= 1000 and max_mz <= 1000
        )
        exact["three_intense_peaks"] += int(intense >= 3)
        exact["precursor_le_1000"] += int(math.isfinite(current_pm) and current_pm <= 1000)
        exact["max_fragment_le_1000"] += int(max_mz <= 1000)
        exact["a_quality_proxy"] += int(high_quality)
        exact["max_intensity_le_1.01"] += int(maximum <= 1.01)
        exact["max_intensity_le_100.01"] += int(maximum <= 100.01)
        if spectra % 16 == 0:
            peak_count_sample.append(n_peaks)
            intense_count_sample.append(intense)
            precursor_sample.append(current_pm)
        if math.isfinite(current_pm):
            ik_pm_min[current_ik] = min(ik_pm_min.get(current_ik, current_pm), current_pm)
            ik_pm_max[current_ik] = max(ik_pm_max.get(current_ik, current_pm), current_pm)
        if current_ion == "POSITIVE":
            ik_ion_mask[current_ik] |= 1
        elif current_ion == "NEGATIVE":
            ik_ion_mask[current_ik] |= 2
        else:
            ik_ion_mask[current_ik] |= 4
        if spectra % 500000 == 0:
            print(f"  {spectra:,} spectra", flush=True)

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                current_ik = ""
                current_pm = math.nan
                current_ion = "UNKNOWN"
                has_formula = False
                has_smiles = False
                peaks_mz = []
                peaks_intensity = []
            elif line == "END IONS":
                finish()
            elif line.startswith("INCHIKEY="):
                current_ik = line[9:].strip()[:14]
            elif line.startswith("PEPMASS="):
                try:
                    current_pm = float(line[8:].split()[0].split("/")[0])
                except ValueError:
                    current_pm = math.nan
            elif line.startswith("IONMODE="):
                current_ion = line[8:].strip().upper() or "UNKNOWN"
            elif line.startswith("FORMULA="):
                has_formula = bool(line[8:].strip())
            elif line.startswith("SMILES="):
                has_smiles = bool(line[7:].strip())
            elif line and (line[0].isdigit() or line[0] == "-"):
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        mz, value = float(fields[0]), float(fields[1])
                    except ValueError:
                        continue
                    if mz > 0 and value > 0:
                        peaks_mz.append(mz)
                        peaks_intensity.append(value)

    pm_spreads = np.asarray([
        ik_pm_max[ik] - minimum for ik, minimum in ik_pm_min.items()
    ])
    return {
        "spectra_scanned": spectra,
        "ion_modes": dict(ion_modes),
        "header_coverage": {
            key: {"count": int(value), "fraction": float(value / spectra)}
            for key, value in header_presence.items()
        },
        "adduct_header_count": 0,
        "collision_energy_header_count": 0,
        "instrument_header_count": 0,
        "source_provenance_header_count": 0,
        "peak_count_sample_1_in_16": quantiles(peak_count_sample),
        "intense_peak_count_sample_1_in_16": quantiles(intense_count_sample),
        "precursor_mz_sample_1_in_16": quantiles([
            value for value in precursor_sample if math.isfinite(value)
        ]),
        "quality_counts": dict(exact),
        "quality_fractions": {key: float(value / spectra) for key, value in exact.items()},
        "ik14_with_precursor": len(ik_pm_min),
        "ik14_precursor_spread_gt_0.1_da": int(np.sum(pm_spreads > 0.1)),
        "ik14_precursor_spread_gt_1_da": int(np.sum(pm_spreads > 1.0)),
        "ik14_mixed_polarity": int(
            sum(bool(mask & 1) and bool(mask & 2) for mask in ik_ion_mask.values())
        ),
    }


def main():
    args = parse_args()
    _, annotated_counts, annotated_index = load_annotated_index(args.indices)
    annotated_iks = set(annotated_counts)
    metadata_train, metadata_val, metadata_test = load_massspecgym_test_iks(
        args.massspecgym_metadata
    )
    msg_iks, msg_folds, massspecgym = audit_massspecgym(args.massspecgym)
    h5_train = set(msg_iks[msg_folds == "train"])
    h5_val = set(msg_iks[msg_folds == "val"])

    overlap = {
        "annotated01_vs_massspecgym_hdf5_train": len(annotated_iks & h5_train),
        "annotated01_vs_massspecgym_hdf5_val": len(annotated_iks & h5_val),
        "annotated01_vs_massspecgym_metadata_test": len(annotated_iks & metadata_test),
        "massspecgym_hdf5_train_vs_val": len(h5_train & h5_val),
        "metadata_sets": {
            "train_ik14": len(metadata_train),
            "val_ik14": len(metadata_val),
            "test_ik14": len(metadata_test),
        },
        "annotated01_fraction_overlapping_any_massspecgym": float(
            len(annotated_iks & (metadata_train | metadata_val | metadata_test)) / len(annotated_iks)
        ),
    }

    result = {
        "annotated01_index": annotated_index,
        "massspecgym": massspecgym,
        "overlap": overlap,
    }
    if not args.skip_annotated_scan:
        result["annotated01_spectrum_scan"] = audit_annotated_mgf(args.annotated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Audit written to {args.output}")
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
