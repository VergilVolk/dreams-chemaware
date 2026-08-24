"""M0 feasibility audit for consensus-based directional peak noise.

This script does not train a model and does not create labels.  It restricts
itself to the sealed P2 real-training allow-list, constructs within-identity
peak prevalence summaries, and checks whether enough groups can support the
pre-registered margin-level causal experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--p3-dir", type=Path, default=ROOT / "data/validation/g8r_p3_test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m0")
    parser.add_argument("--minimum-spectra", type=int, default=3)
    parser.add_argument("--maximum-spectra", type=int, default=12)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--core-prevalence", type=float, default=0.60)
    parser.add_argument("--conditional-prevalence", type=float, default=0.40)
    parser.add_argument("--conditional-intensity-max", type=float, default=0.20)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ], dtype=object)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(row: int, seed: int) -> int:
    body = hashlib.blake2b(f"{seed}|{row}".encode(), digest_size=8).digest()
    return int.from_bytes(body, "little")


def condition_label(instrument: str, collision_energy: float) -> str:
    inst = instrument if instrument and instrument.lower() not in {"nan", "none"} else "unknown-instrument"
    ce = "unknown-ce" if not np.isfinite(collision_energy) else f"ce-{int(round(collision_energy / 10.0) * 10)}"
    return f"{inst}|{ce}"


def choose_condition_diverse_rows(rows: list[int], labels: dict[int, str], maximum: int, seed: int) -> list[int]:
    if len(rows) <= maximum:
        return sorted(rows)
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        buckets[labels[row]].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_order(row, seed))
    chosen: list[int] = []
    names = sorted(buckets)
    while len(chosen) < maximum:
        progress = False
        for name in names:
            if buckets[name] and len(chosen) < maximum:
                chosen.append(buckets[name].pop(0))
                progress = True
        if not progress:
            break
    return sorted(chosen)


def valid_peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=float)
    intensity = np.asarray(spectrum[1], dtype=float)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="mergesort")
    return mz[order], intensity[order]


def peak_present(mz_values: np.ndarray, target: float, tolerance: float) -> bool:
    position = int(np.searchsorted(mz_values, target))
    candidates = []
    if position < len(mz_values):
        candidates.append(abs(float(mz_values[position]) - target))
    if position:
        candidates.append(abs(float(mz_values[position - 1]) - target))
    return bool(candidates and min(candidates) <= tolerance)


def classify_group_peaks(
    spectra: dict[int, np.ndarray], rows: list[int], labels: dict[int, str], tolerance: float,
    core_prevalence: float, conditional_prevalence: float, conditional_intensity_max: float,
) -> list[dict]:
    parsed = {row: valid_peaks(spectra[row]) for row in rows}
    conditions = sorted({labels[row] for row in rows})
    output: list[dict] = []
    for source in rows:
        source_mz, source_intensity = parsed[source]
        core_mz, candidate_mz, candidate_intensity = [], [], []
        for mz, intensity in zip(source_mz, source_intensity):
            present_rows = [row for row in rows if peak_present(parsed[row][0], float(mz), tolerance)]
            present_conditions = {labels[row] for row in present_rows}
            prevalence = len(present_rows) / len(rows)
            condition_prevalence = len(present_conditions) / len(conditions)
            is_core = prevalence >= core_prevalence and condition_prevalence >= 0.5
            is_conditional = (
                prevalence <= conditional_prevalence
                and len(rows) - len(present_rows) >= 2
                and float(intensity) < conditional_intensity_max
                and not is_core
            )
            if is_core:
                core_mz.append(float(mz))
            if is_conditional:
                candidate_mz.append(float(mz))
                candidate_intensity.append(float(intensity))
        output.append({
            "row": int(source),
            "n_valid_peaks": int(len(source_mz)),
            "n_core_peaks": int(len(core_mz)),
            "n_conditional_candidates": int(len(candidate_mz)),
            "core_mz": core_mz,
            "conditional_mz": candidate_mz,
            "conditional_intensity": candidate_intensity,
        })
    return output


def build_mass_index(
    allowed_rows: np.ndarray, precursor: np.ndarray, adduct: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for ion in np.unique(adduct[allowed_rows]):
        rows = allowed_rows[adduct[allowed_rows] == ion]
        order = np.argsort(precursor[rows], kind="mergesort")
        output[str(ion)] = (precursor[rows][order], rows[order])
    return output


def hard_negative_rows(
    row: int, precursor: np.ndarray, adduct: np.ndarray, ik14: np.ndarray,
    mass_index: dict[str, tuple[np.ndarray, np.ndarray]], ppm: float,
) -> np.ndarray:
    tolerance = float(precursor[row]) * ppm * 1e-6
    masses, rows = mass_index[str(adduct[row])]
    left = int(np.searchsorted(masses, precursor[row] - tolerance, side="left"))
    right = int(np.searchsorted(masses, precursor[row] + tolerance, side="right"))
    candidates = rows[left:right]
    return candidates[ik14[candidates] != ik14[row]]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    allow_path = args.p3_dir / "p3_p2_allowed_training_ik14.json"
    for path in (args.data, allow_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    allow_body = json.loads(allow_path.read_text(encoding="utf-8"))
    allowed_rows = np.asarray(allow_body["real_train_primary"]["rows"], dtype=np.int64)
    allowed_ik = set(map(str, allow_body["real_train_primary"]["ik14"]))

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])], dtype=object)
        adduct = decode(handle["adduct"][:])
        fold = decode(handle["fold"][:])
        simulation = decode(handle["SIMULATION_CHALLENGE"][:])
        instrument = decode(handle["INSTRUMENT_TYPE"][:])
        ce = np.asarray(handle["COLLISION_ENERGY"][:], dtype=float)
        precursor = np.asarray(handle["precursor_mz"][:], dtype=float)
        formula = decode(handle["FORMULA"][:])
        if np.any(fold[allowed_rows] != "train") or np.any(simulation[allowed_rows] != "False"):
            raise RuntimeError("P2 real allow-list contains non-real or non-train rows")
        if not set(ik14[allowed_rows]).issubset(allowed_ik):
            raise RuntimeError("P2 row and identity allow-lists disagree")
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in allowed_rows:
            groups[(str(ik14[row]), str(adduct[row]))].append(int(row))
        eligible_groups = {key: rows for key, rows in groups.items() if len(rows) >= args.minimum_spectra}
        labels = {
            int(row): condition_label(str(instrument[row]), float(ce[row])) for row in allowed_rows
        }
        mass_index = build_mass_index(allowed_rows, precursor, adduct)
        group_records, source_records = [], []
        for position, ((identity, ion), rows) in enumerate(sorted(eligible_groups.items())):
            selected_rows = choose_condition_diverse_rows(rows, labels, args.maximum_spectra, args.seed)
            spectra = {row: np.asarray(handle["spectrum"][row], dtype=float) for row in selected_rows}
            peak_rows = classify_group_peaks(
                spectra, selected_rows, labels, args.fragment_tolerance,
                args.core_prevalence, args.conditional_prevalence,
                args.conditional_intensity_max,
            )
            conditions = {labels[row] for row in selected_rows}
            cross_condition = len(conditions) >= 2
            for record in peak_rows:
                row = int(record["row"])
                negatives = hard_negative_rows(
                    row, precursor, adduct, ik14, mass_index, args.ppm,
                )
                record.update({
                    "ik14": identity,
                    "adduct": ion,
                    "formula": str(formula[row]),
                    "condition": labels[row],
                    "n_group_spectra": int(len(selected_rows)),
                    "n_conditions": int(len(conditions)),
                    "cross_condition_group": bool(cross_condition),
                    "has_strict_10ppm_negative": bool(len(negatives)),
                })
                source_records.append(record)
            group_records.append({
                "ik14": identity,
                "adduct": ion,
                "n_available_spectra": int(len(rows)),
                "n_selected_spectra": int(len(selected_rows)),
                "n_conditions": int(len(conditions)),
                "cross_condition_group": bool(cross_condition),
                "median_core_peaks": float(np.median([row["n_core_peaks"] for row in peak_rows])),
                "sources_with_conditional_candidates": int(sum(row["n_conditional_candidates"] > 0 for row in peak_rows)),
                "sources_with_strict_10ppm_negative": int(sum(record["has_strict_10ppm_negative"] for record in source_records[-len(peak_rows):])),
            })
            if (position + 1) % 250 == 0:
                print(f"[consensus] {position + 1:,}/{len(eligible_groups):,} groups", flush=True)

    group_frame = pd.DataFrame(group_records)
    source_frame = pd.DataFrame(source_records)
    causal_ready = (
        (source_frame["n_conditional_candidates"] > 0)
        & source_frame["has_strict_10ppm_negative"]
        & (source_frame["n_core_peaks"] >= 3)
    )
    report = {
        "status": "directional_noise_v2_m0_complete",
        "allowed_real_train_rows": int(len(allowed_rows)),
        "eligible_identity_adduct_groups": int(len(group_frame)),
        "eligible_identities": int(group_frame["ik14"].nunique()),
        "cross_condition_groups": int(group_frame["cross_condition_group"].sum()),
        "median_group_core_peaks": float(group_frame["median_core_peaks"].median()),
        "source_spectra_with_conditional_candidates": int((source_frame["n_conditional_candidates"] > 0).sum()),
        "causal_ready_source_spectra": int(causal_ready.sum()),
        "causal_ready_identities": int(source_frame.loc[causal_ready, "ik14"].nunique()),
        "gates": {
            "groups_ge_1000": bool(len(group_frame) >= 1000),
            "cross_condition_groups_ge_500": bool(group_frame["cross_condition_group"].sum() >= 500),
            "median_core_peaks_ge_3": bool(group_frame["median_core_peaks"].median() >= 3),
            "causal_ready_identities_ge_300": bool(source_frame.loc[causal_ready, "ik14"].nunique() >= 300),
        },
        "parameters": vars(args) | {
            "data": str(args.data), "p3_dir": str(args.p3_dir), "output_dir": str(args.output_dir),
        },
        "provenance": {"hdf5_sha256": sha256(args.data), "p2_allow_sha256": sha256(allow_path)},
    }
    report["pass"] = bool(all(report["gates"].values()))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        group_frame.to_csv(staging / "identity_adduct_groups.csv.gz", index=False, compression="gzip")
        json_columns = ["core_mz", "conditional_mz", "conditional_intensity"]
        for column in json_columns:
            source_frame[column] = source_frame[column].map(lambda value: json.dumps(value, separators=(",", ":")))
        source_frame.to_csv(staging / "source_peak_consensus.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
