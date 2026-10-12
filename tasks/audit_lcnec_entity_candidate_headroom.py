"""Phenotype-blind cross-platform candidate-space audit for LCNEC entities.

This script deliberately does not emit calibrated match probabilities or identity
assignments. It asks whether the frozen discovery and validation QC universes have
enough same-precursor candidate overlap to justify a later calibration experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--discovery-mgf", type=Path, required=True)
    parser.add_argument("--validation-mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precursor-ppm", type=float, default=5.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument(
        "--min-global-coverage", type=float, default=0.10,
        help="Phenotype-blind engineering gate fixed before reading study effects.",
    )
    return parser.parse_args()


def read_mgf(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    name: str | None = None
    mz: list[float] = []
    intensity: list[float] = []
    in_spectrum = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "BEGIN IONS":
            in_spectrum, name, mz, intensity = True, None, [], []
        elif line == "END IONS":
            if name is None:
                raise ValueError(f"spectrum without NAME in {path}")
            family_id = int(name.rsplit("_", 1)[-1])
            spectra[family_id] = (np.asarray(mz), np.asarray(intensity))
            in_spectrum = False
        elif in_spectrum and line.startswith("NAME="):
            name = line.split("=", 1)[1]
        elif in_spectrum and line and "=" not in line:
            fields = line.split()
            if len(fields) >= 2:
                mz.append(float(fields[0]))
                intensity.append(float(fields[1]))
    return spectra


def sqrt_cosine(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> tuple[float, int]:
    left_mz, left_intensity = left
    right_mz, right_intensity = right
    left_weight = np.sqrt(np.maximum(left_intensity, 0.0))
    right_weight = np.sqrt(np.maximum(right_intensity, 0.0))
    left_norm = float(np.linalg.norm(left_weight))
    right_norm = float(np.linalg.norm(right_weight))
    if left_norm == 0 or right_norm == 0:
        return 0.0, 0
    proposals: list[tuple[float, int, int]] = []
    for i, mass in enumerate(left_mz):
        lo = int(np.searchsorted(right_mz, mass - tolerance, side="left"))
        hi = int(np.searchsorted(right_mz, mass + tolerance, side="right"))
        for j in range(lo, hi):
            proposals.append((float(left_weight[i] * right_weight[j]), i, j))
    used_left: set[int] = set()
    used_right: set[int] = set()
    dot = 0.0
    for contribution, i, j in sorted(proposals, reverse=True):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            dot += contribution
    return dot / (left_norm * right_norm), len(used_left)


def main() -> None:
    args = parse_args()
    discovery = pd.read_csv(args.discovery_manifest)
    validation = pd.read_csv(args.validation_manifest)
    discovery_spectra = read_mgf(args.discovery_mgf)
    validation_spectra = read_mgf(args.validation_mgf)
    required = {"family_id", "target_mz", "n_peaks"}
    for label, table in (("discovery", discovery), ("validation", validation)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{label} manifest missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int]] = []
    validation_mz = validation["target_mz"].to_numpy(float)
    for source in discovery.itertuples(index=False):
        ppm = np.abs(validation_mz - float(source.target_mz)) / float(source.target_mz) * 1e6
        for position in np.flatnonzero(ppm <= args.precursor_ppm):
            target = validation.iloc[int(position)]
            source_id, target_id = int(source.family_id), int(target.family_id)
            score, matched_peaks = sqrt_cosine(
                discovery_spectra[source_id], validation_spectra[target_id], args.fragment_tolerance_da
            )
            rows.append({
                "discovery_family_id": source_id,
                "validation_family_id": target_id,
                "discovery_mz": float(source.target_mz),
                "validation_mz": float(target.target_mz),
                "precursor_ppm": float(ppm[position]),
                "sqrt_cosine": float(score),
                "matched_peaks": matched_peaks,
                "discovery_n_peaks": int(source.n_peaks),
                "validation_n_peaks": int(target.n_peaks),
            })
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["discovery_family_id", "sqrt_cosine", "precursor_ppm"],
            ascending=[True, False, True],
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "candidate_pairs.csv", index=False)

    counts = pairs.groupby("discovery_family_id").size() if not pairs.empty else pd.Series(dtype=int)
    validation_counts = (
        pairs.groupby("validation_family_id").size() if not pairs.empty else pd.Series(dtype=int)
    )
    covered = int(counts.size)
    coverage = covered / len(discovery) if len(discovery) else 0.0
    report = {
        "status": "phenotype_blind_candidate_headroom_complete",
        "formal": True,
        "phenotype_labels_read": False,
        "discovery_entities": int(len(discovery)),
        "validation_entities": int(len(validation)),
        "precursor_ppm": args.precursor_ppm,
        "fragment_tolerance_da": args.fragment_tolerance_da,
        "candidate_pairs": int(len(pairs)),
        "discovery_entities_with_candidate": covered,
        "discovery_candidate_coverage": coverage,
        "unique_candidate_entities": int((counts == 1).sum()),
        "ambiguous_candidate_entities": int((counts > 1).sum()),
        "validation_entities_reached": int(validation_counts.size),
        "validation_entities_with_source_collision": int((validation_counts > 1).sum()),
        "median_sqrt_cosine": None if pairs.empty else float(pairs["sqrt_cosine"].median()),
        "pairs_sqrt_cosine_ge_0_7": 0 if pairs.empty else int((pairs["sqrt_cosine"] >= 0.7).sum()),
        "min_global_coverage": args.min_global_coverage,
        "pass_to_global_transport": bool(coverage >= args.min_global_coverage),
        "claim_limit": (
            "Uncalibrated phenotype-blind candidate-space headroom only; precursor agreement and "
            "spectral cosine do not establish identity, match p-values, effect replication, or biology."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
