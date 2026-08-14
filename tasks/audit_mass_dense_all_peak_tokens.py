"""Audit split isolation and nuisance distributions before factor discovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_split(path: Path) -> dict:
    spectra = json.loads((path / "spectra.json").read_text(encoding="utf-8"))
    mask = np.load(path / "peak_mask.npy")
    values = np.load(path / "peak_values.npy")
    precursor = np.asarray(
        [record.get("precursor_mz") for record in spectra], dtype=float
    )
    mz = values[:, :, 0][mask].astype(float)
    intensity = values[:, :, 1][mask].astype(float)
    spectrum_index = np.repeat(np.arange(len(spectra)), mask.sum(axis=1))
    neutral_loss = precursor[spectrum_index] - mz
    return {
        "spectra": spectra,
        "mask": mask,
        "mz": mz,
        "intensity": intensity,
        "neutral_loss": neutral_loss,
        "precursor_mz": precursor,
    }


def quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.5)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def categorical(records: list[dict], key: str) -> dict:
    values = [str(record.get(key) or "missing") for record in records]
    counts = Counter(values)
    return {name: {"n": count, "fraction": count / len(values)} for name, count in counts.most_common()}


def main() -> None:
    args = parse_args()
    discovery = load_split(args.discovery)
    confirmation = load_split(args.confirmation)
    disc_ik = {str(record.get("ik14")) for record in discovery["spectra"]}
    conf_ik = {str(record.get("ik14")) for record in confirmation["spectra"]}
    overlap = sorted(disc_ik & conf_ik)

    continuous = {}
    for key in ("mz", "intensity", "neutral_loss", "precursor_mz"):
        a, b = discovery[key], confirmation[key]
        continuous[key] = {
            "discovery": quantiles(a),
            "confirmation": quantiles(b),
            "standardized_mean_difference_discovery_minus_confirmation": standardized_mean_difference(a, b),
        }
    categories = {}
    for key in ("adduct", "INSTRUMENT_TYPE", "fold"):
        categories[key] = {
            "discovery": categorical(discovery["spectra"], key),
            "confirmation": categorical(confirmation["spectra"], key),
        }
    for split_name, split in (("discovery", discovery), ("confirmation", confirmation)):
        energies = np.asarray(
            [record.get("COLLISION_ENERGY") for record in split["spectra"]], dtype=float
        )
        continuous[f"collision_energy_{split_name}"] = quantiles(energies)

    report = {
        "status": "pre_factor_discovery_audit",
        "isolation": {
            "discovery_unique_ik14": len(disc_ik),
            "confirmation_unique_ik14": len(conf_ik),
            "ik14_overlap_count": len(overlap),
            "ik14_overlap": overlap,
            "passed": len(overlap) == 0,
        },
        "continuous_distributions": continuous,
        "categorical_distributions": categories,
        "interpretation": (
            "Confirmation is held out from factor fitting. Distribution summaries "
            "are nuisance audits, not evidence of chemical-factor validity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["isolation"], indent=2))
    for key in ("mz", "intensity", "neutral_loss", "precursor_mz"):
        smd = continuous[key]["standardized_mean_difference_discovery_minus_confirmation"]
        print(f"{key}: SMD={smd:.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
