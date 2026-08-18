"""Audit every positive and negative edge in the E1 triplet pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path, default=ROOT / "data/e1/e1_train_triplet_pool.npz"
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_audit.json",
    )
    parser.add_argument("--anchor-chunk", type=int, default=4096)
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def audit_edges(
    anchors: np.ndarray,
    ptr: np.ndarray,
    targets: np.ndarray,
    precursor_mz: np.ndarray,
    ik14: np.ndarray,
    folds: np.ndarray,
    chunk: int,
) -> dict[str, object]:
    total = same_identity = within_10ppm = same_fold = finite_mass = 0
    max_ppm = 0.0
    ppm_samples = []
    rng = np.random.default_rng(20260815)
    for start in range(0, len(anchors), chunk):
        stop = min(start + chunk, len(anchors))
        counts = np.diff(ptr[start:stop + 1])
        if not counts.sum():
            continue
        source = np.repeat(anchors[start:stop], counts)
        target = targets[ptr[start]:ptr[stop]]
        mass_a, mass_b = precursor_mz[source], precursor_mz[target]
        valid = np.isfinite(mass_a) & np.isfinite(mass_b) & ((mass_a + mass_b) > 0)
        ppm = np.full(len(source), np.nan, dtype=float)
        ppm[valid] = np.abs(mass_a[valid] - mass_b[valid]) / (
            (mass_a[valid] + mass_b[valid]) / 2.0
        ) * 1e6
        total += len(source)
        finite_mass += int(valid.sum())
        same_identity += int((ik14[source] == ik14[target]).sum())
        within_10ppm += int((ppm[valid] <= 10.0).sum())
        same_fold += int((folds[source] == folds[target]).sum())
        if valid.any():
            max_ppm = max(max_ppm, float(np.nanmax(ppm)))
            take = min(1000, int(valid.sum()))
            ppm_samples.extend(rng.choice(ppm[valid], size=take, replace=False).tolist())
    return {
        "edges": int(total),
        "finite_precursor_mass_edges": int(finite_mass),
        "same_ik14_edges": int(same_identity),
        "different_ik14_edges": int(total - same_identity),
        "within_10ppm_edges": int(within_10ppm),
        "within_10ppm_fraction_of_finite": float(within_10ppm / max(finite_mass, 1)),
        "same_fold_edges": int(same_fold),
        "max_ppm": float(max_ppm),
        "sampled_ppm_quantiles": {
            str(q): float(np.quantile(ppm_samples, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
    }


def main() -> None:
    args = parse_args()
    pool = np.load(args.pool)
    with h5py.File(args.data, "r") as handle:
        precursor_mz = np.asarray(handle["precursor_mz"], dtype=float)
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])], dtype=object)
        folds = decode(handle["fold"][:])
    anchors = pool["anchor_idx"].astype(np.int64)
    report = {
        "status": "e1_triplet_pool_full_edge_audit",
        "pool": str(args.pool),
        "anchors": int(len(anchors)),
        "unique_anchor_spectra": int(np.unique(anchors).size),
        "anchor_folds": {
            str(key): int(value)
            for key, value in zip(*np.unique(folds[anchors], return_counts=True))
        },
        "positive_edges": audit_edges(
            anchors, pool["positive_ptr"], pool["positive_idx"],
            precursor_mz, ik14, folds, args.anchor_chunk,
        ),
        "negative_edges": audit_edges(
            anchors, pool["negative_ptr"], pool["negative_idx"],
            precursor_mz, ik14, folds, args.anchor_chunk,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
