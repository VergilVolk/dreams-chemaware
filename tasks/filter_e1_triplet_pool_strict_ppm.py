"""Filter a legacy fixed-Da E1 pool into an exact query-centred ppm pool.

The legacy candidate universe already enforces fold, adduct, IK14 identity and
duplicate-spectrum constraints. A strict ppm pool is therefore an exact subset
and can be produced without recomputing expensive spectrum hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument(
        "--keep-wide-positive", action="store_true",
        help="Do not apply the ppm filter to same-IK14 positive edges.",
    )
    return parser.parse_args()


def within_ppm(anchor: int, targets: np.ndarray, masses: np.ndarray, ppm: float) -> np.ndarray:
    anchor_mass = float(masses[anchor])
    if not np.isfinite(anchor_mass) or anchor_mass <= 0:
        return np.zeros(len(targets), dtype=bool)
    return (
        np.isfinite(masses[targets])
        & (np.abs(masses[targets] - anchor_mass) <= anchor_mass * ppm * 1e-6)
    )


def describe(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "min": int(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.ppm <= 0:
        raise ValueError("--ppm must be positive")
    source = np.load(args.input)
    with h5py.File(args.data, "r") as handle:
        masses = np.asarray(handle["precursor_mz"], dtype=float)

    anchors_out: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    positive_out: list[int] = []
    negative_out: list[int] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    removed_no_positive = removed_no_negative = 0

    anchors = source["anchor_idx"].astype(np.int64)
    p_ptr, n_ptr = source["positive_ptr"], source["negative_ptr"]
    p_values, n_values = source["positive_idx"], source["negative_idx"]
    for row, anchor in enumerate(anchors):
        positives = p_values[p_ptr[row]:p_ptr[row + 1]].astype(np.int64)
        negatives = n_values[n_ptr[row]:n_ptr[row + 1]].astype(np.int64)
        if not args.keep_wide_positive:
            positives = positives[within_ppm(anchor, positives, masses, args.ppm)]
        negatives = negatives[within_ppm(anchor, negatives, masses, args.ppm)]
        if not len(positives):
            removed_no_positive += 1
            continue
        if not len(negatives):
            removed_no_negative += 1
            continue
        anchors_out.append(int(anchor))
        positive_out.extend(positives.tolist())
        negative_out.extend(negatives.tolist())
        positive_ptr.append(len(positive_out))
        negative_ptr.append(len(negative_out))
        positive_counts.append(len(positives))
        negative_counts.append(len(negatives))

    if not anchors_out:
        raise RuntimeError("No anchors survive the strict ppm filter")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        anchor_idx=np.asarray(anchors_out, dtype=np.int64),
        positive_ptr=np.asarray(positive_ptr, dtype=np.int64),
        positive_idx=np.asarray(positive_out, dtype=np.int64),
        negative_ptr=np.asarray(negative_ptr, dtype=np.int64),
        negative_idx=np.asarray(negative_out, dtype=np.int64),
    )
    report = {
        "status": "strict_ppm_e1_pool",
        "source_pool": str(args.input),
        "ppm": args.ppm,
        "positive_ppm_filter": not args.keep_wide_positive,
        "source_anchors": int(len(anchors)),
        "eligible_anchors": int(len(anchors_out)),
        "removed_no_strict_positive": int(removed_no_positive),
        "removed_no_strict_negative_after_positive": int(removed_no_negative),
        "positive_edges": int(len(positive_out)),
        "negative_edges": int(len(negative_out)),
        "positive_candidates_per_anchor": describe(positive_counts),
        "negative_candidates_per_anchor": describe(negative_counts),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
