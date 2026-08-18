"""Build leakage-safe hard-triplet candidate pools for E1.

E1 reproduces the identity-only DreaMS contrastive objective:

* anchor / positive: different spectra with the same 14-character InChIKey
* negative: a different molecule within ``mass_window_da`` of the anchor
* all spectra come from one explicit HDF5 fold and one adduct

The output stores candidate lists, not preselected triplets.  The trainer samples
fresh positives and negatives every epoch while preserving a fully auditable,
deterministic candidate universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "models" / "MassSpecGym_MurckoHist_split.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E1 hard-triplet candidate pool")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fold", choices=("train", "val"), required=True)
    parser.add_argument("--adduct", default="[M+H]+")
    parser.add_argument("--mass-window-da", type=float, default=0.05)
    parser.add_argument(
        "--mass-window-ppm", type=float, default=0.0,
        help="Query-centred ppm window; when positive, overrides --mass-window-da "
             "and also filters same-identity positive edges.",
    )
    parser.add_argument("--peak-hash-mz-tol", type=float, default=0.01)
    parser.add_argument("--peak-hash-intensity-tol", type=float, default=0.01)
    parser.add_argument("--max-positive-candidates", type=int, default=0)
    parser.add_argument("--max-negative-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-spectra", type=int, default=0,
                        help="Deterministic debug subset; 0 keeps the complete fold")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def decode_array(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def spectrum_hash(spectrum: np.ndarray, mz_tol: float, intensity_tol: float) -> str:
    """Hash nonzero peaks after tolerance-aware quantization."""
    mz = np.asarray(spectrum[0])
    intensity = np.asarray(spectrum[1])
    keep = mz > 0
    if not np.any(keep):
        return "empty"
    mz_bin = np.rint(mz[keep] / mz_tol).astype(np.int32)
    int_bin = np.rint(intensity[keep] / intensity_tol).astype(np.int32)
    order = np.argsort(mz_bin, kind="stable")
    packed = np.stack((mz_bin[order], int_bin[order]), axis=1)
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()


def cap_candidates(values: np.ndarray, limit: int, rng: np.random.RandomState) -> np.ndarray:
    if limit <= 0 or len(values) <= limit:
        return values
    chosen = rng.choice(len(values), size=limit, replace=False)
    return np.sort(values[chosen])


def describe_counts(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.mass_window_da <= 0 and args.mass_window_ppm <= 0:
        raise ValueError("--mass-window-da must be positive")

    output = args.output
    if output is None:
        output = REPO_ROOT / "data" / "e1" / f"e1_{args.fold}_triplet_pool.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data:   {args.data}")
    print(f"Fold:   {args.fold}")
    print(f"Adduct: {args.adduct}")
    if args.mass_window_ppm > 0:
        print(f"Window: +/- {args.mass_window_ppm:.3f} ppm (query-centred)")
    else:
        print(f"Window: +/- {args.mass_window_da:.5f} Da")

    with h5py.File(args.data, "r") as handle:
        folds = decode_array(handle["fold"][:])
        adducts = decode_array(handle["adduct"][:])
        selected = np.flatnonzero((folds == args.fold) & (adducts == args.adduct))
        if args.max_spectra and len(selected) > args.max_spectra:
            subset_rng = np.random.RandomState(args.seed)
            selected = np.sort(subset_rng.choice(selected, args.max_spectra, replace=False))

        # Reading one large fancy-indexed object array from HDF5 is extremely slow
        # on Windows.  Read the compact metadata columns once, then select in NumPy.
        all_inchikeys = decode_array(handle["INCHIKEY"][:])
        inchikeys = np.asarray([value[:14] for value in all_inchikeys[selected]])
        precursor_mz = np.asarray(handle["precursor_mz"][:], dtype=np.float64)[selected]

        print(f"Selected spectra: {len(selected):,}")
        print("Computing peak hashes...")
        hashes = np.empty(len(selected), dtype="U16")
        global_to_local = np.full(len(folds), -1, dtype=np.int64)
        global_to_local[selected] = np.arange(len(selected))
        chunk_size = 4096
        processed = 0
        for start in range(0, len(folds), chunk_size):
            end = min(start + chunk_size, len(folds))
            local_rows = global_to_local[start:end]
            keep_offsets = np.flatnonzero(local_rows >= 0)
            if len(keep_offsets) == 0:
                continue
            spectra_chunk = handle["spectrum"][start:end]
            for offset in keep_offsets:
                local_idx = local_rows[offset]
                hashes[local_idx] = spectrum_hash(
                    spectra_chunk[offset],
                    args.peak_hash_mz_tol,
                    args.peak_hash_intensity_tol,
                )
            processed += len(keep_offsets)
            if processed // 25000 != (processed - len(keep_offsets)) // 25000:
                print(f"  {processed:,}/{len(selected):,}")

    ik_to_local: dict[str, list[int]] = defaultdict(list)
    for local_idx, ik in enumerate(inchikeys):
        ik_to_local[ik].append(local_idx)

    mz_order = np.argsort(precursor_mz, kind="stable")
    sorted_mz = precursor_mz[mz_order]
    rng = np.random.RandomState(args.seed)

    anchors: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    positive_flat: list[int] = []
    negative_flat: list[int] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    skipped_no_positive = 0
    skipped_no_negative = 0

    print("Building candidate lists...")
    for local_anchor in range(len(selected)):
        same_ik = np.asarray(ik_to_local[inchikeys[local_anchor]], dtype=np.int64)
        positive = same_ik[
            (same_ik != local_anchor) & (hashes[same_ik] != hashes[local_anchor])
        ]
        if args.mass_window_ppm > 0:
            positive_tolerance = precursor_mz[local_anchor] * args.mass_window_ppm * 1e-6
            positive = positive[
                np.abs(precursor_mz[positive] - precursor_mz[local_anchor])
                <= positive_tolerance
            ]
        if len(positive) == 0:
            skipped_no_positive += 1
            continue

        negative_tolerance = (
            precursor_mz[local_anchor] * args.mass_window_ppm * 1e-6
            if args.mass_window_ppm > 0 else args.mass_window_da
        )
        left = np.searchsorted(
            sorted_mz, precursor_mz[local_anchor] - negative_tolerance, side="left"
        )
        right = np.searchsorted(
            sorted_mz, precursor_mz[local_anchor] + negative_tolerance, side="right"
        )
        negative = mz_order[left:right]
        negative = negative[
            (inchikeys[negative] != inchikeys[local_anchor])
            & (hashes[negative] != hashes[local_anchor])
        ]
        if len(negative) == 0:
            skipped_no_negative += 1
            continue

        anchor_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        positive = cap_candidates(positive, args.max_positive_candidates, anchor_rng)
        negative = cap_candidates(negative, args.max_negative_candidates, anchor_rng)

        # Store original HDF5 row indices so the pool remains independent of local ordering.
        anchors.append(int(selected[local_anchor]))
        positive_global = selected[positive].astype(np.int64)
        negative_global = selected[negative].astype(np.int64)
        positive_flat.extend(positive_global.tolist())
        negative_flat.extend(negative_global.tolist())
        positive_ptr.append(len(positive_flat))
        negative_ptr.append(len(negative_flat))
        positive_counts.append(len(positive_global))
        negative_counts.append(len(negative_global))

    if not anchors:
        raise RuntimeError("No eligible E1 anchors were found")

    np.savez_compressed(
        output,
        anchor_idx=np.asarray(anchors, dtype=np.int64),
        positive_ptr=np.asarray(positive_ptr, dtype=np.int64),
        positive_idx=np.asarray(positive_flat, dtype=np.int64),
        negative_ptr=np.asarray(negative_ptr, dtype=np.int64),
        negative_idx=np.asarray(negative_flat, dtype=np.int64),
    )

    audit = {
        "e1_pool_version": "1.0",
        "data": str(args.data.resolve()),
        "hdf5_rows": int(len(folds)),
        "fold": args.fold,
        "adduct": args.adduct,
        "mass_window_da": args.mass_window_da,
        "mass_window_ppm": args.mass_window_ppm,
        "window_protocol": (
            "query_centred_ppm_positive_and_negative"
            if args.mass_window_ppm > 0 else "fixed_da_negative_only"
        ),
        "seed": args.seed,
        "selected_spectra": int(len(selected)),
        "unique_molecules": int(len(set(inchikeys))),
        "eligible_anchors": len(anchors),
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_negative_after_positive": skipped_no_negative,
        "positive_candidates_per_anchor": describe_counts(positive_counts),
        "negative_candidates_per_anchor": describe_counts(negative_counts),
        "peak_hash": {
            "mz_tolerance_da": args.peak_hash_mz_tol,
            "intensity_tolerance": args.peak_hash_intensity_tol,
        },
        "candidate_caps": {
            "positive": args.max_positive_candidates,
            "negative": args.max_negative_candidates,
        },
        "output": str(output.resolve()),
    }
    audit_path = output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nE1 pool complete")
    print(f"  Eligible anchors: {len(anchors):,}")
    print(f"  Positive candidates: {len(positive_flat):,}")
    print(f"  Negative candidates: {len(negative_flat):,}")
    print(f"  Pool:  {output}")
    print(f"  Audit: {audit_path}")


if __name__ == "__main__":
    main()
