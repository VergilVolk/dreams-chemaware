#!/usr/bin/env python3
"""Export frozen HDF5 edge triples for exact MetDNA2 R scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="mergesort")
    ordered = np.asarray(dataset[rows[order]])
    return ordered[np.argsort(order, kind="mergesort")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triples_path = args.manifest_dir / "paired_reaction_decoy_triples.csv.gz"
    report_path = args.manifest_dir / "report.json"
    for path in (triples_path, report_path, args.data):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    manifest = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "kgmn_dreams_edge_calibration_manifest_frozen":
        raise RuntimeError("edge calibration manifest status mismatch")
    if manifest["provenance"].get("triples_sha256") != sha256(triples_path):
        raise RuntimeError("edge triple hash mismatch")

    triples = pd.read_csv(triples_path)
    row_columns = ["source_row", "positive_row", "decoy_row"]
    missing = sorted(set(row_columns) - set(triples.columns))
    if missing:
        raise RuntimeError(f"edge triples miss row columns: {missing}")
    unique_rows = np.unique(triples[row_columns].to_numpy(dtype=np.int64).ravel())
    with h5py.File(args.data, "r") as handle:
        if unique_rows.min() < 0 or unique_rows.max() >= len(handle["spectrum"]):
            raise RuntimeError("edge triples contain invalid HDF5 rows")
        spectra = take(handle["spectrum"], unique_rows)
        precursor = take(handle["precursor_mz"], unique_rows).astype(float)
    if np.any(~np.isfinite(precursor)) or np.any(precursor <= 0):
        raise RuntimeError("author DP input has invalid precursor masses")

    long_parts: list[pd.DataFrame] = []
    for row, raw, parent_mz in zip(unique_rows, spectra, precursor, strict=True):
        raw = np.asarray(raw, dtype=float)
        valid = np.isfinite(raw).all(axis=0) & (raw[0] > 0) & (raw[1] > 0)
        fragments = raw[:, valid].T
        if not len(fragments):
            raise RuntimeError(f"author DP input row has no valid fragments: {row}")
        fragments = fragments[np.argsort(fragments[:, 0], kind="mergesort")]
        long_parts.append(
            pd.DataFrame(
                {
                    "hdf5_row": int(row),
                    "precursor_mz": float(parent_mz),
                    "fragment_mz": fragments[:, 0],
                    "intensity": fragments[:, 1],
                }
            )
        )
    spectra_long = pd.concat(long_parts, ignore_index=True)
    pairs = triples[row_columns].copy()
    pairs.insert(0, "triple_index", np.arange(len(pairs), dtype=np.int64))

    args.output_dir.mkdir(parents=True)
    spectra_path = args.output_dir / "spectra_long.csv.gz"
    pairs_path = args.output_dir / "pairs.csv.gz"
    spectra_long.to_csv(spectra_path, index=False, compression="gzip", float_format="%.12g")
    pairs.to_csv(pairs_path, index=False, compression="gzip")
    report = {
        "status": "kgmn_edge_calibration_exact_author_input_frozen",
        "formal": True,
        "triples": int(len(pairs)),
        "unique_spectra": int(len(unique_rows)),
        "fragment_rows": int(len(spectra_long)),
        "contracts": {
            "all_fragments_exported": True,
            "precursor_is_metadata_not_manual_fragment": True,
            "author_score_not_computed_in_python": True,
            "triple_order_preserved": True,
        },
        "provenance": {
            "manifest_report_sha256": sha256(report_path),
            "triples_sha256": sha256(triples_path),
            "hdf5_sha256": sha256(args.data),
            "spectra_long_sha256": sha256(spectra_path),
            "pairs_sha256": sha256(pairs_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Execution bridge for exact author R scoring; no edge or annotation result.",
    }
    output_report = args.output_dir / "report.json"
    output_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
