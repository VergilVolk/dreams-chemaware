"""Build a small real-spectrum reference library for local UI integration tests.

The output is deliberately marked as smoke-only. It validates engineering
plumbing and must never be used for performance or annotation-rate claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("spectrum", "precursor_mz", "INCHIKEY", "FORMULA", "smiles", "adduct", "IDENTIFIER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.source, "r") as source:
        missing = set(FIELDS) - set(source.keys())
        if missing:
            raise KeyError(f"source HDF5 missing {sorted(missing)}")
        total = len(source["spectrum"])
        count = min(int(args.rows), total)
        rows = np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)
        with h5py.File(args.output, "w") as target:
            for key in FIELDS:
                dataset = source[key]
                values = dataset[rows]
                target.create_dataset(key, data=values, dtype=dataset.dtype)
            target.attrs["status"] = "showspace_real_spectrum_smoke_reference"
            target.attrs["source"] = str(args.source.resolve())
            target.attrs["seed"] = args.seed
            target.attrs["claim_limit"] = "engineering smoke test only; no retrieval-performance claim"

        query_position = 0
        spectrum = np.asarray(source["spectrum"][int(rows[query_position])], dtype=float)
        mz, intensity = spectrum[0], spectrum[1]
        keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        query = {
            "library_position": query_position,
            "source_row": int(rows[query_position]),
            "precursor_mz": float(source["precursor_mz"][int(rows[query_position])]),
            "adduct": decode(source["adduct"][int(rows[query_position])]),
            "truth_ik14": decode(source["INCHIKEY"][int(rows[query_position])])[:14],
            "peaks": np.column_stack([mz[keep], intensity[keep]]).tolist(),
            "claim_limit": "self-retrieval plumbing test only",
        }
    query_path = args.output.with_suffix(".query.json")
    query_path.write_text(json.dumps(query, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "showspace_smoke_reference_complete", "rows": count,
        "hdf5": str(args.output), "query": str(query_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
