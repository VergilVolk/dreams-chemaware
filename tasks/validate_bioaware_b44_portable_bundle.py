#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b44_portable_bundle_frozen":
        raise RuntimeError("portable bundle status mismatch")
    bad = [
        relative for relative, expected in report["files"].items()
        if not (args.input_dir / relative).is_file()
        or sha256(args.input_dir / relative) != expected
    ]
    if bad:
        raise RuntimeError(f"portable bundle hash mismatch: {bad}")
    queries = pd.read_csv(args.input_dir / "panel/queries.csv.gz")
    references = pd.read_csv(args.input_dir / "panel/candidate_references.csv.gz")
    expected_rows = np.sort(np.unique(np.concatenate((
        queries["query_hdf5_row"].to_numpy(np.int64),
        references["reference_hdf5_row"].to_numpy(np.int64),
    ))))
    with np.load(args.input_dir / "preprocessed_spectra.npz", allow_pickle=False) as payload:
        rows = payload["hdf5_rows"].astype(np.int64, copy=False)
        spectra = payload["spectra"]
        source_hash = str(payload["source_massbank_sha256"].item())
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError("portable bundle row universe mismatch")
    if spectra.shape != (len(rows), int(report["preprocessed_spectrum_length"]), 2):
        raise RuntimeError(f"portable spectrum shape mismatch: {spectra.shape}")
    if not np.isfinite(spectra).all():
        raise RuntimeError("portable spectra contain non-finite values")
    if source_hash != report["source_massbank_hdf5_sha256"]:
        raise RuntimeError("portable source MassBank provenance mismatch")
    if report.get("server_loads_joblib_model") is not False:
        raise RuntimeError("portable bundle unexpectedly requires server-side joblib loading")
    if report["portable_model_replay"]["maximum_probability_error"] > 1e-14:
        raise RuntimeError("portable model probability replay tolerance failed")
    print(f"[BioAware B44 portable bundle] PASS rows={len(rows):,} size={sum(p.stat().st_size for p in args.input_dir.rglob('*') if p.is_file()) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
