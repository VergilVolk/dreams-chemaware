#!/usr/bin/env python
"""Build the label-free GNPS cache required by frozen ChemAware V2.

Only spectra that occur in either sealed GNPS panel are materialised.  Peak
selection and intensity scaling exactly reuse the established MassSpecGym
ChemAware preprocessing; no candidate label is opened by this script.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from encode_gnps_gold_silver_10ppm_checkpoint import load_selected_spectra, required_rows
from gnps_pair_score_cache import benchmark_fingerprint, sha256_file
from pilot_paired_layer_cka import preprocess_spectrum


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--official-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.n_highest_peaks != 100:
        raise ValueError("frozen ChemAware V2 requires exactly 100 peaks")
    rows = required_rows(args.benchmark)
    spectra = load_selected_spectra(args.benchmark / "spectra.mgf", rows)
    with np.load(args.official_embeddings, allow_pickle=False) as body:
        embedding_rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if not np.array_equal(embedding_rows, rows):
        raise RuntimeError("official embedding rows do not match the sealed GNPS row order")
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise RuntimeError("official embedding cache is malformed")

    shape = (len(rows), args.n_highest_peaks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gnps_chem_kernel.", dir=args.output.parent))
    try:
        mz = np.lib.format.open_memmap(
            staging / "mz_f32.npy", mode="w+", dtype=np.float32, shape=shape,
        )
        intensity = np.lib.format.open_memmap(
            staging / "intensity_f32.npy", mode="w+", dtype=np.float32, shape=shape,
        )
        valid = np.lib.format.open_memmap(
            staging / "valid.npy", mode="w+", dtype=bool, shape=shape,
        )
        precursor = np.empty(len(rows), dtype=np.float32)
        for index, (raw, precursor_mz) in enumerate(spectra):
            processed = preprocess_spectrum(raw, precursor_mz, args.n_highest_peaks).numpy()
            peaks = processed[1:]
            mz[index] = peaks[:, 0]
            intensity[index] = peaks[:, 1]
            valid[index] = peaks[:, 0] > 0
            precursor[index] = np.float32(precursor_mz)
        for array in (mz, intensity, valid):
            array.flush()
        del mz, intensity, valid
        np.save(staging / "rows.npy", rows, allow_pickle=False)
        np.save(staging / "precursor_mz_f32.npy", precursor, allow_pickle=False)
        np.save(
            staging / "official_embeddings_f32.npy",
            np.ascontiguousarray(embeddings), allow_pickle=False,
        )
        report = {
            "status": "GNPS_CHEMAWARE_KERNEL_CACHE_COMPLETE",
            "schema": "gnps_chemaware_kernel_cache_v1",
            "spectra": int(len(rows)),
            "n_highest_peaks": args.n_highest_peaks,
            "labels_opened": False,
            "benchmark": benchmark_fingerprint(args.benchmark),
            "provenance": {
                "official_embeddings_sha256": sha256_file(args.official_embeddings),
                "spectra_mgf_sha256": sha256_file(args.benchmark / "spectra.mgf"),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
