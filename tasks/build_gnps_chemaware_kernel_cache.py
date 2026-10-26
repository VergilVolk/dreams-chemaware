#!/usr/bin/env python
"""Build the label-free GNPS cache required by frozen ChemAware V2.

Only spectra that occur in either sealed GNPS panel are materialised.  Peak
selection and intensity scaling exactly reuse the established MassSpecGym
ChemAware preprocessing; no candidate label is opened by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
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
    parser.add_argument(
        "--staging-dir", type=Path,
        help=(
            "Optional node-local directory used while materialising the cache. "
            "The completed files are copied to an atomic staging directory next "
            "to --output before publication."
        ),
    )
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument(
        "--panels", nargs="+", default=["identity_disjoint", "formula_disjoint"],
    )
    parser.add_argument("--expected-selected-rows", type=int, default=52871)
    parser.add_argument("--expected-records", type=int, default=329607)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.n_highest_peaks != 100:
        raise ValueError("frozen ChemAware V2 requires exactly 100 peaks")
    rows = required_rows(
        args.benchmark, args.expected_selected_rows, tuple(args.panels),
    )
    spectra = load_selected_spectra(
        args.benchmark / "spectra.mgf", rows,
        expected_records=args.expected_records,
    )
    with np.load(args.official_embeddings, allow_pickle=False) as body:
        embedding_rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(embedding_rows):
        raise RuntimeError("official embedding cache is malformed")
    if len(np.unique(embedding_rows)) != len(embedding_rows):
        raise RuntimeError("official embedding cache contains duplicate rows")
    row_to_embedding = {int(row): index for index, row in enumerate(embedding_rows)}
    missing = [int(row) for row in rows if int(row) not in row_to_embedding]
    if missing:
        raise RuntimeError(
            f"official embedding cache misses {len(missing)} selected rows"
        )
    embeddings = embeddings[[row_to_embedding[int(row)] for row in rows]]

    shape = (len(rows), args.n_highest_peaks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = args.staging_dir or args.output.parent
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.build.", dir=staging_root,
    ))
    publish_parent: Path | None = None
    try:
        # These three arrays occupy < 100 MiB for Enveda-180.  Keeping them in
        # ordinary RAM avoids writable mmap pages on the shared filesystem;
        # a failed mmap backing store terminates the process with SIGBUS and
        # bypasses Python exception handling entirely.
        mz = np.empty(shape, dtype=np.float32)
        intensity = np.empty(shape, dtype=np.float32)
        valid = np.empty(shape, dtype=bool)
        precursor = np.empty(len(rows), dtype=np.float32)
        started = time.time()
        for index, (raw, precursor_mz) in enumerate(spectra):
            processed = preprocess_spectrum(raw, precursor_mz, args.n_highest_peaks).numpy()
            peaks = processed[1:]
            mz[index] = peaks[:, 0]
            intensity[index] = peaks[:, 1]
            valid[index] = peaks[:, 0] > 0
            precursor[index] = np.float32(precursor_mz)
            if (index + 1) == len(rows) or (index + 1) % 10000 == 0:
                print(
                    f"[chem-token-cache] {index + 1:,}/{len(rows):,}; "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )
        np.save(staging / "mz_f32.npy", mz, allow_pickle=False)
        np.save(staging / "intensity_f32.npy", intensity, allow_pickle=False)
        np.save(staging / "valid.npy", valid, allow_pickle=False)
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
            "panels": list(args.panels),
            "benchmark": benchmark_fingerprint(args.benchmark),
            "materialization": {
                "writable_memmap": False,
                "node_local_staging_requested": args.staging_dir is not None,
            },
            "provenance": {
                "official_embeddings_sha256": sha256_file(args.official_embeddings),
                "spectra_mgf_sha256": sha256_file(args.benchmark / "spectra.mgf"),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if os.stat(staging).st_dev == os.stat(args.output.parent).st_dev:
            staging.replace(args.output)
        else:
            # Copy complete files from node-local storage into a hidden sibling,
            # then rename on the destination filesystem.  Consumers can never
            # observe a partial cache.
            publish_parent = Path(tempfile.mkdtemp(
                prefix=f".{args.output.name}.publish.", dir=args.output.parent,
            ))
            publish_payload = publish_parent / "payload"
            shutil.copytree(staging, publish_payload)
            publish_payload.replace(args.output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if publish_parent is not None:
            shutil.rmtree(publish_parent, ignore_errors=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
