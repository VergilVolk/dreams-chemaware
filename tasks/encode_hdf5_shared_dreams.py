"""Encode ordered HDF5 MS2 files with one frozen shared DreaMS encoder.

The manifest is kept in exact file/row order so an official and an experimental
cache can be compared spectrum by spectrum.  Existing outputs are never
overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from pilot_paired_layer_cka import SpectrumRows  # noqa: E402
from shared_dreams_inference import load_inference_model, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--shared-encoder-checkpoint", type=Path)
    parser.add_argument(
        "--reference-manifest", type=Path,
        help="Reorder HDF5 inputs to the exact first-occurrence file order in this manifest.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def order_hdf5_by_manifest(paths: list[Path], manifest_path: Path) -> list[Path]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    by_stem = {path.stem: path for path in paths}
    if len(by_stem) != len(paths):
        raise RuntimeError("duplicate HDF5 file stems")
    reference = pd.read_csv(manifest_path, usecols=["file_name"])
    order = reference.file_name.astype(str).drop_duplicates().tolist()
    missing = sorted(set(order) - set(by_stem))
    extra = sorted(set(by_stem) - set(order))
    if missing or extra:
        raise RuntimeError(
            f"HDF5/reference-manifest file mismatch: missing={missing}, extra={extra}"
        )
    return [by_stem[name] for name in order]


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in args.hdf5:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.reference_manifest is not None:
        args.hdf5 = order_hdf5_by_manifest(list(args.hdf5), args.reference_manifest)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    counts = []
    for path in args.hdf5:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["spectrum"].shape[0]))
    total = int(sum(counts))
    if total < 1:
        raise RuntimeError("no spectra in HDF5 inputs")

    device = torch.device(args.device)
    model, model_metadata = load_inference_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks, args.shared_encoder_checkpoint,
    )
    dimension = int(model.head.out_features)
    dtype = next(model.parameters()).dtype
    embedding_path = out / "embeddings.npy"
    embeddings = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(total, dimension)
    )
    manifest_parts = []
    offset = 0
    started = time.time()
    with torch.inference_mode():
        for file_index, (path, count) in enumerate(zip(args.hdf5, counts), 1):
            with h5py.File(path, "r") as handle:
                metadata = {
                    column: np.asarray(handle[column][:])
                    for column in ("scan_number", "precursor_mz", "charge", "RT")
                }
            rows = np.arange(count, dtype=np.int64)
            loader = DataLoader(
                SpectrumRows(path, rows, args.n_highest_peaks),
                batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            cursor = offset
            for batch in loader:
                batch = batch.to(device=device, dtype=dtype)
                values = model(batch).float().cpu().numpy()
                embeddings[cursor:cursor + len(values)] = values
                cursor += len(values)
            if cursor != offset + count:
                raise RuntimeError(f"row-count drift for {path}")
            manifest_parts.append(pd.DataFrame({
                "file_name": path.stem,
                "scan_number": metadata["scan_number"].astype(np.int64),
                "precursor_mz": metadata["precursor_mz"].astype(np.float64),
                "charge": metadata["charge"].astype(np.int64),
                "RT": metadata["RT"].astype(np.float64),
                "row_in_file": rows,
            }))
            offset = cursor
            print(f"[query embedding] {file_index}/{len(args.hdf5)} files; {offset:,}/{total:,}", flush=True)
    embeddings.flush()
    manifest = pd.concat(manifest_parts, ignore_index=True)
    if len(manifest) != total:
        raise RuntimeError("manifest/embedding length mismatch")
    manifest_path = out / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    probe = np.load(embedding_path, mmap_mode="r")
    if probe.shape != (total, dimension) or not np.isfinite(probe[: min(total, 1000)]).all():
        raise RuntimeError("invalid shared-embedding cache")
    report = {
        "status": "ordered_hdf5_shared_dreams_embeddings_complete",
        "spectra": total,
        "embedding_dim": dimension,
        "model": model_metadata,
        "files": [str(path.resolve()) for path in args.hdf5],
        "input_file_sizes": {str(path): int(path.stat().st_size) for path in args.hdf5},
        "manifest_sha256": sha256_file(manifest_path),
        "reference_manifest": (
            str(args.reference_manifest.resolve()) if args.reference_manifest is not None else None
        ),
        "reference_manifest_sha256": (
            sha256_file(args.reference_manifest) if args.reference_manifest is not None else None
        ),
        "elapsed_seconds": time.time() - started,
        "contract": "one frozen shared encoder; clean spectra only; exact HDF5 row order",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
