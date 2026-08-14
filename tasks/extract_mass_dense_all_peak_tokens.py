"""Extract every valid layer-7 peak token for the mass-dense split.

This is the spectral-first input to fragmentation-factor discovery.  It uses
only the official DreaMS embedding checkpoint, preserves the existing
molecule-disjoint discovery/confirmation split, and keeps a direct mapping
from every retained token to its preprocessed (m/z, intensity) peak.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import extract_mass_dense_factor_activations as dense
import pilot_multilevel_factor_activations as multi


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=multi.DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=dense.DEFAULT_MANIFEST)
    parser.add_argument(
        "--split", choices=("discovery", "confirmation"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument(
        "--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL
    )
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_rows(handle: h5py.File, key: str, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary row order while satisfying h5py's sorted-index rule."""
    order = np.argsort(rows)
    inverse = np.argsort(order)
    dataset = handle[key]
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return dataset.asstr()[rows[order]][inverse]
    return np.asarray(dataset[rows[order]])[inverse]


def spectrum_metadata(data: Path, rows: np.ndarray) -> tuple[list[dict], list[str]]:
    requested = [
        "IDENTIFIER",
        "INCHIKEY",
        "smiles",
        "FORMULA",
        "PRECURSOR_FORMULA",
        "precursor_mz",
        "adduct",
        "INSTRUMENT_TYPE",
        "COLLISION_ENERGY",
        "fold",
    ]
    with h5py.File(data, "r") as handle:
        available = [key for key in requested if key in handle]
        values = {key: read_rows(handle, key, rows) for key in available}
    records: list[dict] = []
    for i, row in enumerate(rows):
        record = {"spectrum_index": i, "hdf5_row": int(row)}
        for key in available:
            value = values[key][i]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            record[key] = value
        if "INCHIKEY" in record and record["INCHIKEY"]:
            record["ik14"] = str(record["INCHIKEY"])[:14]
        records.append(record)
    return records, available


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable")
    if args.n_highest_peaks < 3 or args.n_highest_peaks > 128:
        raise ValueError("--n-highest-peaks must be in [3, 128]")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    units = dense.select_units(manifest, args.split, args.max_units)
    rows, pairs = dense.load_metadata(args.data, units)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"split={args.split}; molecules={len(units)}; spectra={len(rows)}; "
        f"peak slots={args.n_highest_peaks}",
        flush=True,
    )

    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(args.device)

    # The slim official checkpoint intentionally omits model arguments.  The
    # audited raw package supplies architecture arguments only; its weights are
    # never used for this extraction.
    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(
        args.official_checkpoint, map_location="cpu"
    )
    if multi.checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("--raw-checkpoint must use raw SSL format")
    if multi.checkpoint_kind(official_package) not in (
        "official_embedding",
        "official_embedding_slim",
    ):
        raise ValueError("--official-checkpoint must use official embedding format")
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    del raw_package, official_package
    gc.collect()

    precursor, peaks, peak_indices, peak_mask, peak_values, diagnostics = (
        multi.extract_multilevel(
            model,
            loader,
            device,
            layer_numbers=[7],
            peak_tokens=args.n_highest_peaks,
        )
    )
    del model
    gc.collect()

    # Remove the single layer dimension.  Padding stays explicit in peak_mask;
    # consumers must never treat padded token rows as observations.
    precursor = precursor[:, 0]
    peaks = peaks[:, 0]
    valid_counts = peak_mask.sum(axis=1)
    if np.any(valid_counts == 0):
        raise RuntimeError("At least one spectrum has no valid peak after preprocessing")
    if not np.all(peak_values[peak_mask, 0] > 0):
        raise RuntimeError("A retained peak has non-positive m/z")
    if not np.all(peak_values[peak_mask, 1] > 0):
        raise RuntimeError("A retained peak has non-positive intensity")
    for row in range(len(rows)):
        idx = peak_indices[row, peak_mask[row]]
        if len(idx) and (np.any(np.diff(idx) <= 0) or idx.min() < 1):
            raise RuntimeError(f"Peak-token indices are not strictly ordered at {row}")

    records, metadata_fields = spectrum_metadata(args.data, rows)
    np.save(args.output_dir / "rows.npy", rows)
    np.save(args.output_dir / "official_precursor.npy", precursor)
    np.save(args.output_dir / "official_peak.npy", peaks)
    np.save(args.output_dir / "peak_indices.npy", peak_indices)
    np.save(args.output_dir / "peak_mask.npy", peak_mask)
    np.save(args.output_dir / "peak_values.npy", peak_values)
    (args.output_dir / "pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "spectra.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "status": "mass_dense_all_peak_tokens",
        "scientific_role": (
            "Spectral-first discovery input; existing fragmentation rules were "
            "not used to select spectra, peaks, or tokens."
        ),
        "config": {
            "data": str(args.data),
            "manifest": str(args.manifest),
            "split": args.split,
            "checkpoint": str(args.official_checkpoint),
            "layer": 7,
            "n_molecules": len(units),
            "n_spectra": len(rows),
            "n_highest_peaks": args.n_highest_peaks,
            "max_units": args.max_units,
            "metadata_fields": metadata_fields,
        },
        "alignment_audit": {
            "valid_peak_tokens": int(peak_mask.sum()),
            "valid_peaks_per_spectrum_min": int(valid_counts.min()),
            "valid_peaks_per_spectrum_median": float(np.median(valid_counts)),
            "valid_peaks_per_spectrum_max": int(valid_counts.max()),
            "all_retained_mz_positive": True,
            "all_retained_intensity_positive": True,
            "all_peak_indices_strictly_ordered": True,
            "peak_token_shape": list(peaks.shape),
            "peak_value_shape": list(peak_values.shape),
            "precursor_token_shape": list(precursor.shape),
        },
        "model_diagnostics": diagnostics,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["alignment_audit"], indent=2), flush=True)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
