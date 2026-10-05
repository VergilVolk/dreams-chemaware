#!/usr/bin/env python
"""Encode the frozen GNPS Gold/Silver benchmark with one DreaMS checkpoint.

Only rows used by either sealed retrieval panel are encoded.  The input path is
the exact 100-peak, precursor-intensity-1.1 pipeline used by native DreaMS Noise
evaluation; no benchmark label is visible to the encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from build_gnps_gold_silver_10ppm_benchmark import iter_mgf


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label", default="checkpoint")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def required_rows(benchmark: Path) -> np.ndarray:
    rows: list[np.ndarray] = []
    for panel_name in ("identity_disjoint", "formula_disjoint"):
        with np.load(benchmark / f"panel_{panel_name}.npz", allow_pickle=False) as body:
            rows.extend((
                np.asarray(body["query_row"], dtype=np.int64),
                np.asarray(body["candidate_row"], dtype=np.int64),
            ))
    selected = np.unique(np.concatenate(rows))
    # The identity panel uses 52,854 spectra.  The formula-disjoint panel adds
    # 17 rows that are not present there, so the two-panel union is 52,871.
    if len(selected) != 52871:
        raise RuntimeError(f"sealed GNPS used-row count drifted: {len(selected)} != 52871")
    return selected


def load_selected_spectra(
    mgf: Path,
    selected: np.ndarray,
    *,
    expected_records: int = 329607,
) -> list[tuple[np.ndarray, float]]:
    selected_set = set(map(int, selected))
    spectra: dict[int, tuple[np.ndarray, float]] = {}
    records = 0
    started = time.time()
    for row, (fields, peaks) in enumerate(iter_mgf(mgf)):
        records = row + 1
        if row not in selected_set:
            continue
        precursor_text = fields.get("PEPMASS", "").split()
        if not precursor_text:
            raise RuntimeError(f"selected MGF row {row} has no PEPMASS")
        peak_array = np.asarray(peaks, dtype=np.float32)
        if peak_array.ndim != 2 or peak_array.shape[1] != 2 or len(peak_array) == 0:
            raise RuntimeError(f"selected MGF row {row} has malformed peaks")
        spectra[row] = (peak_array.T, float(precursor_text[0]))
    if records != expected_records:
        raise RuntimeError(
            f"sealed GNPS MGF row count drifted: {records} != {expected_records}"
        )
    missing = selected_set.difference(spectra)
    if missing:
        raise RuntimeError(f"GNPS MGF misses {len(missing)} selected rows")
    print(
        f"[{len(spectra):,} selected spectra loaded from {records:,} MGF rows; "
        f"{time.time() - started:.0f}s]",
        flush=True,
    )
    return [spectra[int(row)] for row in selected]


@torch.inference_mode()
def encode(
    model: torch.nn.Module,
    selected: np.ndarray,
    spectra: list[tuple[np.ndarray, float]],
    preprocessor: Any,
    batch_size: int,
    label: str,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output = np.empty((len(selected), 1024), dtype=np.float32)
    model.eval()
    device = torch.device("cuda")
    started = time.time()
    for left in range(0, len(selected), batch_size):
        right = min(left + batch_size, len(selected))
        batch = []
        for peaks, precursor_mz in spectra[left:right]:
            batch.append(preprocessor(
                peaks,
                prec_mz=float(precursor_mz),
                high_form=False,
            ))
        tensor = torch.from_numpy(np.stack(batch).astype(np.float32)).to(device)
        output[left:right] = F.normalize(model(tensor).float(), dim=1).cpu().numpy()
        if right == len(selected) or right % (batch_size * 20) == 0:
            print(
                f"[{label}-GNPS-encode] {right:,}/{len(selected):,}; "
                f"{time.time() - started:.0f}s",
                flush=True,
            )
    norms = np.linalg.norm(output, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("GNPS embeddings are not finite unit vectors")
    return output


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("GNPS DreaMS encoding requires an allocated GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.benchmark / "spectra.mgf",
        args.benchmark / "panel_identity_disjoint.npz",
        args.benchmark / "panel_formula_disjoint.npz",
        args.checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep CPU-only row/format tests importable on machines without the full
    # DreaMS runtime.  These imports are required only for actual GPU encoding.
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    from train_e1_identity import load_base_model

    selected = required_rows(args.benchmark)
    spectra = load_selected_spectra(args.benchmark / "spectra.mgf", selected)
    model, checkpoint_kind = load_base_model(
        args.checkpoint,
        args.architecture_checkpoint,
        torch.device("cuda"),
        100,
    )
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(),
        prec_intens=1.1,
        n_highest_peaks=100,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0,
        mz_shift_aug_max=0,
    )
    embeddings = encode(
        model, selected, spectra, preprocessor, args.batch_size, args.label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".npz", dir=args.output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, rows=selected, embeddings=embeddings)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "gnps_gold_silver_10ppm_checkpoint_encoding_complete",
        "label": args.label,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "rows": int(len(selected)),
        "embedding_dimension": int(embeddings.shape[1]),
        "native_preprocessing": {
            "n_highest_peaks": 100,
            "precursor_intensity": 1.1,
            "precision": 32,
            "augmentation": False,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
