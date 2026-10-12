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
import pandas as pd
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
    parser.add_argument("--reuse-benchmark", type=Path)
    parser.add_argument("--reuse-embeddings", type=Path)
    parser.add_argument(
        "--panels", nargs="+", default=["identity_disjoint", "formula_disjoint"],
    )
    parser.add_argument(
        "--panel-file", action="append", default=[], metavar="NAME=PATH",
        help="Override a panel path when it is stored outside --benchmark.",
    )
    parser.add_argument(
        "--expected-selected-rows", type=int, default=52871,
        help="Set to 0 for another frozen benchmark whose selected-row count is report-defined.",
    )
    parser.add_argument(
        "--expected-records", type=int, default=329607,
        help="Set to 0 to accept the complete record count of another curated benchmark.",
    )
    return parser.parse_args()


def reuse_by_source_row(
    selected: np.ndarray,
    benchmark: Path,
    reuse_benchmark: Path,
    reuse_embeddings: Path,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a prior label-blind cache onto a new panel by immutable source row."""
    old_benchmark_report = json.loads(
        (reuse_benchmark / "report.json").read_text(encoding="utf-8")
    )
    new_benchmark_report = json.loads(
        (benchmark / "report.json").read_text(encoding="utf-8")
    )
    old_source_sha = old_benchmark_report.get("leakage_guard", {}).get("source_mgf_sha256")
    new_source_sha = new_benchmark_report.get("leakage_guard", {}).get("source_mgf_sha256")
    if not old_source_sha or old_source_sha != new_source_sha:
        raise RuntimeError("old and new benchmarks are not bound to the same source MGF")
    reuse_report_path = reuse_embeddings.with_suffix(".json")
    if not reuse_report_path.is_file():
        raise FileNotFoundError(reuse_report_path)
    reuse_report = json.loads(reuse_report_path.read_text(encoding="utf-8"))
    if reuse_report.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("reuse cache checkpoint differs from requested checkpoint")
    preprocessing = reuse_report.get("native_preprocessing", {})
    if preprocessing != {
        "n_highest_peaks": 100,
        "precursor_intensity": 1.1,
        "precision": 32,
        "augmentation": False,
    }:
        raise RuntimeError("reuse cache preprocessing differs from native encoding contract")

    with np.load(reuse_embeddings, allow_pickle=False) as body:
        old_rows = np.asarray(body["rows"], dtype=np.int64)
        old_embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if old_embeddings.shape != (len(old_rows), 1024):
        raise RuntimeError("reuse embedding cache has invalid shape")
    if len(np.unique(old_rows)) != len(old_rows):
        raise RuntimeError("reuse embedding cache has duplicate local rows")

    old_manifest = pd.read_csv(
        reuse_benchmark / "manifest.csv.gz", usecols=["row", "source_row"],
    ).set_index("row")
    new_manifest = pd.read_csv(
        benchmark / "manifest.csv.gz", usecols=["row", "source_row"],
    ).set_index("row")
    try:
        old_sources = old_manifest.loc[old_rows, "source_row"].to_numpy(np.int64)
        new_sources = new_manifest.loc[selected, "source_row"].to_numpy(np.int64)
    except KeyError as exc:
        raise RuntimeError("embedding rows are absent from their benchmark manifest") from exc
    if len(np.unique(old_sources)) != len(old_sources):
        raise RuntimeError("reuse benchmark has duplicate immutable source rows")
    source_to_old = {int(source): index for index, source in enumerate(old_sources)}
    output = np.full((len(selected), 1024), np.nan, dtype=np.float32)
    reused = np.zeros(len(selected), dtype=bool)
    for index, source in enumerate(new_sources):
        old_index = source_to_old.get(int(source))
        if old_index is not None:
            output[index] = old_embeddings[old_index]
            reused[index] = True
    if reused.any():
        norms = np.linalg.norm(output[reused], axis=1)
        if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
            raise RuntimeError("reused embeddings are not finite unit vectors")
    return output, reused


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def required_rows(
    benchmark_or_panel_paths: Path | dict[str, Path],
    expected_selected_rows: int = 52871,
    panels: tuple[str, ...] = ("identity_disjoint", "formula_disjoint"),
) -> np.ndarray:
    if isinstance(benchmark_or_panel_paths, dict):
        panel_paths = benchmark_or_panel_paths
    else:
        panel_paths = {
            name: benchmark_or_panel_paths / f"panel_{name}.npz" for name in panels
        }
    rows: list[np.ndarray] = []
    for panel_name in panels:
        with np.load(panel_paths[panel_name], allow_pickle=False) as body:
            rows.extend((
                np.asarray(body["query_row"], dtype=np.int64),
                np.asarray(body["candidate_row"], dtype=np.int64),
            ))
    selected = np.unique(np.concatenate(rows))
    # The identity panel uses 52,854 spectra.  The formula-disjoint panel adds
    # 17 rows that are not present there, so the two-panel union is 52,871.
    if expected_selected_rows and len(selected) != expected_selected_rows:
        raise RuntimeError(
            f"frozen benchmark used-row count drifted: "
            f"{len(selected)} != {expected_selected_rows}"
        )
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
    if expected_records and records != expected_records:
        raise RuntimeError(
            f"frozen benchmark MGF row count drifted: {records} != {expected_records}"
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
    if (args.reuse_benchmark is None) != (args.reuse_embeddings is None):
        raise RuntimeError("--reuse-benchmark and --reuse-embeddings must be provided together")
    if args.output.exists():
        raise FileExistsError(args.output)
    panel_paths = {
        name: args.benchmark / f"panel_{name}.npz" for name in args.panels
    }
    for specification in args.panel_file:
        if "=" not in specification:
            raise ValueError("--panel-file must have NAME=PATH form")
        name, value = specification.split("=", 1)
        if name not in panel_paths:
            raise ValueError(f"--panel-file names an undeclared panel: {name}")
        panel_paths[name] = Path(value)
    for path in (
        args.benchmark / "spectra.mgf",
        args.checkpoint,
        args.architecture_checkpoint,
        *(panel_paths[name] for name in args.panels),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = required_rows(
        panel_paths, args.expected_selected_rows, tuple(args.panels),
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    embeddings = np.full((len(selected), 1024), np.nan, dtype=np.float32)
    reused = np.zeros(len(selected), dtype=bool)
    checkpoint_kind = None
    if args.reuse_embeddings is not None:
        embeddings, reused = reuse_by_source_row(
            selected, args.benchmark, args.reuse_benchmark,
            args.reuse_embeddings, checkpoint_sha256,
        )
        checkpoint_kind = json.loads(
            args.reuse_embeddings.with_suffix(".json").read_text(encoding="utf-8")
        ).get("checkpoint_kind")
        print(
            f"[{args.label}] reused {int(reused.sum()):,}/{len(selected):,} "
            "embeddings by immutable source_row",
            flush=True,
        )
    missing_rows = selected[~reused]
    if len(missing_rows):
        if not torch.cuda.is_available():
            raise RuntimeError("missing DreaMS embeddings require an allocated GPU")
        # Heavy DreaMS imports and model construction are unnecessary when the
        # prior cache covers every source spectrum in the corrected panel.
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.utils.dformats import DataFormatA
        from train_e1_identity import load_base_model

        spectra = load_selected_spectra(
            args.benchmark / "spectra.mgf", missing_rows,
            expected_records=args.expected_records,
        )
        model, checkpoint_kind = load_base_model(
            args.checkpoint, args.architecture_checkpoint,
            torch.device("cuda"), 100,
        )
        preprocessor = SpectrumPreprocessor(
            dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
            spec_entropy_cleaning=False, precision=32,
            mz_shift_aug_p=0, mz_shift_aug_max=0,
        )
        embeddings[~reused] = encode(
            model, missing_rows, spectra, preprocessor, args.batch_size, args.label,
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("merged embedding cache is not finite unit vectors")
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
        "checkpoint_sha256": checkpoint_sha256,
        "rows": int(len(selected)),
        "rows_reused_by_source_row": int(reused.sum()),
        "rows_newly_encoded": int((~reused).sum()),
        "reuse_benchmark": str(args.reuse_benchmark) if args.reuse_benchmark else None,
        "reuse_embeddings": str(args.reuse_embeddings) if args.reuse_embeddings else None,
        "embedding_dimension": int(embeddings.shape[1]),
        "panels": {name: str(panel_paths[name]) for name in args.panels},
        "panel_sha256": {
            name: sha256_file(panel_paths[name]) for name in args.panels
        },
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
