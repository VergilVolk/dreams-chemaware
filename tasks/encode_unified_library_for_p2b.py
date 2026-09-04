"""Encode a unified MGF library and cache the peak arrays needed by P2b.

The cache is deliberately model- and order-locked: the embeddings, peak arrays
and manifest are emitted in the exact MGF record order.  It is an execution
cache, not a learned model.  Existing complete outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from pilot_paired_layer_cka import (  # noqa: E402
    preprocess_spectrum,
)
from shared_dreams_inference import load_inference_model, sha256_file  # noqa: E402


def iter_mgf(path: Path):
    fields = None
    peaks: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                fields, peaks = {}, []
            elif line == "END IONS":
                if fields is not None and peaks and "PEPMASS" in fields:
                    try:
                        precursor = float(fields["PEPMASS"].split()[0])
                    except (ValueError, IndexError):
                        fields = None
                        continue
                    yield fields, precursor, np.asarray(peaks, dtype=np.float32).T
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
            elif fields is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass


def count_records(path: Path) -> int:
    return sum(1 for _ in iter_mgf(path))


def top_peak_array(raw: np.ndarray, n_peaks: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.isfinite(raw).all(axis=0) & (raw[0] > 0) & (raw[1] > 0)
    raw = raw[:, valid]
    if raw.shape[1] > n_peaks:
        keep = np.argpartition(raw[1], -n_peaks)[-n_peaks:]
        raw = raw[:, keep]
    raw = raw[:, np.argsort(raw[0], kind="stable")]
    out = np.zeros((2, n_peaks), dtype=np.float32)
    out[:, : raw.shape[1]] = raw
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--shared-encoder-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-peaks", type=int, default=100)
    parser.add_argument("--cache-peaks", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mgf.exists():
        raise FileNotFoundError(args.mgf)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = args.output_dir.resolve()
    report_path = out / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = [out / name for name in ("manifest.csv", "embeddings.npy", "spectra.npy")]
        if report.get("status") == "unified_library_p2b_cache_complete" and all(path.exists() for path in expected):
            raise FileExistsError(f"refusing to overwrite complete cache: {out}")
    out.mkdir(parents=True, exist_ok=True)

    total = count_records(args.mgf)
    if args.limit:
        total = min(total, args.limit)
    if total < 1:
        raise RuntimeError("MGF contains no valid spectra")
    print(f"[library] valid records={total:,}", flush=True)

    device = torch.device(args.device)
    model, model_metadata = load_inference_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.model_peaks, args.shared_encoder_checkpoint,
    )
    dtype = next(model.parameters()).dtype
    dimension = int(model.head.out_features)

    embedding_path = out / "embeddings.npy"
    spectra_path = out / "spectra.npy"
    manifest_path = out / "manifest.csv"
    embeddings = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(total, dimension)
    )
    spectra = np.lib.format.open_memmap(
        spectra_path, mode="w+", dtype=np.float32, shape=(total, 2, args.cache_peaks)
    )
    columns = ["library_row", "smiles", "inchikey", "ik14", "name", "source", "adduct", "precursor_mz"]
    batch_tensors: list[torch.Tensor] = []
    batch_rows: list[int] = []
    written = 0

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_rows
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device=device, dtype=dtype)
        with torch.inference_mode():
            values = model(batch).float().cpu().numpy()
        embeddings[np.asarray(batch_rows, dtype=np.int64)] = values
        batch_tensors, batch_rows = [], []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=columns)
        writer.writeheader()
        for fields, precursor, raw in iter_mgf(args.mgf):
            if written >= total:
                break
            cached = top_peak_array(raw, args.cache_peaks)
            spectra[written] = cached
            inchikey = fields.get("INCHIKEY", "")
            writer.writerow({
                "library_row": written,
                "smiles": fields.get("SMILES", ""),
                "inchikey": inchikey,
                "ik14": inchikey[:14],
                "name": fields.get("NAME", ""),
                "source": fields.get("SOURCE", ""),
                "adduct": fields.get("ADDUCT", ""),
                "precursor_mz": precursor,
            })
            batch_tensors.append(preprocess_spectrum(cached, precursor, args.model_peaks))
            batch_rows.append(written)
            written += 1
            if len(batch_tensors) >= args.batch_size:
                flush_batch()
            if written % 3200 == 0 or written == total:
                print(f"[library] {written:,}/{total:,}", flush=True)
        flush_batch()
    embeddings.flush()
    spectra.flush()
    if written != total:
        raise RuntimeError(f"MGF count drift: expected {total}, wrote {written}")
    probe = np.load(embedding_path, mmap_mode="r")
    if probe.shape != (total, dimension) or not np.isfinite(probe[: min(total, 1000)]).all():
        raise RuntimeError("invalid embedding cache")
    report = {
        "status": "unified_library_p2b_cache_complete",
        "spectra": total,
        "embedding_dim": dimension,
        "model": model_metadata,
        "cached_peaks": args.cache_peaks,
        "model_peaks": args.model_peaks,
        "mgf": str(args.mgf.resolve()),
        "provenance": {
            "mgf_sha256": sha256_file(args.mgf),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "contract": "one frozen shared DreaMS encoder and ordered raw-peak execution cache; no labels or phenotype",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
