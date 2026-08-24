"""Encode validated MTBLS1905 QC-DDA MS2 spectra with official DreaMS.

Unlike ``MSData.load``, this reads the vendor-converted DDA mzML files
directly.  The study's five files are MS2-focused acquisition windows, for
which the generic MS1+MS2 converter proved inappropriate.  This program is
inference only and records every spectrum-level input used by DreaMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyteomics import mzml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import (  # noqa: E402
    checkpoint_kind, official_backbone_state, official_head_state, torch_load_compat,
)
from pilot_paired_layer_cka import preprocess_spectrum, reconstruct_backbone  # noqa: E402

DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"


def extract_precursor(spec: dict) -> float | None:
    try:
        return float(spec["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def extract_rt(spec: dict) -> float | None:
    try:
        return float(spec["scanList"]["scan"][0]["scan start time"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def read_spectra(paths: list[Path], limit: int | None) -> tuple[list[torch.Tensor], list[dict]]:
    tensors: list[torch.Tensor] = []
    manifest: list[dict] = []
    for path in paths:
        for spec in mzml.read(str(path)):
            if spec.get("ms level") != 2:
                continue
            precursor = extract_precursor(spec)
            mz = np.asarray(spec.get("m/z array", []), dtype=np.float32)
            intensity = np.asarray(spec.get("intensity array", []), dtype=np.float32)
            if precursor is None or len(mz) < 2 or len(mz) != len(intensity):
                continue
            raw = np.vstack((mz, intensity))
            tensors.append(preprocess_spectrum(raw, precursor, 100))
            manifest.append({
                "source_file": path.name, "spectrum_id": str(spec.get("id", "")),
                "precursor_mz": precursor, "rt_min": extract_rt(spec),
                "n_raw_peaks": int(len(mz)),
            })
            if limit is not None and len(tensors) >= limit:
                return tensors, manifest
    return tensors, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/dreams_official"))
    parser.add_argument("--limit", type=int, default=None, help="Use a small deterministic smoke subset when set")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.cpu_threads)
    paths = sorted(args.input_dir.glob("QC*_MSMS_*.mzML"))
    paths = [p for p in paths if not p.name.endswith("270_1050.mzML")]
    if not paths:
        raise FileNotFoundError("No validated QC-DDA mzML files found")
    print("[1/3] reading direct MS2 inputs", flush=True)
    tensors, manifest = read_spectra(paths, args.limit)
    if not tensors:
        raise RuntimeError("No usable MS2 spectra")
    print(f"  usable spectra={len(tensors)}", flush=True)

    print("[2/3] loading official DreaMS", flush=True)
    device = torch.device(args.device)
    raw = torch_load_compat(DEFAULT_RAW, map_location="cpu")
    official = torch_load_compat(DEFAULT_OFFICIAL, map_location="cpu")
    if checkpoint_kind(raw) != "raw_ssl" or checkpoint_kind(official) != "official_embedding_slim":
        raise RuntimeError("Expected project raw SSL and slim official embedding checkpoints")
    model = reconstruct_backbone(raw, official_backbone_state(official), 100, device)
    head = official_head_state(official)
    dtype = next(model.parameters()).dtype
    weight = head["weight"].to(device=device, dtype=dtype)
    bias = head["bias"].to(device=device, dtype=dtype)

    print("[3/3] encoding", flush=True)
    embs: list[np.ndarray] = []
    started = time.time()
    with torch.inference_mode():
        for begin in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[begin:begin + args.batch_size]).to(device=device, dtype=dtype)
            precursor_tokens = model(batch, None)[:, 0]
            e = F.normalize(F.linear(precursor_tokens, weight, bias), dim=-1)
            embs.append(e.float().cpu().numpy())
            done = min(begin + args.batch_size, len(tensors))
            if done == len(tensors) or done % 200 == 0:
                print(f"  {done}/{len(tensors)} ({time.time()-started:.0f}s)", flush=True)
    embedding = np.concatenate(embs).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise RuntimeError("Non-finite embedding")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "official_embeddings.npy", embedding)
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    report = {
        "status": "official_dreams_qc_dda_embeddings",
        "study": "MTBLS1905", "n_spectra": len(manifest), "embedding_dim": int(embedding.shape[1]),
        "checkpoint": str(DEFAULT_OFFICIAL), "raw_checkpoint": str(DEFAULT_RAW),
        "preprocessing": "DreaMS-compatible top-100 peak selection, max-intensity normalization, precursor token",
        "source_files": [p.name for p in paths], "limit": args.limit,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
