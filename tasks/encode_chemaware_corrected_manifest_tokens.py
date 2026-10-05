"""Resumable official-DreaMS token cache for the corrected 83k-query manifest.

The cache contains each reachable *training* spectrum exactly once.  It is an
execution accelerator for large candidate-conditioned ChemAware training; the
deployed model still consumes one raw spectrum and emits one shared embedding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

import pilot_multilevel_factor_activations as multi  # noqa: E402
from e1_checkpoint_io import checkpoint_kind, official_head_state  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--raw-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def write_progress(path: Path, body: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = arguments()
    started = time.time()
    required = (args.manifest, args.data, args.raw_checkpoint, args.official_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise RuntimeError(f"completed cache already exists: {report_path}")

    with np.load(args.manifest) as body:
        rows = np.unique(np.concatenate((body["query_row"], body["pair_candidate_row"]))).astype(np.int64)
    if args.max_rows:
        rows = rows[:args.max_rows]
    rows_path = args.output_dir / "rows.npy"
    if rows_path.exists():
        existing = np.load(rows_path)
        if not np.array_equal(existing, rows):
            raise RuntimeError("resume rows differ from the frozen manifest")
    else:
        np.save(rows_path, rows)

    device = torch.device(args.device)
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(official) not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("requires the locked official embedding checkpoint")
    backbone = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = torch.nn.Linear(int(backbone.d_model), int(backbone.d_model), bias=True).to(device)
    head.load_state_dict(official_head_state(official), strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    dimension = int(backbone.d_model)
    peak_shape = (len(rows), args.n_highest_peaks)
    arrays = {
        "tokens_f16.npy": (np.float16, (*peak_shape, dimension)),
        "mz_f32.npy": (np.float32, peak_shape),
        "intensity_f32.npy": (np.float32, peak_shape),
        "valid.npy": (bool, peak_shape),
        "precursor_mz_f32.npy": (np.float32, (len(rows),)),
        "official_embeddings_f32.npy": (np.float32, (len(rows), dimension)),
    }
    progress_path = args.output_dir / "progress.json"
    cursor = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        cursor = int(progress["completed_rows"])
        if progress.get("manifest_sha256") != sha256_file(args.manifest):
            raise RuntimeError("resume manifest provenance mismatch")
    outputs = {}
    for name, (dtype, shape) in arrays.items():
        path = args.output_dir / name
        if path.exists():
            outputs[name] = np.lib.format.open_memmap(path, mode="r+")
            if outputs[name].shape != shape or outputs[name].dtype != np.dtype(dtype):
                raise RuntimeError(f"resume array mismatch: {path}")
        else:
            if cursor:
                raise RuntimeError(f"resume cache is incomplete: {path}")
            outputs[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    remaining = rows[cursor:]
    loader = DataLoader(
        multi.SpectrumRows(args.data, remaining, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dtype = next(backbone.parameters()).dtype
    session_started = time.time()
    with torch.inference_mode():
        for batch_index, spectra in enumerate(loader, start=1):
            spectra = spectra.to(device=device, dtype=dtype, non_blocking=True)
            contextual = backbone(spectra, None)
            valid = spectra[:, 1:, 0] > 0
            fragments = contextual[:, 1:, :].masked_fill(~valid.unsqueeze(-1), 0)
            embedding = F.normalize(head(contextual[:, 0, :].float()), dim=-1)
            count = len(spectra)
            block = slice(cursor, cursor + count)
            outputs["tokens_f16.npy"][block] = fragments.half().cpu().numpy()
            outputs["mz_f32.npy"][block] = spectra[:, 1:, 0].float().cpu().numpy()
            outputs["intensity_f32.npy"][block] = spectra[:, 1:, 1].float().cpu().numpy()
            outputs["valid.npy"][block] = valid.cpu().numpy()
            outputs["precursor_mz_f32.npy"][block] = spectra[:, 0, 0].float().cpu().numpy()
            outputs["official_embeddings_f32.npy"][block] = embedding.cpu().numpy()
            cursor += count
            if batch_index % 25 == 0 or cursor == len(rows):
                for output in outputs.values():
                    output.flush()
                elapsed = time.time() - session_started
                rate = (cursor - (len(rows) - len(remaining))) / max(elapsed, 1e-9)
                progress = {
                    "status": "in_progress", "completed_rows": cursor,
                    "total_rows": int(len(rows)), "rows_per_second": rate,
                    "manifest_sha256": sha256_file(args.manifest),
                    "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
                }
                write_progress(progress_path, progress)
                print(
                    f"[full ChemAware cache] {cursor:,}/{len(rows):,}; {rate:.2f} spectra/s",
                    flush=True,
                )
    if cursor != len(rows):
        raise RuntimeError("cache row-count mismatch")
    probe = outputs["official_embeddings_f32.npy"]
    norms = np.linalg.norm(np.asarray(probe[:: max(1, len(rows) // 1000)]), axis=1)
    if not np.all(np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 2e-3):
        raise RuntimeError("official embedding cache failed unit-norm audit")
    report = {
        "status": "chemaware_corrected_manifest_token_cache_complete",
        "spectra": int(len(rows)), "tokens_per_spectrum": args.n_highest_peaks,
        "token_dimension": dimension,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "hdf5_sha256": sha256_file(args.data),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
        "contract": {
            "candidate_inputs_used_by_encoder": False,
            "training_split_only": True,
            "deployment_remains_one_raw_spectrum_to_one_shared_embedding": True,
        },
        "runtime_seconds_current_session": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
