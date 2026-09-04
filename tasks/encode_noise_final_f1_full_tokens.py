"""Cache full official DreaMS contextual peak tokens for symmetric F1.

The cache is label-free and contains every spectrum reachable from the locked
training candidate graph.  It is an execution cache only: F1 inference remains
raw MS/MS -> official backbone -> shared peak adapter -> new embedding.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi
from noise_final_core import CandidateGraph, sha256_file


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--f0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=0, help="smoke only")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = [args.graph, args.data, args.raw_checkpoint, args.official_checkpoint, args.f0_dir / "decision.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    f0 = json.loads((args.f0_dir / "decision.json").read_text(encoding="utf-8"))
    if f0.get("status") != "noise_final_f0_symmetric_protocol_passed" or not f0.get("pass"):
        raise RuntimeError("F1 token cache requires a passing symmetric F0")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite token cache: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    if args.max_rows:
        rows = rows[:args.max_rows]
    formal = args.max_rows == 0
    if formal and len(rows) != 25275:
        raise RuntimeError(f"formal F1 expects 25,275 reachable spectra, observed {len(rows):,}")

    device = torch.device(args.device)
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_f1_tokens_", dir=args.output_dir.parent))
    try:
        np.save(staging / "rows.npy", rows)
        token_shape = (len(rows), args.n_highest_peaks, int(model.d_model))
        tokens_out = np.lib.format.open_memmap(
            staging / "tokens_f16.npy", mode="w+", dtype=np.float16, shape=token_shape,
        )
        mz_out = np.lib.format.open_memmap(
            staging / "mz_f32.npy", mode="w+", dtype=np.float32,
            shape=(len(rows), args.n_highest_peaks),
        )
        intensity_out = np.lib.format.open_memmap(
            staging / "intensity_f32.npy", mode="w+", dtype=np.float32,
            shape=(len(rows), args.n_highest_peaks),
        )
        valid_out = np.lib.format.open_memmap(
            staging / "valid.npy", mode="w+", dtype=bool,
            shape=(len(rows), args.n_highest_peaks),
        )
        cursor = 0
        with torch.inference_mode():
            for batch_index, spectra in enumerate(loader, start=1):
                spectra = spectra.to(device=device, dtype=dtype, non_blocking=True)
                contextual = model(spectra, None)[:, 1:, :]
                valid = spectra[:, 1:, 0] > 0
                contextual = contextual.masked_fill(~valid.unsqueeze(-1), 0)
                count = len(spectra)
                tokens_out[cursor:cursor + count] = contextual.half().cpu().numpy()
                mz_out[cursor:cursor + count] = spectra[:, 1:, 0].float().cpu().numpy()
                intensity_out[cursor:cursor + count] = spectra[:, 1:, 1].float().cpu().numpy()
                valid_out[cursor:cursor + count] = valid.cpu().numpy()
                cursor += count
                if batch_index % 50 == 0 or batch_index == len(loader):
                    print(f"[F1-token] {cursor:,}/{len(rows):,} spectra", flush=True)
        if cursor != len(rows):
            raise RuntimeError("F1 token cache row-count mismatch")
        for array in (tokens_out, mz_out, intensity_out, valid_out):
            array.flush()
        del tokens_out, mz_out, intensity_out, valid_out
        report = {
            "status": "noise_final_f1_full_token_cache_complete",
            "formal": formal, "spectra": int(len(rows)),
            "tokens_per_spectrum": int(args.n_highest_peaks),
            "token_dimension": int(model.d_model),
            "source": "official fine-tuned DreaMS final-layer contextual peak tokens",
            "identity_labels_used": False,
            "contract": "execution cache for one shared query/reference encoder",
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "hdf5_sha256": sha256_file(args.data),
                "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
                "f0_decision_sha256": sha256_file(args.f0_dir / "decision.json"),
                "script_sha256": sha256_file(Path(__file__)),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
