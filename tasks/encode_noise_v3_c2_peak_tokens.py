"""Encode official contextual peak tokens for a declared C1 graph row scope.

The official DreaMS backbone is frozen.  Its final-layer fragment tokens are
L2-normalized and mapped through one preregistered label-free Gaussian random
projection.  The cache is query-row keyed and contains no identity labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi
from build_g8r_real_error_atlas import Cache


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2_peak_tokens")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--row-scope", choices=("queries", "reachable"), default="queries",
        help="Encode query rows only, or every query/candidate spectrum reachable in the graph.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.graph, args.data, args.raw_checkpoint, args.official_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph = Cache(args.graph)
    if args.row_scope == "queries":
        rows = np.unique(graph.query_row).astype(np.int64)
    else:
        rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    if args.max_rows:
        rows = rows[:args.max_rows]
    formal = args.max_rows == 0
    expected_rows = {"queries": 23876, "reachable": 25275}[args.row_scope]
    if formal and len(rows) != expected_rows:
        raise RuntimeError(
            f"formal C2 token cache expects {expected_rows:,} rows for scope={args.row_scope}, got {len(rows):,}"
        )
    device = torch.device(args.device)
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device)
    model.eval()
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    projection_cpu = torch.randn(
        model.d_model, args.projection_dim, generator=generator, dtype=torch.float32,
    ) / np.sqrt(args.projection_dim)
    projection = projection_cpu.to(device=device, dtype=dtype)
    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    staging = Path(tempfile.mkdtemp(prefix="c2_tokens_", dir=args.output_dir.parent))
    try:
        np.save(staging / "rows.npy", rows)
        np.save(staging / "projection.npy", projection_cpu.numpy())
        shape = (len(rows), args.n_highest_peaks, args.projection_dim)
        tokens_out = np.lib.format.open_memmap(staging / "tokens_f16.npy", mode="w+", dtype=np.float16, shape=shape)
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
            for batch_index, batch in enumerate(loader, start=1):
                batch = batch.to(device=device, dtype=dtype)
                contextual = F.normalize(model(batch, None)[:, 1:, :], dim=-1)
                projected = F.normalize(contextual @ projection, dim=-1)
                valid = batch[:, 1:, 0] > 0
                projected = projected.masked_fill(~valid.unsqueeze(-1), 0)
                count = len(batch)
                tokens_out[cursor:cursor + count] = projected.half().cpu().numpy()
                mz_out[cursor:cursor + count] = batch[:, 1:, 0].float().cpu().numpy()
                intensity_out[cursor:cursor + count] = batch[:, 1:, 1].float().cpu().numpy()
                valid_out[cursor:cursor + count] = valid.cpu().numpy()
                cursor += count
                if batch_index % 50 == 0 or batch_index == len(loader):
                    print(f"[C2-M0] {cursor:,}/{len(rows):,} spectra", flush=True)
        if cursor != len(rows):
            raise RuntimeError("C2 token encoder row-count mismatch")
        for array in (tokens_out, mz_out, intensity_out, valid_out):
            array.flush()
        del tokens_out, mz_out, intensity_out, valid_out
        valid = np.load(staging / "valid.npy", mmap_mode="r")
        report = {
            "status": "noise_v3_c2_peak_token_cache_complete", "formal": formal,
            "row_scope": args.row_scope,
            "spectra": int(len(rows)), "tokens_per_spectrum": int(args.n_highest_peaks),
            "projection_dim": int(args.projection_dim),
            "valid_peak_tokens": int(np.sum(valid)),
            "source": "official fine-tuned DreaMS final-layer contextual fragment tokens",
            "projection": f"label-free Gaussian random projection; seed={args.seed}",
            "identity_labels_used": False,
            "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
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
