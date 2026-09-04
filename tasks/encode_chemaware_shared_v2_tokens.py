"""Build the locked contextual-token cache for ChemAware shared embedding v2.

The cache is label-free.  It contains every spectrum reachable from the locked
candidate graph, its official final-layer peak tokens, precursor/peak
measurements, and the official normalized embedding from the same forward
pass.  Candidate information is never supplied to the encoder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi
from chemaware_shared_v2_core import TOKEN_STATUS
from e1_checkpoint_io import checkpoint_kind, official_head_state
from noise_final_core import CandidateGraph, sha256_file


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--preflight", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_tokens")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=0, help="non-formal smoke limit")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = [args.graph, args.data, args.raw_checkpoint, args.official_checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite token cache: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    formal = args.max_rows == 0
    if formal:
        if not args.preflight.is_file():
            raise FileNotFoundError(args.preflight)
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if preflight.get("status") != "chemaware_shared_v2_preflight_passed" or not preflight.get("formal"):
            raise RuntimeError("formal token cache requires passing ChemAware-v2 preflight")
    if args.max_rows:
        rows = rows[:args.max_rows]
    if formal and len(rows) != 25275:
        raise RuntimeError(f"formal G8R graph expects 25,275 reachable spectra, observed {len(rows):,}")

    device = torch.device(args.device)
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(official) not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("ChemAware v2 requires the locked official embedding checkpoint")
    backbone = multi.reconstruct_backbone(
        raw, multi.official_backbone_state(official), device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = torch.nn.Linear(int(backbone.d_model), int(backbone.d_model), bias=True).to(device)
    head.load_state_dict(official_head_state(official), strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    dtype = next(backbone.parameters()).dtype
    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".chemaware_v2_tokens_", dir=args.output_dir.parent))
    try:
        np.save(staging / "rows.npy", rows)
        peak_shape = (len(rows), args.n_highest_peaks)
        tokens_out = np.lib.format.open_memmap(
            staging / "tokens_f16.npy", mode="w+", dtype=np.float16,
            shape=(*peak_shape, int(backbone.d_model)),
        )
        mz_out = np.lib.format.open_memmap(
            staging / "mz_f32.npy", mode="w+", dtype=np.float32, shape=peak_shape
        )
        intensity_out = np.lib.format.open_memmap(
            staging / "intensity_f32.npy", mode="w+", dtype=np.float32, shape=peak_shape
        )
        valid_out = np.lib.format.open_memmap(
            staging / "valid.npy", mode="w+", dtype=bool, shape=peak_shape
        )
        precursor_out = np.lib.format.open_memmap(
            staging / "precursor_mz_f32.npy", mode="w+", dtype=np.float32, shape=(len(rows),)
        )
        embedding_out = np.lib.format.open_memmap(
            staging / "official_embeddings_f32.npy", mode="w+", dtype=np.float32,
            shape=(len(rows), int(backbone.d_model)),
        )
        cursor = 0
        with torch.inference_mode():
            for batch_index, spectra in enumerate(loader, start=1):
                spectra = spectra.to(device=device, dtype=dtype, non_blocking=True)
                contextual = backbone(spectra, None)
                valid = spectra[:, 1:, 0] > 0
                fragments = contextual[:, 1:, :].masked_fill(~valid.unsqueeze(-1), 0)
                embedding = F.normalize(head(contextual[:, 0, :].float()), dim=-1)
                count = len(spectra)
                block = slice(cursor, cursor + count)
                tokens_out[block] = fragments.half().cpu().numpy()
                mz_out[block] = spectra[:, 1:, 0].float().cpu().numpy()
                intensity_out[block] = spectra[:, 1:, 1].float().cpu().numpy()
                valid_out[block] = valid.cpu().numpy()
                precursor_out[block] = spectra[:, 0, 0].float().cpu().numpy()
                embedding_out[block] = embedding.cpu().numpy()
                cursor += count
                if batch_index % 50 == 0 or batch_index == len(loader):
                    print(f"[ChemAware-v2 cache] {cursor:,}/{len(rows):,}", flush=True)
        if cursor != len(rows):
            raise RuntimeError("ChemAware token-cache row-count mismatch")
        for array in (
            tokens_out, mz_out, intensity_out, valid_out, precursor_out, embedding_out
        ):
            array.flush()
        del tokens_out, mz_out, intensity_out, valid_out, precursor_out, embedding_out
        report = {
            "status": TOKEN_STATUS,
            "formal": formal,
            "spectra": int(len(rows)),
            "tokens_per_spectrum": int(args.n_highest_peaks),
            "token_dimension": int(backbone.d_model),
            "source": "official DreaMS final contextual tokens and normalized embedding",
            "identity_labels_used": False,
            "candidate_inputs_used": False,
            "precursor_mz_stored": True,
            "contract": "execution cache only; deployment is raw spectrum to one shared encoder",
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "hdf5_sha256": sha256_file(args.data),
                "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
                "preflight_sha256": sha256_file(args.preflight) if formal else None,
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
