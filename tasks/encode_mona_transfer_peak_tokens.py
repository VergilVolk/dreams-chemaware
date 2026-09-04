#!/usr/bin/env python
"""Encode contextual peak tokens only for rows in the sealed MoNA panel."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from encode_mona_neg_library import parse_mgf  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from pilot_paired_layer_cka import DEFAULT_RAW, preprocess_spectrum, reconstruct_backbone  # noqa: E402
from e1_checkpoint_io import checkpoint_kind, official_backbone_state, torch_load_compat  # noqa: E402


class SelectedSpectra(Dataset):
    def __init__(self, records: list[dict], rows: np.ndarray, n_highest: int):
        self.records = records
        self.rows = rows
        self.n_highest = n_highest

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[int(self.rows[index])]
        return preprocess_spectrum(record["peaks"], record["precursor_mz"], self.n_highest)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=ROOT / "data/validation/mona_identity_disjoint_transfer_panel")
    parser.add_argument("--mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy")
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/mona_identity_disjoint_transfer_tokens")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = [args.panel_dir / "panel.npz", args.panel_dir / "report.json", args.mgf,
                args.manifest, args.embeddings, args.raw_checkpoint, args.official_checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite transfer token cache: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    panel_report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    if not panel_report.get("formal") or panel_report.get("construction_uses_model_scores"):
        raise RuntimeError("invalid sealed MoNA transfer panel")
    with np.load(args.panel_dir / "panel.npz") as panel:
        rows = np.unique(np.concatenate((panel["query_row"], panel["candidate_row"]))).astype(np.int64)
    manifest = pd.read_csv(args.manifest).fillna("")
    official_embeddings = np.load(args.embeddings, mmap_mode="r")
    records = parse_mgf(args.mgf)
    if len(records) != len(manifest) or len(records) != len(official_embeddings):
        raise RuntimeError("MGF, manifest and official embedding rows do not align")
    # Validate the row alignment before any expensive encoding.
    for row in rows:
        record = records[int(row)]
        if str(record.get("inchikey", ""))[:14] != str(manifest.iloc[int(row)].inchikey)[:14]:
            raise RuntimeError(f"MGF/manifest identity mismatch at row {row}")
        if not np.isclose(float(record["precursor_mz"]), float(manifest.iloc[int(row)].precursor_mz), rtol=2e-5, atol=1e-4):
            raise RuntimeError(f"MGF/manifest precursor mismatch at row {row}")

    raw = torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(raw) != "raw_ssl" or checkpoint_kind(official) != "official_embedding_slim":
        raise RuntimeError("unexpected checkpoint format")
    device = torch.device(args.device)
    model = reconstruct_backbone(raw, official_backbone_state(official), args.n_highest_peaks, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    loader = DataLoader(SelectedSpectra(records, rows, args.n_highest_peaks), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mona_transfer_tokens_", dir=args.output_dir.parent))
    try:
        np.save(staging / "rows.npy", rows)
        np.savez_compressed(staging / "official_embeddings.npz", rows=rows,
                            embeddings=np.asarray(official_embeddings[rows], dtype=np.float32))
        shape = (len(rows), args.n_highest_peaks)
        tokens = np.lib.format.open_memmap(staging / "tokens_f16.npy", mode="w+", dtype=np.float16,
                                           shape=shape + (int(model.d_model),))
        mz = np.lib.format.open_memmap(staging / "mz_f32.npy", mode="w+", dtype=np.float32, shape=shape)
        intensity = np.lib.format.open_memmap(staging / "intensity_f32.npy", mode="w+", dtype=np.float32, shape=shape)
        valid_out = np.lib.format.open_memmap(staging / "valid.npy", mode="w+", dtype=bool, shape=shape)
        cursor = 0
        with torch.inference_mode():
            for batch_index, spectra in enumerate(loader, start=1):
                spectra = spectra.to(device=device, dtype=dtype, non_blocking=True)
                contextual = model(spectra, None)[:, 1:, :]
                valid = spectra[:, 1:, 0] > 0
                contextual = contextual.masked_fill(~valid.unsqueeze(-1), 0)
                count = len(spectra)
                tokens[cursor:cursor + count] = contextual.half().cpu().numpy()
                mz[cursor:cursor + count] = spectra[:, 1:, 0].float().cpu().numpy()
                intensity[cursor:cursor + count] = spectra[:, 1:, 1].float().cpu().numpy()
                valid_out[cursor:cursor + count] = valid.cpu().numpy()
                cursor += count
                if batch_index % 50 == 0 or batch_index == len(loader):
                    print(f"[MoNA tokens] {cursor:,}/{len(rows):,}", flush=True)
        for array in (tokens, mz, intensity, valid_out):
            array.flush()
        del tokens, mz, intensity, valid_out
        if cursor != len(rows):
            raise RuntimeError("token cache row count mismatch")
        report = {
            "status": "mona_identity_disjoint_transfer_token_cache_complete",
            "formal": True,
            "spectra": int(len(rows)), "tokens_per_spectrum": args.n_highest_peaks,
            "token_dimension": int(model.d_model), "identity_labels_used": False,
            "contract": "label-free execution cache for frozen, identity-disjoint MoNA transfer panel",
            "provenance": {
                "panel_sha256": sha256_file(args.panel_dir / "panel.npz"),
                "panel_report_sha256": sha256_file(args.panel_dir / "report.json"),
                "mgf_sha256": sha256_file(args.mgf), "manifest_sha256": sha256_file(args.manifest),
                "source_embeddings_sha256": sha256_file(args.embeddings),
                "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
                "embedding_cache_sha256": sha256_file(staging / "official_embeddings.npz"),
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
