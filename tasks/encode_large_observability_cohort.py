"""Encode the large observability cohort with the complete official DreaMS head."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi
from e1_checkpoint_io import official_head_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, default=Path("data/validation/large_observability_cohort"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_observability_embeddings"))
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--data", type=Path, default=multi.DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.cohort_dir / "selected_spectra.csv")
    selected = selected.loc[selected["audit_split"].isin(args.splits)].copy()
    selected = selected.sort_values(["audit_split", "formula", "ik14", "hdf5_row"]).reset_index(drop=True)
    rows = selected["hdf5_row"].to_numpy(np.int64)

    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            precursor = model(batch.to(device=device, dtype=next(model.parameters()).dtype), None)[:, 0]
            output.append(F.normalize(F.linear(precursor, weight, bias), dim=-1).float().cpu().numpy())
    embedding = np.concatenate(output)
    del model, raw_package, official_package
    gc.collect()
    if embedding.shape[0] != len(selected) or not np.isfinite(embedding).all():
        raise RuntimeError("Embedding alignment or finiteness audit failed")
    np.save(args.output_dir / "official_embeddings.npy", embedding)
    selected.to_csv(args.output_dir / "manifest.csv", index=False)
    report = {
        "status": "large_observability_official_embeddings",
        "checkpoint": str(args.official_checkpoint),
        "embedding_definition": "L2-normalized official linear-head(backbone precursor token)",
        "splits": args.splits,
        "spectra": len(selected), "molecules": int(selected["ik14"].nunique()),
        "formulas": int(selected["formula"].nunique()), "shape": list(embedding.shape),
        "test_split_encoded": bool("test" in args.splits),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
