"""Preflight for formal counterfactual DreaMS fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="Permit smoke tests without CUDA")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e1/counterfactual_training_preflight.json")
    args = parser.parse_args()
    files = {
        "training split": ROOT / "data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv",
        "MassSpecGym HDF5": ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
        "discovery manifest": ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
        "official discovery embeddings": ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy",
        "raw architecture checkpoint": ROOT / "dreams/models/pretrained/ssl_model_server.pt",
        "official slim checkpoint": ROOT / "data/e1/official_embedding_slim.pt",
    }
    failed = False
    print("Counterfactual DreaMS training preflight")
    for label, path in files.items():
        exists = path.is_file()
        failed |= not exists
        size = f"{path.stat().st_size / 2**20:.1f} MiB" if exists else "MISSING"
        print(f"  [{'OK' if exists else 'FAIL'}] {label}: {path} ({size})")
    cuda = torch.cuda.is_available()
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {cuda}")
    if cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory free/total: {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    elif not args.allow_cpu:
        print("  [FAIL] Formal backbone fine-tuning requires CUDA; use --allow-cpu only for smoke tests")
        failed = True
    details = {}
    if all(path.is_file() for path in files.values()):
        split = pd.read_csv(files["training split"])
        manifest = pd.read_csv(files["discovery manifest"])
        train_formula = set(split.loc[split["pilot_split"] == "train", "formula"])
        val_formula = set(split.loc[split["pilot_split"] == "validation", "formula"])
        overlap = train_formula & val_formula
        missing_rows = set(split["query_hdf5_row"]) - set(manifest["hdf5_row"])
        with h5py.File(files["MassSpecGym HDF5"], "r") as handle:
            hdf5_size = len(handle["spectrum"])
        out_of_bounds = int((split[["query_hdf5_row", "identity_hdf5_row", "confounder_hdf5_row"]].to_numpy() >= hdf5_size).sum())
        details = {
            "examples": len(split), "train_formulas": len(train_formula),
            "validation_formulas": len(val_formula), "formula_overlap": len(overlap),
            "missing_query_rows_in_manifest": len(missing_rows), "out_of_bounds_rows": out_of_bounds,
            "identity_interventions": int(split["has_identity_intervention"].sum()),
            "confounder_interventions": int(split["has_confounder_intervention"].sum()),
        }
        print(f"  Examples: {len(split):,}; train/validation formulas: {len(train_formula)}/{len(val_formula)}")
        print(f"  Formula overlap: {len(overlap)}; invalid HDF5 rows: {out_of_bounds}")
        failed |= bool(overlap) or bool(missing_rows) or bool(out_of_bounds)
    report = {"ok": not failed, "cuda": cuda, "files": {key: str(value) for key, value in files.items()}, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("Preflight failed; fix the FAIL items before formal training.")
    print("Preflight passed.")


if __name__ == "__main__":
    main()
