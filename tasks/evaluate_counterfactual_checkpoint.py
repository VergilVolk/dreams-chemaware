"""Evaluate a formal counterfactual checkpoint on the internal full candidate pool."""

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
from evaluate_counterfactual_head_retrieval import bootstrap, query_metrics, summarize
from train_e1_identity import load_base_model


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv")
    parser.add_argument("--official-embeddings", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-ckpt", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    manifest = pd.read_csv(args.manifest)
    split = pd.read_csv(args.split)
    validation_formulas = set(split.loc[split["pilot_split"] == "validation", "formula"])
    # Encode only spectra in held-out formulas, then place them back into the
    # manifest-indexed matrix required by the retrieval evaluator.
    selected = manifest.index[manifest["formula"].isin(validation_formulas)].to_numpy(np.int64)
    loader = DataLoader(
        multi.SpectrumRows(args.data, manifest.loc[selected, "hdf5_row"].to_numpy(np.int64), args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    model, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if package.get("format") != "counterfactual_dreams_v1":
        raise ValueError("Checkpoint is not counterfactual_dreams_v1")
    model.backbone.load_state_dict(package["backbone_state_dict"], strict=True)
    model.head.load_state_dict(package["head_state_dict"], strict=True)
    model.eval()
    values = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                values.append(model(batch).float().cpu().numpy())
    trained_selected = np.concatenate(values)
    del model
    gc.collect()
    official = np.load(args.official_embeddings).astype(np.float32)
    trained = official.copy()
    trained[selected] = trained_selected
    baseline_queries = query_metrics(manifest, official, validation_formulas, args.ppm)
    trained_queries = query_metrics(manifest, trained, validation_formulas, args.ppm)
    baseline_queries.to_csv(args.output_dir / "official_queries.csv", index=False)
    trained_queries.to_csv(args.output_dir / "trained_queries.csv", index=False)
    baseline = summarize(baseline_queries)
    result = summarize(trained_queries)
    report = {
        "status": "formal_counterfactual_internal_retrieval",
        "checkpoint": str(args.checkpoint), "stage": package.get("stage"),
        "protocol": "formula-isolated internal validation; same formula; <=10 ppm; duplicate hashes excluded",
        "baseline": baseline,
        "trained": result | {
            "top1_minus_official": result["top1"] - baseline["top1"],
            "top1_formula_bootstrap_ci95": bootstrap(baseline_queries, trained_queries, "top1", args.bootstrap, 20260814),
            "mrr_minus_official": result["mrr"] - baseline["mrr"],
            "mrr_formula_bootstrap_ci95": bootstrap(baseline_queries, trained_queries, "mrr", args.bootstrap, 20260815),
            "mean_embedding_cosine_to_official": float(np.mean(np.sum(trained_selected * official[selected], axis=1))),
        },
        "confirmation_usage": "none", "test_usage": "none",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
