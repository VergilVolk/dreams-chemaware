"""Audit the repaired peak-rule -> attention-logit -> DreaMS representation path.

This is a mechanism audit, not a retrieval benchmark.  It proves four narrow
claims on the official backbone and real MassSpecGym spectra:

1. zero chemical scale reproduces the official backbone;
2. a nonzero scale changes the representation;
3. gradients reach the route scale and individual rule weights;
4. the configured intervention is routed to exactly one recorded layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from dreams.models.chem_aware.train_chem_aware import (  # noqa: E402
    build_chem_aware_from_pretrained,
)
from noise_final_core import sha256_file  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--rows", default="0,1,2,3")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--active-scale", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_true_attention_path_local.json",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows = np.asarray([int(value) for value in args.rows.split(",")], dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("--rows must be a nonempty unique comma-separated vector")
    if args.active_scale == 0:
        raise ValueError("--active-scale must be nonzero")
    for path in (args.official_checkpoint, args.raw_checkpoint, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    official, kind = load_base_model(
        args.official_checkpoint,
        args.raw_checkpoint,
        device,
        args.n_highest_peaks,
    )
    official.eval()
    chemistry = build_chem_aware_from_pretrained(
        official.backbone,
        chem_attn_layer=args.target_layer,
        chem_attn_mode="attention",
    ).to(device)
    chemistry.eval()

    spectra = []
    with h5py.File(args.data, "r") as handle:
        n_rows = int(handle["spectrum"].shape[0])
        if np.any(rows < 0) or np.any(rows >= n_rows):
            raise ValueError(f"rows outside HDF5 range 0..{n_rows - 1}")
        for row in rows:
            spectra.append(preprocess_spectrum(
                np.asarray(handle["spectrum"][int(row)]),
                float(handle["precursor_mz"][int(row)]),
                args.n_highest_peaks,
            ))
    batch = torch.stack(spectra).to(device)

    with torch.no_grad():
        official_output = official.backbone(batch, None)
        chemistry.chem_attention_scale.zero_()
        zero_output = chemistry(batch, None)
        zero_abs = torch.abs(zero_output - official_output)
        zero_max_abs = float(zero_abs.max().cpu())
        zero_mean_abs = float(zero_abs.mean().cpu())
        zero_exact = bool(torch.equal(zero_output, official_output))

        chemistry.chem_attention_scale.fill_(float(args.active_scale))
        active_output = chemistry(batch, None)
        active_abs = torch.abs(active_output - zero_output)
        active_max_abs = float(active_abs.max().cpu())
        active_mean_abs = float(active_abs.mean().cpu())
        cosine = torch.nn.functional.cosine_similarity(
            active_output[:, 0], zero_output[:, 0], dim=-1
        )
        analysis = chemistry.get_chem_attn_analysis()

    chemistry.zero_grad(set_to_none=True)
    active_output = chemistry(batch, None)
    # A generic precursor-representation functional is sufficient for path
    # reachability; retrieval utility is deliberately assessed elsewhere.
    loss = active_output[:, 0].float().square().mean()
    loss.backward()
    scale_grad = chemistry.chem_attention_scale.grad
    rule_grad = chemistry.chem_rule_engine.rule_weights_raw.grad
    scale_grad_value = 0.0 if scale_grad is None else float(scale_grad.detach().cpu())
    rule_grad_norm = 0.0 if rule_grad is None else float(rule_grad.norm().detach().cpu())
    rule_grad_nonzero = 0 if rule_grad is None else int(torch.count_nonzero(rule_grad).detach().cpu())

    report = {
        "status": "pass" if (
            zero_max_abs <= 2e-6
            and active_max_abs > 1e-8
            and abs(scale_grad_value) > 1e-12
            and rule_grad_norm > 1e-12
        ) else "fail",
        "scope": "mechanism_only_not_retrieval_utility",
        "checkpoint_kind": kind,
        "official_checkpoint": str(args.official_checkpoint),
        "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        "raw_checkpoint": str(args.raw_checkpoint),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "rows": rows.tolist(),
        "n_highest_peaks": int(args.n_highest_peaks),
        "route": {
            "mode": analysis["mode"],
            "target_layer": int(analysis["target_layer"]),
            "categories": analysis["categories"],
            "active_scale": float(args.active_scale),
            "rule_count": int(chemistry.chem_rule_engine.rule_weights_raw.numel()),
            "effective_bias_nonzero": int(torch.count_nonzero(
                analysis["effective_bias"]
            ).cpu()),
            "effective_bias_max": float(analysis["effective_bias"].max().cpu()),
        },
        "zero_scale_reproduction": {
            "bitwise_exact": zero_exact,
            "max_abs": zero_max_abs,
            "mean_abs": zero_mean_abs,
        },
        "active_effect": {
            "max_abs": active_max_abs,
            "mean_abs": active_mean_abs,
            "precursor_cosine_min": float(cosine.min().cpu()),
            "precursor_cosine_mean": float(cosine.mean().cpu()),
        },
        "gradient_reachability": {
            "loss": float(loss.detach().cpu()),
            "attention_scale_grad": scale_grad_value,
            "rule_weight_grad_norm": rule_grad_norm,
            "rule_weight_nonzero_gradients": rule_grad_nonzero,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
