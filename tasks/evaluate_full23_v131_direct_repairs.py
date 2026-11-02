#!/usr/bin/env python3
"""Evaluate direct one-factor successors of selected Full23 v1.3.1."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from evaluate_full23_multireference_on_gnps import _load_torch
from evaluate_full23_transfer_safe_tensor_on_gnps import predict
from full23_tensor_set_readout_core import model_from_payload
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _comparison,
    _joint_query_balanced,
    _metric_summary,
)
from train_full23_multireference_readout import _load_banks, _make_representation_panels
from train_full23_v131_direct_repairs import ARM_ORDER
from unified_additive_release import sha256_file, validate_checkpoint


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--noise-embeddings", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--full23-checkpoint", type=Path, required=True)
    parser.add_argument("--v131-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(comparator_embedding=[], comparator_score=[], full23_oof=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    full23 = validate_checkpoint(args.full23_checkpoint)
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(args, full23, cpu_banks)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()

    method_order = ["full23_v131"] + [f"v13_{arm}" for arm in ARM_ORDER]
    checkpoint_paths = {"full23_v131": args.v131_checkpoint}
    checkpoint_paths.update(
        {f"v13_{arm}": args.candidate_dir / f"v13_{arm}.pt" for arm in ARM_ORDER}
    )
    predicted, bounds, parameter_counts, hashes = {}, {}, {}, {}
    for method in method_order:
        checkpoint = checkpoint_paths[method]
        payload = _load_torch(checkpoint)
        model = model_from_payload(payload, args.device)
        bounds[method] = float(payload["residual_bound"])
        parameter_counts[method] = model.trainable_parameters
        hashes[method] = sha256_file(checkpoint)
        predicted[method] = predict(model, panels, banks, args.device)
        del model, payload
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    panel_bases = {name: panel.base for name, panel in panels.items()}
    scores, metrics, comparisons, saturation = {}, {}, {}, {}
    for panel_index, panel_name in enumerate(PANELS):
        panel = panel_bases[panel_name]
        scores[panel_name] = {
            method: predicted[method][panel_name] for method in method_order
        }
        metrics[panel_name] = {
            method: _metric_summary(value, panel)[0]
            for method, value in scores[panel_name].items()
        }
        comparisons[panel_name] = {}
        saturation[panel_name] = {}
        anchor = panels[panel_name].anchor
        for arm_index, arm in enumerate(ARM_ORDER):
            method = f"v13_{arm}"
            comparisons[panel_name][f"{method}_vs_full23_v131"] = _comparison(
                scores[panel_name][method],
                scores[panel_name]["full23_v131"],
                panel,
                5000,
                20261370 + panel_index * 100 + arm_index,
            )
        for method in method_order:
            residual = scores[panel_name][method] - anchor
            bound = bounds[method]
            query_saturated = []
            for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
                local = residual[int(left) : int(right)]
                query_saturated.append(
                    float(np.max(local) - np.min(local)) >= 0.90 * (2.0 * bound)
                )
            saturation[panel_name][method] = {
                "candidate_abs_residual_ge_0.99_bound": float(
                    np.mean(np.abs(residual) >= 0.99 * bound)
                ),
                "query_residual_span_ge_0.90_maximum": float(
                    np.mean(query_saturated)
                ),
            }

    joint = {}
    for method in method_order:
        joint[method], _ = _joint_query_balanced(panel_bases, scores, method)
    selected = method_order[0]
    for method in method_order[1:]:
        if joint[method]["top1_utility"] > joint[selected]["top1_utility"]:
            selected = method

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel = panel_bases[panel_name]
            np.savez_compressed(
                staging / f"candidate_predictions_{panel_name}.npz",
                query_ptr=panel.query_ptr,
                labels=panel.labels,
                **scores[panel_name],
            )
        shutil.copy2(checkpoint_paths[selected], staging / "selected_model.pt")
        report = {
            "status": "FULL23_V131_DIRECT_REPAIRS_GNPS_DEVELOPMENT_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development and model selection",
                "queries": {
                    name: len(panel.query_ptr) - 1 for name, panel in panel_bases.items()
                },
            },
            "method_order": method_order,
            "selected": selected,
            "selected_source": str(checkpoint_paths[selected]),
            "checkpoint_sha256": hashes,
            "selection_policy": {
                "primary_endpoint": "query-balanced unique-query Top-1 point estimate",
                "winner": "highest point estimate across v1.3.1 and three direct one-factor repairs",
                "tie_break": "v1.3.1, then earlier fixed arm order",
                "risk_net_lambda2": "descriptive only; never a veto",
            },
            "metrics": metrics,
            "comparisons": comparisons,
            "query_balanced_joint": joint,
            "residual_saturation": saturation,
            "trainable_parameters": parameter_counts,
            "claim_boundary": (
                "GNPS is consumed development and selected the winner; this is "
                "not external confirmation."
            ),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"status": report["status"], "selected": selected, "output": str(args.output)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
