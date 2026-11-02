#!/usr/bin/env python3
"""Evaluate and select among six fixed cumulative Full23 v1.3 repairs."""
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
    PANELS, _comparison, _joint_query_balanced, _metric_summary,
)
from train_full23_multireference_readout import _load_banks, _make_representation_panels
from train_full23_v13_cumulative_repairs import ARM_ORDER
from unified_additive_release import validate_checkpoint


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
    parser.add_argument("--formal-v13-checkpoint", type=Path, required=True)
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

    method_order = ["full23_v13"] + [f"v13_{arm}" for arm in ARM_ORDER]
    predicted = {}
    formal_payload = _load_torch(args.formal_v13_checkpoint)
    residual_bounds = {
        "full23_v13": float(formal_payload["residual_bound"]),
    }
    parameter_counts = {}
    formal_model = model_from_payload(formal_payload, args.device)
    parameter_counts["full23_v13"] = formal_model.trainable_parameters
    predicted["full23_v13"] = predict(
        formal_model, panels, banks, args.device,
    )
    del formal_model, formal_payload
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    for arm in ARM_ORDER:
        method = f"v13_{arm}"
        checkpoint = args.candidate_dir / f"v13_{arm}.pt"
        payload = _load_torch(checkpoint)
        residual_bounds[method] = float(payload["residual_bound"])
        model = model_from_payload(payload, args.device)
        parameter_counts[method] = model.trainable_parameters
        predicted[method] = predict(model, panels, banks, args.device)
        del model, payload
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    scores, metrics, comparisons, saturation = {}, {}, {}, {}
    for panel_index, panel_name in enumerate(PANELS):
        panel = panels[panel_name].base
        scores[panel_name] = {
            method: predicted[method][panel_name] for method in method_order
        }
        metrics[panel_name] = {
            method: _metric_summary(value, panel)[0]
            for method, value in scores[panel_name].items()
        }
        saturation[panel_name] = {}
        anchor = panels[panel_name].anchor
        for method in method_order:
            residual = scores[panel_name][method] - anchor
            bound = residual_bounds[method]
            saturated_candidates = float(np.mean(np.abs(residual) >= 0.99 * bound))
            saturated_queries = []
            for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
                local = residual[int(left):int(right)]
                saturated_queries.append(
                    float(np.max(local) - np.min(local)) >= 0.90 * (2.0 * bound)
                )
            saturation[panel_name][method] = {
                "candidate_abs_residual_ge_0.99_bound": saturated_candidates,
                "query_residual_span_ge_0.90_maximum": float(np.mean(saturated_queries)),
            }
        comparisons[panel_name] = {}
        for arm_index, arm in enumerate(ARM_ORDER):
            method = f"v13_{arm}"
            predecessor = "full23_v13" if arm_index == 0 else f"v13_{ARM_ORDER[arm_index-1]}"
            comparisons[panel_name][f"{method}_vs_formal_v13"] = _comparison(
                scores[panel_name][method], scores[panel_name]["full23_v13"], panel,
                5000, 20261170 + panel_index * 100 + arm_index,
            )
            comparisons[panel_name][f"{method}_vs_{predecessor}"] = _comparison(
                scores[panel_name][method], scores[panel_name][predecessor], panel,
                5000, 20261270 + panel_index * 100 + arm_index,
            )

    joint = {}
    panel_bases = {name: panel.base for name, panel in panels.items()}
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
            panel = panels[panel_name].base
            np.savez_compressed(
                staging / f"candidate_predictions_{panel_name}.npz",
                query_ptr=panel.query_ptr, labels=panel.labels, **scores[panel_name],
            )
        selected_source = (
            args.formal_v13_checkpoint if selected == "full23_v13"
            else args.candidate_dir / f"{selected}.pt"
        )
        shutil.copy2(selected_source, staging / "selected_model.pt")
        report = {
            "status": "FULL23_V13_CUMULATIVE_REPAIRS_GNPS_DEVELOPMENT_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development and model selection",
                "queries": {
                    name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
                },
            },
            "method_order": method_order,
            "selected": selected,
            "selected_source": str(selected_source),
            "selection_policy": {
                "primary_endpoint": "query-balanced unique-query Top-1 point estimate",
                "winner": "highest point estimate across formal v1.3 and six fixed arms",
                "tie_break": "earlier and simpler cumulative arm",
                "risk_net_lambda2": "descriptive only; never a veto",
            },
            "metrics": metrics,
            "comparisons": comparisons,
            "query_balanced_joint": joint,
            "residual_saturation": saturation,
            "trainable_parameters": parameter_counts,
            "claim_boundary": (
                "GNPS is consumed development and was used to choose the winner; "
                "this is not external confirmation."
            ),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
