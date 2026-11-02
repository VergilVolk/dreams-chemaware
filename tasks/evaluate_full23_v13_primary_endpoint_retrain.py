#!/usr/bin/env python3
"""Compare the one-factor v1.3 loss repair with formal Full23 v1.3."""
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
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
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
    banks = {name: torch.as_tensor(values, dtype=dtype, device=args.device)
             for name, (_, values) in cpu_banks.items()}
    del cpu_banks
    gc.collect()
    formal_scores = predict(
        model_from_payload(_load_torch(args.formal_v13_checkpoint), args.device),
        panels, banks, args.device,
    )
    candidate_scores = predict(
        model_from_payload(_load_torch(args.candidate_checkpoint), args.device),
        panels, banks, args.device,
    )

    scores, metrics, comparisons = {}, {}, {}
    for index, panel_name in enumerate(PANELS):
        panel = panels[panel_name].base
        scores[panel_name] = {
            "full23_v1": panels[panel_name].anchor,
            "full23_v13": formal_scores[panel_name],
            "primary_endpoint_retrain": candidate_scores[panel_name],
        }
        metrics[panel_name] = {
            name: _metric_summary(value, panel)[0]
            for name, value in scores[panel_name].items()
        }
        comparisons[panel_name] = {
            "candidate_vs_v13": _comparison(
                candidate_scores[panel_name], formal_scores[panel_name], panel,
                5000, 20261140+index,
            ),
            "candidate_vs_v1": _comparison(
                candidate_scores[panel_name], panels[panel_name].anchor, panel,
                5000, 20261150+index,
            ),
        }
    joint = {}
    for method in ("full23_v1", "full23_v13", "primary_endpoint_retrain"):
        joint[method], _ = _joint_query_balanced(
            {name: panel.base for name, panel in panels.items()}, scores, method,
        )
    selected = (
        "primary_endpoint_retrain"
        if joint["primary_endpoint_retrain"]["top1_utility"]
        > joint["full23_v13"]["top1_utility"]
        else "full23_v13"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel = panels[panel_name].base
            np.savez_compressed(
                staging / f"candidate_predictions_{panel_name}.npz",
                query_ptr=panel.query_ptr, labels=panel.labels, **scores[panel_name],
            )
        report = {
            "status": "FULL23_V13_PRIMARY_ENDPOINT_RETRAIN_GNPS_DEVELOPMENT_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development",
                "queries": {name: len(panel.base.query_ptr)-1
                            for name, panel in panels.items()},
            },
            "selected": selected,
            "only_intervention": "anchor_correct_multiplier: 2.0 -> 1.0",
            "primary_endpoint": "query-balanced unique-query Top-1 point estimate",
            "metrics": metrics,
            "comparisons": comparisons,
            "query_balanced_joint": joint,
            "claim_boundary": "Consumed development only; not external evidence.",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "selected": selected,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
