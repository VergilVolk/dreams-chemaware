#!/usr/bin/env python3
"""Evaluate the fixed joint-evidence successor against formal Full23 v1.3."""
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

from evaluate_full23_multireference_on_gnps import BATCH_QUERIES, _load_torch
from full23_joint_evidence_set_core import joint_model_from_payload
from full23_multireference_readout_core import TOPK_ARM
from full23_tensor_set_readout_core import model_from_payload
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _comparison,
    _joint_query_balanced,
    _metric_summary,
)
from train_full23_joint_evidence_successor import ARM, pack_joint, spectral_sets
from train_full23_multireference_readout import (
    _load_banks,
    _make_representation_panels,
    _pack,
)
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
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--v13-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(comparator_embedding=[], comparator_score=[], full23_oof=None)
    return parser.parse_args()


@torch.inference_mode()
def predict_joint(model, panels, banks, spectral, device):
    output = {name: np.full(len(panel.base.labels), np.nan, np.float32)
              for name, panel in panels.items()}
    model.eval()
    for panel_name in PANELS:
        queries = np.arange(len(panels[panel_name].base.query_ptr)-1)
        for start in range(0, len(queries), BATCH_QUERIES):
            chosen = [(panel_name, int(query))
                      for query in queries[start:start+BATCH_QUERIES]]
            (anchor, _, query, refs, mask, ptr, _,
             spectral_values, spectral_mask) = pack_joint(panels, chosen, spectral)
            value = model(
                torch.as_tensor(anchor, device=device),
                torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device),
                banks,
                torch.as_tensor(spectral_values, device=device),
                torch.as_tensor(spectral_mask, device=device),
                query_ptr=ptr,
            ).float().cpu().numpy()
            cursor = 0
            for _, query_index in chosen:
                left, right = map(
                    int, panels[panel_name].base.query_ptr[query_index:query_index+2],
                )
                width = right-left
                output[panel_name][left:right] = value[cursor:cursor+width]
                cursor += width
    if any(not np.all(np.isfinite(value)) for value in output.values()):
        raise RuntimeError("joint prediction coverage incomplete")
    return output


@torch.inference_mode()
def predict_v13(model, panels, banks, device):
    output = {name: np.full(len(panel.base.labels), np.nan, np.float32)
              for name, panel in panels.items()}
    model.eval()
    for panel_name in PANELS:
        queries = np.arange(len(panels[panel_name].base.query_ptr)-1)
        for start in range(0, len(queries), BATCH_QUERIES):
            chosen = [(panel_name, int(query))
                      for query in queries[start:start+BATCH_QUERIES]]
            anchor, _, query, refs, mask, ptr, _ = _pack(panels, chosen, TOPK_ARM)
            value = model(
                torch.as_tensor(anchor, device=device),
                torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device),
                banks, query_ptr=ptr,
            ).float().cpu().numpy()
            cursor = 0
            for _, query_index in chosen:
                left, right = map(
                    int, panels[panel_name].base.query_ptr[query_index:query_index+2],
                )
                width = right-left
                output[panel_name][left:right] = value[cursor:cursor+width]
                cursor += width
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    full23 = validate_checkpoint(args.full23_checkpoint)
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(args, full23, cpu_banks)
    spectral, _, _ = spectral_sets(args, panels)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {name: torch.as_tensor(values, dtype=dtype, device=args.device)
             for name, (_, values) in cpu_banks.items()}
    del cpu_banks
    gc.collect()
    joint_payload = _load_torch(args.joint_checkpoint)
    v13_payload = _load_torch(args.v13_checkpoint)
    joint_scores = predict_joint(
        joint_model_from_payload(joint_payload, args.device),
        panels, banks, spectral, args.device,
    )
    v13_scores = predict_v13(
        model_from_payload(v13_payload, args.device), panels, banks, args.device,
    )

    scores, metrics, comparisons = {}, {}, {}
    for index, panel_name in enumerate(PANELS):
        panel = panels[panel_name].base
        scores[panel_name] = {
            "full23_v1": panels[panel_name].anchor,
            "full23_v13": v13_scores[panel_name],
            "joint_successor": joint_scores[panel_name],
        }
        metrics[panel_name] = {
            name: _metric_summary(value, panel)[0]
            for name, value in scores[panel_name].items()
        }
        comparisons[panel_name] = {
            "joint_vs_v13": _comparison(
                joint_scores[panel_name], v13_scores[panel_name], panel,
                5000, 20261102+index,
            ),
            "joint_vs_v1": _comparison(
                joint_scores[panel_name], panels[panel_name].anchor, panel,
                5000, 20261112+index,
            ),
        }
    joint = {}
    for method in ("full23_v1", "full23_v13", "joint_successor"):
        joint[method], _ = _joint_query_balanced(
            {name: panel.base for name, panel in panels.items()}, scores, method,
        )
    promote = (
        joint["joint_successor"]["top1_utility"]
        > joint["full23_v13"]["top1_utility"]
    )
    selected = "joint_successor" if promote else "full23_v13"

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
            "status": "FULL23_JOINT_EVIDENCE_SUCCESSOR_GNPS_DEVELOPMENT_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development",
                "queries": {name: len(panel.base.query_ptr)-1
                            for name, panel in panels.items()},
            },
            "selected": selected,
            "selection_policy": (
                "promote whenever query-balanced unique-query Top-1 point estimate "
                "exceeds formal v1.3; all panel and risk metrics remain descriptive"
            ),
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
