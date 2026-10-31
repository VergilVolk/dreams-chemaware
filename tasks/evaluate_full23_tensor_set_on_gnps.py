#!/usr/bin/env python3
"""Evaluate one frozen Full23 tensor-set readout on consumed GNPS development."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from evaluate_full23_multireference_on_gnps import (
    BOOTSTRAP_RESAMPLES,
    _input_hashes,
    _load_torch,
    _predict,
    arguments,
)
from full23_multireference_readout_core import TOPK_ARM
from full23_tensor_set_readout_core import SCHEMA, model_from_payload
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _comparison,
    _joint_query_balanced,
    _metric_summary,
)
from train_full23_multireference_readout import (
    _load_banks,
    _make_representation_panels,
    _reference_multiplicity,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


METHOD = "full23_tensor_set"


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    full23 = validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 checkpoint identity drifted")
    payload = _load_torch(args.readout_checkpoint)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("tensor-set checkpoint schema drifted")
    if payload.get("full23_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("tensor-set parent differs from Full23 v1")

    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(args, full23, cpu_banks)
    bank_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=bank_dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    model = model_from_payload(payload, args.device)
    predicted = _predict(model, TOPK_ARM, panels, banks, args.device)

    scores: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict]] = {}
    comparisons: dict[str, dict] = {}
    for panel_index, panel_name in enumerate(PANELS):
        panel = panels[panel_name].base
        scores[panel_name] = {
            "full23_v1": panels[panel_name].anchor,
            METHOD: predicted[panel_name],
            "official_dreams": panel.scores["official_dreams"],
            "weighted_spectral_entropy": panel.scores["weighted_spectral_entropy"],
            "noise_v1": panel.scores["noise_v1"],
            "p2b_noise_v1_frozen": panel.scores["p2b_noise_v1_frozen"],
            "noise_msg_fusion_stack_v1": panel.scores["noise_msg_fusion_stack_v1"],
            "chemaware_stage1": panel.scores["chemaware_stage1"],
            "chemaware_phasea": panel.scores["chemaware_phasea"],
            "noise_chemaware_fullmsg": panel.scores["noise_chemaware_fullmsg"],
        }
        metrics[panel_name] = {
            name: _metric_summary(value, panel)[0]
            for name, value in scores[panel_name].items()
        }
        comparisons[panel_name] = _comparison(
            predicted[panel_name], panels[panel_name].anchor, panel,
            BOOTSTRAP_RESAMPLES, 20261031 + panel_index,
        )

    joint = {}
    for method in ("full23_v1", METHOD):
        joint[method], _ = _joint_query_balanced(
            {name: panel.base for name, panel in panels.items()}, scores, method,
        )
    eligible = all(
        float(comparisons[name]["recall1_formula_cluster_ci"]["ci_low"]) > 0
        and int(comparisons[name]["corrected"]) > int(comparisons[name]["introduced"])
        and metrics[name][METHOD]["near"]["recall@1"]
            >= metrics[name]["full23_v1"]["near"]["recall@1"]
        for name in PANELS
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel = panels[panel_name].base
            np.savez_compressed(
                staging / f"candidate_predictions_{panel_name}.npz",
                query_ptr=panel.query_ptr,
                query_ids=panel.query_keys,
                candidate_ids=panel.graph.molecule_ik14,
                labels=panel.labels,
                **scores[panel_name],
            )
        report = {
            "status": "FULL23_TENSOR_SET_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development",
                "allow_final_external_claim": False,
                "queries": {
                    name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
                },
            },
            "selected": METHOD if eligible else "full23_v1",
            "promotion_from_full23_v1": bool(eligible),
            "metrics": metrics,
            "comparisons": {
                name: {f"{METHOD}_vs_full23_v1": value}
                for name, value in comparisons.items()
            },
            "query_balanced_joint": joint,
            "reference_multiplicity": _reference_multiplicity(panels),
            "checkpoint_sha256": sha256_file(args.readout_checkpoint),
            "input_sha256": _input_hashes(args),
            "claim_boundary": (
                "GNPS is consumed development. This evaluation can reject or retain a "
                "readout but cannot establish an untouched external or SOTA claim."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "FULL23_TENSOR_SET_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
        "selected": METHOD if eligible else "full23_v1",
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
