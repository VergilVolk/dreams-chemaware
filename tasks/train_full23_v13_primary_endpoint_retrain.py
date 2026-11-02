#!/usr/bin/env python3
"""Retrain Full23 v1.3 after removing its legacy 2x anchor-protection weight.

Every architecture, input, seed and optimizer setting remains the formal v1.3
recipe. The sole intervention is `anchor_correct_multiplier: 2 -> 1`, aligning
training with the query-balanced Top-1 endpoint instead of the retired
corrected-minus-two-introduced preference.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import torch

from full23_tensor_set_readout_core import SCHEMA
from train_enveda_unified_candidate_integrator import _occurrences
from train_full23_multireference_readout import (
    FULL23_OOF_SHA256,
    _embedding_score_alignment,
    _load_banks,
    _make_representation_panels,
    _reference_multiplicity,
    _training_input_hashes,
    arguments,
    calibrate_training_residual_bound,
)
from train_full23_tensor_set_readout import EPOCHS, TRAIN_SEED
from train_full23_transfer_safe_tensor_readout import ARMS, fit_arm
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


ARM = "v13_primary_endpoint"
SETTINGS = ARMS["no_count_query_centered"]
ANCHOR_CORRECT_MULTIPLIER = 1.0


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.full23_oof is None:
        raise RuntimeError("training requires Full23 v1 formula-OOF parent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 anchor identity drifted")
    if sha256_file(args.full23_oof) != FULL23_OOF_SHA256:
        raise RuntimeError("Full23 v1 OOF parent identity drifted")

    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(
        args, validate_checkpoint(args.full23_checkpoint), cpu_banks,
    )
    residual_bound, calibration = calibrate_training_residual_bound(panels)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()
    model, history = fit_arm(
        SETTINGS, panels, occurrences, banks, args.device, residual_bound,
        anchor_correct_multiplier=ANCHOR_CORRECT_MULTIPLIER,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        payload = {
            "schema": SCHEMA,
            "model_role": "Full23_v13_primary_endpoint_retrain",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "full23_parent_oof_sha256": FULL23_OOF_SHA256,
            "official_embedding_direct_input": False,
            "settings": SETTINGS.__dict__,
            "residual_bound": residual_bound,
            "residual_bound_calibration": calibration,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": {
                "source": "Enveda-180 consumed meta-development",
                "epochs": EPOCHS,
                "seed": TRAIN_SEED,
                "recipe": "formal v1.3 with primary-endpoint-aligned loss",
                "only_intervention": "anchor_correct_multiplier: 2.0 -> 1.0",
            },
            "upstream_sha256": _training_input_hashes(args),
        }
        torch.save(payload, staging / f"{ARM}.pt")
        report = {
            "status": "FULL23_V13_PRIMARY_ENDPOINT_RETRAIN_COMPLETE",
            "dataset": {
                "name": "Enveda-180",
                "role": "consumed meta-development",
                "queries": {name: len(panel.base.query_ptr)-1 for name, panel in panels.items()},
            },
            "architecture": "identical to formal Full23 v1.3",
            "only_intervention": {
                "field": "anchor_correct_multiplier",
                "formal_v13": 2.0,
                "candidate": ANCHOR_CORRECT_MULTIPLIER,
                "reason": "remove retired asymmetric risk preference from training",
            },
            "settings": SETTINGS.__dict__,
            "trainable_parameters": model.trainable_parameters,
            "training_history": history,
            "residual_bound": residual_bound,
            "reference_multiplicity": _reference_multiplicity(panels),
            "embedding_score_alignment_max_abs_drift": _embedding_score_alignment(panels),
            "gnps_access": "none",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
