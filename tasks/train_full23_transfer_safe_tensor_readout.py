#!/usr/bin/env python3
"""Train two fixed transfer-safe Full23 tensor readout variants on Enveda."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from enveda_unified_candidate_integrator_core import weighted_query_residual_penalty
from full23_multireference_readout_core import TOPK_ARM
from full23_tensor_set_readout_core import (
    SCHEMA, Full23TensorSetReadout, TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
)
from train_enveda_unified_candidate_integrator import _group_occurrences, _occurrences
from train_full23_multireference_readout import (
    FULL23_OOF_SHA256, _embedding_score_alignment, _load_banks,
    _make_representation_panels, _pack, _reference_multiplicity, _seed,
    _training_input_hashes, arguments, calibrate_training_residual_bound,
)
from train_full23_tensor_set_readout import (
    ANCHOR_CORRECT_MULTIPLIER, ANCHOR_STRENGTH, BATCH_QUERIES, EPOCHS,
    LEARNING_RATE, PAIRWISE_MARGIN, PAIRWISE_WEIGHT, TRAIN_SEED, WEIGHT_DECAY,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


ARMS = {
    "no_reference_count": TensorSetSettings(include_reference_count=False),
    "no_count_query_centered": TensorSetSettings(
        include_reference_count=False, query_center_raw_residual=True,
    ),
}


def fit_arm(settings, panels, occurrences, banks, device, residual_bound):
    _seed(TRAIN_SEED, device)
    model = Full23TensorSetReadout(settings, residual_bound).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    groups = _group_occurrences(panels, occurrences)
    rng = np.random.default_rng(TRAIN_SEED)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(groups))
        totals = {"loss": 0.0, "listwise": 0.0, "hardest_pairwise": 0.0}
        mass = 0.0
        model.train()
        for start in range(0, len(order), BATCH_QUERIES):
            chosen = [item for group in order[start:start + BATCH_QUERIES]
                      for item in groups[int(group)]]
            anchor, labels, query, refs, mask, ptr, weights = _pack(
                panels, chosen, TOPK_ARM,
            )
            anchor_t = torch.as_tensor(anchor, device=device)
            logits = model(
                anchor_t, torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device), banks, query_ptr=ptr,
            )
            ranking, parts = weighted_anchor_safe_ranking_loss(
                logits, anchor_t, ptr, labels, weights,
                pairwise_margin=PAIRWISE_MARGIN, pairwise_weight=PAIRWISE_WEIGHT,
                anchor_correct_multiplier=ANCHOR_CORRECT_MULTIPLIER,
            )
            loss = ranking + ANCHOR_STRENGTH * weighted_query_residual_penalty(
                logits - anchor_t, ptr, weights,
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            batch_mass = float(np.sum(weights)); mass += batch_mass
            totals["loss"] += float(loss.detach().cpu()) * batch_mass
            totals["listwise"] += float(parts["listwise"].detach().cpu()) * batch_mass
            totals["hardest_pairwise"] += float(parts["hardest_pairwise"].detach().cpu()) * batch_mass
        row = {key: value / mass for key, value in totals.items()}
        row["epoch"] = epoch + 1; history.append(row)
        print(json.dumps({"arm": "transfer_safe", **row}), flush=True)
    return model.eval(), history


def main() -> None:
    args = arguments()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.full23_oof is None: raise RuntimeError("training requires Full23 OOF parent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 anchor identity drifted")
    if sha256_file(args.full23_oof) != FULL23_OOF_SHA256:
        raise RuntimeError("Full23 OOF parent identity drifted")
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(
        args, validate_checkpoint(args.full23_checkpoint), cpu_banks,
    )
    residual_bound, calibration = calibrate_training_residual_bound(panels)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {name: torch.as_tensor(values, dtype=dtype, device=args.device)
             for name, (_, values) in cpu_banks.items()}
    del cpu_banks; gc.collect()
    models, histories = {}, {}
    for name, settings in ARMS.items():
        models[name], histories[name] = fit_arm(
            settings, panels, occurrences, banks, args.device, residual_bound,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for name, model in models.items():
            payload = {
                "schema": SCHEMA, "model_role": f"Full23_transfer_safe_{name}",
                "full23_checkpoint_sha256": CHECKPOINT_SHA256,
                "full23_parent_oof_sha256": FULL23_OOF_SHA256,
                "official_embedding_direct_input": False,
                "settings": ARMS[name].__dict__, "residual_bound": residual_bound,
                "residual_bound_calibration": calibration,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "training": {"source": "Enveda-180 consumed meta-development",
                             "epochs": EPOCHS, "seed": TRAIN_SEED,
                             "recipe": "fixed transfer shortcut repair"},
                "upstream_sha256": _training_input_hashes(args),
            }
            torch.save(payload, staging / f"{name}.pt")
        report = {
            "status": "FULL23_TRANSFER_SAFE_TENSOR_TRAINING_COMPLETE",
            "dataset": {"name": "Enveda-180", "role": "consumed meta-development",
                        "queries": {n: len(p.base.query_ptr)-1 for n,p in panels.items()}},
            "arms": {name: {"settings": settings.__dict__,
                            "trainable_parameters": models[name].trainable_parameters,
                            "training_history": histories[name]}
                     for name, settings in ARMS.items()},
            "residual_bound": residual_bound,
            "reference_multiplicity": _reference_multiplicity(panels),
            "embedding_score_alignment_max_abs_drift": _embedding_score_alignment(panels),
            "gnps_access": "none",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps({"status": "FULL23_TRANSFER_SAFE_TENSOR_TRAINING_COMPLETE",
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__": main()
