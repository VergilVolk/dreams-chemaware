#!/usr/bin/env python3
"""Train the information-preserving Full23 tensor-set residual on Enveda OOF."""
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
    SCHEMA,
    Full23TensorSetReadout,
    TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
)
from train_enveda_unified_candidate_integrator import _group_occurrences, _occurrences
from train_full23_multireference_readout import (
    FULL23_OOF_SHA256,
    RepresentationPanel,
    _embedding_paths,
    _embedding_score_alignment,
    _load_banks,
    _make_representation_panels,
    _pack,
    _reference_multiplicity,
    _save_reference_registries,
    _seed,
    _training_input_hashes,
    arguments,
    calibrate_training_residual_bound,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


TRAIN_SEED = 20261031
EPOCHS = 6
BATCH_QUERIES = 48
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
ANCHOR_STRENGTH = 1e-2
PAIRWISE_MARGIN = 0.10
PAIRWISE_WEIGHT = 0.50
ANCHOR_CORRECT_MULTIPLIER = 2.0


def _fit(
    panels: dict[str, RepresentationPanel],
    occurrences: list[tuple[str, int]],
    banks: dict[str, torch.Tensor],
    device: str,
    residual_bound: float,
) -> tuple[Full23TensorSetReadout, list[dict[str, float]]]:
    _seed(TRAIN_SEED, device)
    model = Full23TensorSetReadout(TensorSetSettings(), residual_bound).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    groups = _group_occurrences(panels, occurrences)
    rng = np.random.default_rng(TRAIN_SEED)
    history: list[dict[str, float]] = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(groups))
        totals = {"loss": 0.0, "listwise": 0.0, "hardest_pairwise": 0.0}
        mass = 0.0
        model.train()
        for start in range(0, len(order), BATCH_QUERIES):
            chosen = [
                occurrence
                for group_index in order[start:start + BATCH_QUERIES]
                for occurrence in groups[int(group_index)]
            ]
            anchor, labels, query, refs, mask, ptr, weights = _pack(
                panels, chosen, TOPK_ARM,
            )
            anchor_t = torch.as_tensor(anchor, device=device)
            logits = model(
                anchor_t,
                torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device),
                banks,
            )
            ranking, parts = weighted_anchor_safe_ranking_loss(
                logits, anchor_t, ptr, labels, weights,
                pairwise_margin=PAIRWISE_MARGIN,
                pairwise_weight=PAIRWISE_WEIGHT,
                anchor_correct_multiplier=ANCHOR_CORRECT_MULTIPLIER,
            )
            residual_penalty = weighted_query_residual_penalty(
                logits - anchor_t, ptr, weights,
            )
            loss = ranking + ANCHOR_STRENGTH * residual_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_mass = float(np.sum(weights))
            totals["loss"] += float(loss.detach().cpu()) * batch_mass
            totals["listwise"] += float(parts["listwise"].detach().cpu()) * batch_mass
            totals["hardest_pairwise"] += float(
                parts["hardest_pairwise"].detach().cpu()
            ) * batch_mass
            mass += batch_mass
        row = {key: value / mass for key, value in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print(json.dumps({"arm": "tensor_set", **row}), flush=True)
    return model.eval(), history


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.full23_oof is None:
        raise RuntimeError("training requires Full23 formula-OOF parent predictions")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 anchor identity drifted")
    if sha256_file(args.full23_oof) != FULL23_OOF_SHA256:
        raise RuntimeError("Full23 OOF parent identity drifted")

    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    full23 = validate_checkpoint(args.full23_checkpoint)
    panels = _make_representation_panels(args, full23, cpu_banks)
    residual_bound, calibration = calibrate_training_residual_bound(panels)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    bank_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=bank_dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()
    model, history = _fit(panels, occurrences, banks, args.device, residual_bound)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        registries = _save_reference_registries(staging, panels)
        settings = TensorSetSettings()
        checkpoint = {
            "schema": SCHEMA,
            "model_role": "Full23_v1_anchored_information_preserving_tensor_set_increment",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "full23_parent_oof_sha256": FULL23_OOF_SHA256,
            "official_embedding_direct_input": False,
            "encoder_names": settings.encoder_names,
            "settings": settings.__dict__,
            "residual_bound": residual_bound,
            "residual_bound_calibration": calibration,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": {
                "source": "Enveda-180 consumed meta-development",
                "epochs": EPOCHS,
                "batch_queries": BATCH_QUERIES,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "anchor_strength": ANCHOR_STRENGTH,
                "pairwise_margin": PAIRWISE_MARGIN,
                "pairwise_weight": PAIRWISE_WEIGHT,
                "anchor_correct_multiplier": ANCHOR_CORRECT_MULTIPLIER,
                "seed": TRAIN_SEED,
                "loss": "listwise_plus_hardest_negative_plus_anchor_correct_protection",
            },
            "reference_registry_sha256": registries,
            "upstream_sha256": _training_input_hashes(args),
        }
        torch.save(checkpoint, staging / "full23_tensor_set_readout.pt")
        report = {
            "status": "FULL23_TENSOR_SET_READOUT_TRAINING_COMPLETE",
            "dataset": {
                "name": "Enveda-180",
                "role": "consumed meta-development fitting only",
                "performance_reported": False,
                "queries": {
                    name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
                },
            },
            "architecture": {
                "preinteraction_projection": False,
                "pair_interaction": "full_1024d_abs_difference_and_product",
                "reference_pooling": "masked_attention_mean_and_standard_deviation",
                "cross_encoder_interaction": "four_token_self_attention",
                "parent": "Full23_v1",
            },
            "trainable_parameters": model.trainable_parameters,
            "settings": settings.__dict__,
            "residual_bound": residual_bound,
            "training_history": history,
            "reference_multiplicity": _reference_multiplicity(panels),
            "embedding_score_alignment_max_abs_drift": _embedding_score_alignment(panels),
            "gnps_access": "none",
            "claim_boundary": (
                "No Enveda performance claim is emitted because both parent and tensor "
                "readout are fitted on Enveda. GNPS remains consumed development only."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "FULL23_TENSOR_SET_READOUT_TRAINING_COMPLETE",
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
