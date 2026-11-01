#!/usr/bin/env python3
"""Train one Full23 successor without prematurely pooling reference evidence.

The frozen Full23 v1 score is the anchor. Four complete project embeddings and
four per-reference spectral channels are read jointly. Reference-count
shortcuts are disabled and the bounded residual is query-centred, preserving
the two repairs that produced Full23 v1.3.
"""
from __future__ import annotations

from dataclasses import asdict
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
from full23_joint_evidence_set_core import (
    SPECTRAL_NAMES,
    Full23JointEvidenceSetReadout,
    JointEvidenceSettings,
)
from full23_multireference_readout_core import TOPK_ARM
from full23_tensor_set_readout_core import (
    TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
)
from gnps_pair_score_cache import load_pair_score_cache
from train_enveda_unified_candidate_integrator import (
    MSG,
    PANELS,
    _group_occurrences,
    _load_score_bundle,
    _occurrences,
)
from train_full23_multireference_readout import (
    _load_banks,
    _make_representation_panels,
    _pack,
    _reference_multiplicity,
    _seed,
    _training_input_hashes,
    arguments,
    calibrate_training_residual_bound,
)
from train_full23_tensor_set_readout import (
    ANCHOR_CORRECT_MULTIPLIER,
    ANCHOR_STRENGTH,
    BATCH_QUERIES,
    EPOCHS,
    LEARNING_RATE,
    PAIRWISE_MARGIN,
    PAIRWISE_WEIGHT,
    TRAIN_SEED,
    WEIGHT_DECAY,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


SCHEMA = "full23_joint_evidence_set_readout_v1"
ARM = "joint_embedding_spectral_query_centered"
SETTINGS = JointEvidenceSettings(
    embedding=TensorSetSettings(
        include_reference_count=False,
        query_center_raw_residual=False,
    ),
    include_spectral_count=False,
    query_center_raw_residual=True,
)


def spectral_sets(args, panels):
    methods, method_scores = _load_score_bundle(args.score_bundle, args.benchmark)
    _, msg_pairs = load_pair_score_cache(args.msg_cache, args.benchmark)
    lookup = {name: index for index, name in enumerate(methods)}
    required = ("noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen")
    missing = [name for name in required if name not in lookup]
    if missing:
        raise RuntimeError(f"spectral score bundle misses {missing}")
    result = {}
    all_values = []
    for panel_name in PANELS:
        graph = panels[panel_name].base.graph
        pair = np.stack((
            method_scores[panel_name][lookup["noise_v1"]],
            method_scores[panel_name][lookup["weighted_spectral_entropy"]],
            method_scores[panel_name][lookup["p2b_noise_v1_frozen"]],
            np.asarray(msg_pairs[panel_name], dtype=np.float32),
        ), axis=1).astype(np.float32)
        if pair.shape != (len(graph.pair_candidate_row), len(SPECTRAL_NAMES)):
            raise RuntimeError(f"spectral pair axis drifted for {panel_name}")
        if not np.all(np.isfinite(pair)):
            raise RuntimeError(f"non-finite spectral evidence in {panel_name}")
        counts = np.diff(graph.molecule_ptr)
        if np.any(counts < 1) or int(np.max(counts)) > SETTINGS.embedding.top_k:
            raise RuntimeError(f"reference multiplicity exceeds contract in {panel_name}")
        candidate_count = len(counts)
        tensor = np.zeros(
            (candidate_count, SETTINGS.embedding.top_k, len(SPECTRAL_NAMES)),
            dtype=np.float32,
        )
        mask = np.zeros((candidate_count, SETTINGS.embedding.top_k), dtype=bool)
        for candidate, (left, right) in enumerate(
            zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])
        ):
            left, right = int(left), int(right)
            tensor[candidate, :right-left] = pair[left:right]
            mask[candidate, :right-left] = True
        result[panel_name] = (tensor, mask)
        all_values.append(pair)
    values = np.concatenate(all_values, axis=0).astype(np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    scale = np.maximum(values.std(axis=0).astype(np.float32), np.float32(1e-6))
    return result, mean, scale


def pack_joint(panels, occurrences, spectral):
    base = _pack(panels, occurrences, TOPK_ARM)
    spectral_values, spectral_masks = [], []
    for panel_name, query in occurrences:
        panel = panels[panel_name]
        left, right = map(int, panel.base.query_ptr[query:query+2])
        values, mask = spectral[panel_name]
        spectral_values.append(values[left:right])
        spectral_masks.append(mask[left:right])
    return (*base, np.concatenate(spectral_values), np.concatenate(spectral_masks))


def fit(panels, occurrences, banks, spectral, mean, scale, device, residual_bound):
    _seed(TRAIN_SEED, device)
    model = Full23JointEvidenceSetReadout(
        SETTINGS, residual_bound, mean, scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    groups = _group_occurrences(panels, occurrences)
    rng = np.random.default_rng(TRAIN_SEED)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(groups))
        totals = {"loss": 0.0, "listwise": 0.0, "hardest_pairwise": 0.0}
        mass = 0.0
        model.train()
        for start in range(0, len(order), BATCH_QUERIES):
            chosen = [
                item for group in order[start:start+BATCH_QUERIES]
                for item in groups[int(group)]
            ]
            (anchor, labels, query, refs, mask, ptr, weights,
             spectral_values, spectral_mask) = pack_joint(panels, chosen, spectral)
            anchor_t = torch.as_tensor(anchor, device=device)
            logits = model(
                anchor_t,
                torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device),
                banks,
                torch.as_tensor(spectral_values, device=device),
                torch.as_tensor(spectral_mask, device=device),
                query_ptr=ptr,
            )
            ranking, parts = weighted_anchor_safe_ranking_loss(
                logits, anchor_t, ptr, labels, weights,
                pairwise_margin=PAIRWISE_MARGIN,
                pairwise_weight=PAIRWISE_WEIGHT,
                anchor_correct_multiplier=ANCHOR_CORRECT_MULTIPLIER,
            )
            loss = ranking + ANCHOR_STRENGTH * weighted_query_residual_penalty(
                logits - anchor_t, ptr, weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_mass = float(np.sum(weights))
            mass += batch_mass
            totals["loss"] += float(loss.detach().cpu()) * batch_mass
            totals["listwise"] += float(parts["listwise"].detach().cpu()) * batch_mass
            totals["hardest_pairwise"] += (
                float(parts["hardest_pairwise"].detach().cpu()) * batch_mass
            )
        row = {key: value / mass for key, value in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print(json.dumps({"arm": ARM, **row}), flush=True)
    return model.eval(), history


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
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(
        args, validate_checkpoint(args.full23_checkpoint), cpu_banks,
    )
    spectral, spectral_mean, spectral_scale = spectral_sets(args, panels)
    residual_bound, calibration = calibrate_training_residual_bound(panels)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()
    model, history = fit(
        panels, occurrences, banks, spectral, spectral_mean, spectral_scale,
        args.device, residual_bound,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        payload = {
            "schema": SCHEMA,
            "model_role": "Full23_joint_embedding_spectral_query_centered",
            "settings": asdict(SETTINGS),
            "residual_bound": residual_bound,
            "residual_bound_calibration": calibration,
            "spectral_mean": spectral_mean,
            "spectral_scale": spectral_scale,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": {
                "source": "Enveda-180 consumed meta-development",
                "parent": "Full23 v1 formula-OOF scores",
                "epochs": EPOCHS,
                "seed": TRAIN_SEED,
                "recipe": "joint full-embedding and per-reference spectral evidence",
            },
            "upstream_sha256": _training_input_hashes(args),
        }
        torch.save(payload, staging / f"{ARM}.pt")
        report = {
            "status": "FULL23_JOINT_EVIDENCE_SUCCESSOR_TRAINING_COMPLETE",
            "dataset": {
                "name": "Enveda-180",
                "role": "consumed meta-development",
                "queries": {name: len(panel.base.query_ptr)-1 for name, panel in panels.items()},
            },
            "arm": ARM,
            "settings": asdict(SETTINGS),
            "trainable_parameters": model.trainable_parameters,
            "training_history": history,
            "residual_bound": residual_bound,
            "spectral_normalization": {
                "names": list(SPECTRAL_NAMES),
                "mean": spectral_mean.tolist(),
                "scale": spectral_scale.tolist(),
                "source": "Enveda training reference edges only",
            },
            "reference_multiplicity": _reference_multiplicity(panels),
            "gnps_access": "none",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "FULL23_JOINT_EVIDENCE_SUCCESSOR_TRAINING_COMPLETE",
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
