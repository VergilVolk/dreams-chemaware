#!/usr/bin/env python3
"""Fit the one Full23-v1 anchored multi-reference representation residual.

The Full23 Additive checkpoint and all four encoders are frozen.  This program
uses consumed Enveda only as fitting data and deliberately emits no Enveda
performance claim.  A separate program performs consumed-GNPS development
adjudication without refitting this model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from enveda_unified_candidate_integrator_core import (
    CONTROL_ARM,
    predict_from_payload,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from evaluate_gnps_gold_silver_10ppm_embeddings import load_embeddings
from full23_multireference_readout_core import (
    ARMS,
    ENCODER_NAMES,
    SCHEMA,
    TOP_K,
    Full23AnchoredMultiReferenceReadout,
    MultiReferenceSettings,
    TopKReferenceIndex,
    arm_mask,
    stable_topk_references,
)
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _group_occurrences,
    _make_panels,
    _occurrences,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


TRAIN_SEED = 20261027
EPOCHS = 3
BATCH_QUERIES = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ANCHOR_STRENGTH = 1e-2
PROJECTION_DIMENSION = 16
FUSION_HIDDEN = 32
FULL23_OOF_SHA256 = "0d2f8a7a07c701b5482fd0f415cd608fcc71e878a0eb73630f674dcb81111d19"


@dataclass
class RepresentationPanel:
    base: object
    anchor: np.ndarray
    topk: dict[str, TopKReferenceIndex]

    @property
    def query_keys(self) -> np.ndarray:
        return self.base.query_keys

    @property
    def query_weights(self) -> np.ndarray:
        return self.base.query_weights


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
    parser.add_argument(
        "--full23-oof", type=Path, default=None,
        help=(
            "Required for fitting: five-fold OOF Full23 predictions on the training "
            "benchmark. Evaluation omits this argument and uses the final checkpoint."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Private loader compatibility: comparators never enter this experiment.
    parser.set_defaults(comparator_embedding=[], comparator_score=[])
    args = parser.parse_args()
    registered = {
        "epochs": EPOCHS,
        "batch_queries": BATCH_QUERIES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "anchor_strength": ANCHOR_STRENGTH,
        "projection_dimension": PROJECTION_DIMENSION,
        "fusion_hidden": FUSION_HIDDEN,
        "top_k": TOP_K,
    }
    for key, value in registered.items():
        setattr(args, key, value)
    return args


def _seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _embedding_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "noise_v1": args.noise_embeddings,
        "chemaware_stage1": args.stage1_embeddings,
        "chemaware_phasea": args.phasea_embeddings,
        "noise_chemaware_fullmsg": args.fullmsg_embeddings,
    }
    if tuple(paths) != ENCODER_NAMES or any("official" in name for name in paths):
        raise RuntimeError("representation input registry drifted or contains official DreaMS")
    return paths


def _load_banks(
    args: argparse.Namespace, manifest_rows: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {
        name: load_embeddings(path, manifest_rows)
        for name, path in _embedding_paths(args).items()
    }
    dimensions = {embedding.shape[1] for _, embedding in result.values()}
    if dimensions != {1024}:
        raise RuntimeError(f"registered encoder dimensions drifted: {dimensions}")
    return result


def _make_representation_panels(
    args: argparse.Namespace,
    full23: dict,
    banks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, RepresentationPanel]:
    base_panels, _ = _make_panels(args)
    oof_anchors = (
        _load_full23_oof(args.full23_oof, base_panels)
        if args.full23_oof is not None else None
    )
    result: dict[str, RepresentationPanel] = {}
    for panel_name in PANELS:
        base = base_panels[panel_name]
        if oof_anchors is None:
            anchor = predict_from_payload(
                full23, base.feature_names, base.scalar, arm=CONTROL_ARM, device=args.device,
            )
        else:
            anchor = oof_anchors[panel_name]
        topk = {
            name: stable_topk_references(base.graph, rows, embedding, TOP_K)
            for name, (rows, embedding) in banks.items()
        }
        for name, index in topk.items():
            expected = np.asarray(base.scores[name], dtype=np.float32)
            observed = np.asarray(index.molecule_scores, dtype=np.float32)
            if observed.shape != expected.shape:
                raise RuntimeError(
                    f"embedding/Full23 scalar axis drifted for {panel_name}: {name}"
                )
            maximum_drift = float(np.max(np.abs(observed - expected), initial=0.0))
            if not np.allclose(observed, expected, rtol=0.0, atol=2e-6):
                raise RuntimeError(
                    f"embedding cache differs from Full23 scalar evidence for "
                    f"{panel_name}: {name}; max_abs_drift={maximum_drift:.9g}"
                )
        first_counts = next(iter(topk.values())).reference_mask.sum(axis=1)
        if any(
            not np.array_equal(index.reference_mask.sum(axis=1), first_counts)
            for index in topk.values()
        ):
            raise RuntimeError(
                f"reference multiplicity differs across encoders in {panel_name}"
            )
        result[panel_name] = RepresentationPanel(
            base=base,
            anchor=anchor,
            topk=topk,
        )
    return result


def _load_full23_oof(path: Path, panels: dict[str, object]) -> dict[str, np.ndarray]:
    """Load parent predictions that excluded each Enveda formula during fitting."""
    if not path.is_file():
        raise FileNotFoundError(path)
    report_path = path.with_name("report.json")
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        parent_report.get("control_model") != CONTROL_ARM
        or parent_report.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        or parent_report.get("prediction_ledger", {}).get("sha256") != FULL23_OOF_SHA256
        or sha256_file(path) != FULL23_OOF_SHA256
        or parent_report.get("features", {}).get("feature_selection_policy")
        != "exclude_query_constant_context_v1"
    ):
        raise RuntimeError("Full23 OOF parent is not the released 23-feature v1 run")
    with np.load(path, allow_pickle=False) as body:
        archive = {name: np.asarray(body[name]) for name in body.files}
    output: dict[str, np.ndarray] = {}
    score_checks = (
        "official_dreams", "weighted_spectral_entropy", "noise_v1",
        "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
        "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
    )
    for panel_name in PANELS:
        panel = panels[panel_name]
        prefix = f"{panel_name}__"
        required = {
            prefix + "query_ptr", prefix + "labels", prefix + "query_keys",
            prefix + "formula", prefix + "query_weight",
            prefix + CONTROL_ARM,
            *(prefix + name for name in score_checks),
        }
        missing = required - archive.keys()
        if missing:
            raise RuntimeError(f"Full23 OOF archive misses {sorted(missing)}")
        axis_checks = {
            "query_ptr": np.array_equal(archive[prefix + "query_ptr"], panel.query_ptr),
            "labels": np.array_equal(archive[prefix + "labels"], panel.labels),
            "query_keys": np.array_equal(
                archive[prefix + "query_keys"].astype("U"), panel.query_keys.astype("U"),
            ),
            "formula": np.array_equal(
                archive[prefix + "formula"].astype("U"), panel.formulas.astype("U"),
            ),
            "query_weight": np.allclose(
                archive[prefix + "query_weight"], panel.query_weights,
                rtol=0.0, atol=1e-7,
            ),
        }
        failed = [name for name, valid in axis_checks.items() if not valid]
        if failed:
            raise RuntimeError(f"Full23 OOF {panel_name} axis drifted: {failed}")
        for name in score_checks:
            observed = np.asarray(archive[prefix + name], dtype=np.float32)
            expected = np.asarray(panel.scores[name], dtype=np.float32)
            if observed.shape != expected.shape or not np.allclose(
                observed, expected, rtol=0.0, atol=2e-6,
            ):
                raise RuntimeError(
                    f"Full23 OOF candidate order/evidence drifted for {panel_name}: {name}"
                )
        score = np.asarray(archive[prefix + CONTROL_ARM], dtype=np.float32)
        if score.shape != panel.labels.shape or not np.all(np.isfinite(score)):
            raise RuntimeError(f"Full23 OOF parent score is malformed for {panel_name}")
        output[panel_name] = score
    return output


def _pack(
    panels: dict[str, RepresentationPanel],
    occurrences: list[tuple[str, int]],
    arm: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    anchors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    query_positions: list[np.ndarray] = []
    reference_positions: list[np.ndarray] = []
    reference_masks: list[np.ndarray] = []
    lengths: list[int] = []
    weights: list[float] = []
    for name, query in occurrences:
        panel = panels[name]
        left, right = map(int, panel.base.query_ptr[query:query + 2])
        anchors.append(panel.anchor[left:right])
        labels.append(panel.base.labels[left:right])
        query_positions.append(np.stack([
            panel.topk[encoder].query_positions[left:right] for encoder in ENCODER_NAMES
        ], axis=1))
        refs = np.stack([
            panel.topk[encoder].reference_positions[left:right]
            for encoder in ENCODER_NAMES
        ], axis=1)
        masks = np.stack([
            panel.topk[encoder].reference_mask[left:right] for encoder in ENCODER_NAMES
        ], axis=1)
        reference_positions.append(refs)
        reference_masks.append(arm_mask(masks, arm))
        lengths.append(right - left)
        weights.append(float(panel.base.query_weights[query]))
    ptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return (
        np.concatenate(anchors).astype(np.float32),
        np.concatenate(labels).astype(np.int8),
        np.concatenate(query_positions).astype(np.int64),
        np.concatenate(reference_positions).astype(np.int64),
        np.concatenate(reference_masks).astype(bool),
        ptr,
        np.asarray(weights, dtype=np.float32),
    )


def _fit_arm(
    arm: str,
    panels: dict[str, RepresentationPanel],
    occurrences: list[tuple[str, int]],
    banks: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[Full23AnchoredMultiReferenceReadout, list[float]]:
    # Same initialization and batch order make Top-1 vs Top-3 capacity matched.
    _seed(TRAIN_SEED, args.device)
    model = Full23AnchoredMultiReferenceReadout(MultiReferenceSettings()).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    groups = _group_occurrences(panels, occurrences)
    rng = np.random.default_rng(TRAIN_SEED)
    history: list[float] = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(groups))
        total, mass = 0.0, 0.0
        model.train()
        for start in range(0, len(order), BATCH_QUERIES):
            chosen = [
                occurrence
                for group_index in order[start:start + BATCH_QUERIES]
                for occurrence in groups[int(group_index)]
            ]
            anchor, labels, query, refs, mask, ptr, weights = _pack(panels, chosen, arm)
            anchor_t = torch.as_tensor(anchor, device=args.device)
            query_t = torch.as_tensor(query, device=args.device)
            refs_t = torch.as_tensor(refs, device=args.device)
            mask_t = torch.as_tensor(mask, device=args.device)
            logits = model(anchor_t, query_t, refs_t, mask_t, banks)
            loss = weighted_listwise_loss(logits, ptr, labels, weights)
            loss = loss + ANCHOR_STRENGTH * weighted_query_residual_penalty(
                logits - anchor_t, ptr, weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_mass = float(np.sum(weights))
            total += float(loss.detach().cpu()) * batch_mass
            mass += batch_mass
        history.append(total / mass)
        print(json.dumps({
            "arm": arm, "epoch": epoch + 1, "training_loss": history[-1],
        }), flush=True)
    return model.eval(), history


def _save_reference_registries(
    directory: Path, panels: dict[str, RepresentationPanel],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for panel_name, panel in panels.items():
        path = directory / f"top3_references_{panel_name}.npz"
        payload: dict[str, np.ndarray] = {
            "encoder_names": np.asarray(ENCODER_NAMES),
            "query_ptr": panel.base.query_ptr,
            "query_ids": panel.base.query_keys,
            "candidate_ids": panel.base.graph.molecule_ik14,
        }
        for encoder in ENCODER_NAMES:
            payload[f"reference_rows__{encoder}"] = panel.topk[encoder].reference_rows
            payload[f"reference_mask__{encoder}"] = panel.topk[encoder].reference_mask
        np.savez_compressed(path, **payload)
        hashes[panel_name] = sha256_file(path)
    return hashes


def _reference_multiplicity(
    panels: dict[str, RepresentationPanel],
) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, dict[str, int]]] = {}
    for panel_name, panel in panels.items():
        output[panel_name] = {}
        for encoder, index in panel.topk.items():
            counts = index.reference_mask.sum(axis=1)
            output[panel_name][encoder] = {
                str(width): int(np.sum(counts == width)) for width in range(1, TOP_K + 1)
            }
    return output


def _embedding_score_alignment(
    panels: dict[str, RepresentationPanel],
) -> dict[str, dict[str, float]]:
    """Report the exact scalar/readout identity already enforced at construction."""
    return {
        panel_name: {
            encoder: float(np.max(np.abs(
                np.asarray(index.molecule_scores, dtype=np.float32)
                - np.asarray(panel.base.scores[encoder], dtype=np.float32)
            ), initial=0.0))
            for encoder, index in panel.topk.items()
        }
        for panel_name, panel in panels.items()
    }


def _training_input_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths: dict[str, Path] = {
        "manifest": args.benchmark / "manifest.csv.gz",
        "score_bundle": args.score_bundle,
        "msg_report": args.msg_cache / "report.json",
        "chem_report": args.chem_dir / "report.json",
        "full23_checkpoint": args.full23_checkpoint,
        "full23_parent_oof": args.full23_oof,
        "full23_parent_oof_report": args.full23_oof.with_name("report.json"),
        **{f"embedding_{name}": path for name, path in _embedding_paths(args).items()},
    }
    for panel_name in PANELS:
        paths[f"panel_{panel_name}"] = args.benchmark / f"panel_{panel_name}.npz"
        paths[f"msg_pairs_{panel_name}"] = (
            args.msg_cache / f"pair_scores_{panel_name}.npy"
        )
        paths[f"chem_{panel_name}"] = (
            args.chem_dir / f"continuous_chem_{panel_name}.npz"
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training hash registry misses inputs: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    if args.full23_oof is None:
        raise RuntimeError("training requires leakage-safe Full23 OOF parent predictions")
    full23 = validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 anchor identity drifted")
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    if not np.array_equal(manifest.row.to_numpy(np.int64), np.arange(len(manifest))):
        raise RuntimeError("benchmark manifest row registry is not contiguous")
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(args, full23, cpu_banks)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    bank_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    gpu_banks = {
        name: torch.as_tensor(embedding, dtype=bank_dtype, device=args.device)
        for name, (_, embedding) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()

    models: dict[str, Full23AnchoredMultiReferenceReadout] = {}
    histories: dict[str, list[float]] = {}
    for arm in ARMS:
        models[arm], histories[arm] = _fit_arm(
            arm, panels, occurrences, gpu_banks, args,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        registry_hashes = _save_reference_registries(staging, panels)
        settings = MultiReferenceSettings()
        checkpoint = {
            "schema": SCHEMA,
            "model_role": "Full23_v1_anchored_representation_increment",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "encoder_names": ENCODER_NAMES,
            "official_embedding_direct_input": False,
            "full23_anchor_has_indirect_official_dependency": True,
            "top_k": TOP_K,
            "topk_selection": "per-encoder raw cosine; stable row-id tie break; no duplication",
            "pooling": "masked_mean_plus_max_of_16d_pair_tokens",
            "residual": "unbounded_single_residual_with_query_balanced_l2_penalty",
            "settings": settings.__dict__,
            "arms": {
                arm: {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for arm, model in models.items()
            },
            "training": {
                "source": "Enveda-180 consumed meta-development",
                "parent_prediction_role": "five-fold formula-OOF Full23 v1",
                "parent_oof_sha256": sha256_file(args.full23_oof),
                "evaluation_on_training_source": False,
                "epochs": EPOCHS,
                "batch_queries": BATCH_QUERIES,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "anchor_strength": ANCHOR_STRENGTH,
                "seed": TRAIN_SEED,
                "unique_query_occurrence_weighting": True,
            },
            "reference_registry_sha256": registry_hashes,
            "upstream_sha256": _training_input_hashes(args),
        }
        torch.save(checkpoint, staging / "full23_multireference_readout.pt")
        report = {
            "status": "FULL23_MULTIREFERENCE_READOUT_TRAINING_COMPLETE",
            "dataset": {
                "name": "Enveda-180",
                "role": "consumed meta-development fitting only",
                "performance_reported": False,
                "queries": {
                    name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
                },
            },
            "anchor": "frozen Full23 Additive v1",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "full23_parent_oof_sha256": sha256_file(args.full23_oof),
            "encoder_names": list(ENCODER_NAMES),
            "official_embedding_direct_input": False,
            "full23_anchor_has_indirect_official_dependency": True,
            "arms": list(ARMS),
            "primary_arm": ARMS[-1],
            "mechanism_control": ARMS[0],
            "top_k": TOP_K,
            "trainable_parameters": models[ARMS[0]].trainable_parameters,
            "matched_parameter_counts": (
                models[ARMS[0]].trainable_parameters == models[ARMS[1]].trainable_parameters
            ),
            "training_history": histories,
            "reference_registry_sha256": registry_hashes,
            "reference_multiplicity": _reference_multiplicity(panels),
            "embedding_score_alignment_max_abs_drift": _embedding_score_alignment(panels),
            "gnps_access": "none",
            "claim_boundary": (
                "No Enveda performance claim is emitted because Full23 and this residual "
                "are both fitted on Enveda. The separately frozen GNPS program is a "
                "consumed development stress test, not untouched external validation."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "FULL23_MULTIREFERENCE_READOUT_TRAINING_COMPLETE",
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
