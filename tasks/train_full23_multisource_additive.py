#!/usr/bin/env python
"""Refit Full23 Additive on source-balanced Enveda plus GNPS development.

Both resources are already consumed by method development.  This program stops
pretending that either remains an untouched external test and uses them for one
shared, domain-agnostic decision layer.  The 23 frozen candidate-evidence
features, optimization settings, and WSE anchor are unchanged.  No source
indicator is supplied.  Each source contributes total training mass 0.5 and a
global formula hash prevents a formula from crossing OOF folds in either
source.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CONTROL_ARM,
    CandidateEvidenceIntegrator,
    IntegratorSettings,
    candidate_row_weights,
    fit_weighted_normalization,
    formula_fold,
    transform_features,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from evaluate_full23_centered_residual_from_frozen_gnps_ledger import _panel as gnps_panel
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import (
    PANELS,
    WSE,
    PanelData,
    _comparison,
    _metric_summary,
)
from train_full23_centered_residual_from_frozen_ledger import (
    PARENT,
    build_panels,
    load_checkpoint,
)


METHOD = "full23_multisource_additive"
SOURCES = ("enveda", "gnps")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enveda-ledger", type=Path, required=True)
    parser.add_argument("--enveda-chem-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--gnps-score-dir", type=Path, required=True)
    parser.add_argument("--gnps-axis-ledger", type=Path, required=True)
    parser.add_argument("--gnps-chem-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-strength", type=float, default=1e-2)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    parser.set_defaults(center_residual_penalty=False)
    return parser.parse_args()


def combine_panels(
    enveda: PanelData, gnps: PanelData, name: str,
) -> tuple[PanelData, dict[str, tuple[slice, slice]]]:
    if enveda.feature_names != gnps.feature_names:
        raise RuntimeError(f"Full23 feature contract differs across sources: {name}")
    candidate_offset = len(enveda.labels)
    query_offset = len(enveda.query_ptr) - 1
    ptr = np.concatenate((
        np.asarray(enveda.query_ptr, dtype=np.int64),
        candidate_offset + np.asarray(gnps.query_ptr[1:], dtype=np.int64),
    ))
    common_scores = sorted(set(enveda.scores) & set(gnps.scores))
    scores = {
        method: np.concatenate((enveda.scores[method], gnps.scores[method])).astype(np.float32)
        for method in common_scores
    }
    panel = PanelData(
        name=name,
        graph=None,
        scalar=np.concatenate((enveda.scalar, gnps.scalar)).astype(np.float32),
        feature_names=enveda.feature_names,
        labels=np.concatenate((enveda.labels, gnps.labels)).astype(np.int8),
        query_ptr=ptr,
        formulas=np.concatenate((enveda.formulas, gnps.formulas)).astype("U"),
        query_keys=np.concatenate((
            np.asarray([f"enveda::{value}" for value in enveda.query_keys], dtype="U"),
            np.asarray([f"gnps::{value}" for value in gnps.query_keys], dtype="U"),
        )),
        near=np.concatenate((enveda.near, gnps.near)).astype(bool),
        query_weights=np.zeros(query_offset + len(gnps.query_ptr) - 1, dtype=np.float32),
        scores=scores,
    )
    slices = {
        "enveda": (slice(0, query_offset), slice(0, candidate_offset)),
        "gnps": (
            slice(query_offset, len(panel.query_ptr) - 1),
            slice(candidate_offset, len(panel.labels)),
        ),
    }
    return panel, slices


def set_source_balanced_weights(panels: dict[str, PanelData]) -> dict[str, int]:
    counts = Counter(
        str(key) for panel in panels.values() for key in panel.query_keys
    )
    unique = {
        source: len({key for key in counts if key.startswith(source + "::")})
        for source in SOURCES
    }
    for panel in panels.values():
        weights = []
        for key in panel.query_keys:
            key = str(key)
            source = key.split("::", 1)[0]
            weights.append(0.5 / unique[source] / counts[key])
        # Keep these weights in float64: tens of thousands of per-query terms
        # are deliberately tiny, and float32 accumulation can make an exactly
        # balanced 0.5/0.5 construction appear imbalanced by ~3e-5.
        panel.query_weights = np.asarray(weights, dtype=np.float64)
    mass = {
        source: float(sum(
            panel.query_weights[index]
            for panel in panels.values()
            for index, key in enumerate(panel.query_keys)
            if str(key).startswith(source + "::")
        ))
        for source in SOURCES
    }
    if any(not np.isclose(value, 0.5, atol=1e-7, rtol=0.0) for value in mass.values()):
        raise RuntimeError(f"source weights are not balanced: {mass}")
    return unique


def rebalance_selected_source_weights(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
) -> tuple[dict[str, int], dict[str, float]]:
    """Give each source mass 0.5 within the selected training partition.

    Global source balance is insufficient for cross-fitting because a formula
    fold can remove different amounts of Enveda and GNPS mass.  Duplicate query
    occurrences across the two panel views share their query mass.
    """
    keys = [str(panels[name].query_keys[query]) for name, query in selected]
    counts = Counter(keys)
    unique = {
        source: len({key for key in counts if key.startswith(source + "::")})
        for source in SOURCES
    }
    if any(value == 0 for value in unique.values()):
        raise RuntimeError(f"selected partition lacks a source: {unique}")
    mass = {source: 0.0 for source in SOURCES}
    for (name, query), key in zip(selected, keys, strict=True):
        source = key.split("::", 1)[0]
        weight = 0.5 / unique[source] / counts[key]
        panels[name].query_weights[query] = weight
        mass[source] += weight
    if any(not np.isclose(value, 0.5, atol=1e-10, rtol=0.0) for value in mass.values()):
        raise RuntimeError(f"selected source weights are not balanced: {mass}")
    return unique, mass


def occurrences(panels: dict[str, PanelData]) -> list[tuple[str, int]]:
    return [
        (name, query)
        for name, panel in panels.items()
        for query in range(len(panel.query_ptr) - 1)
    ]


def grouped_occurrences(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
) -> list[list[tuple[str, int]]]:
    grouped: dict[str, list[tuple[str, int]]] = {}
    for name, query in selected:
        grouped.setdefault(str(panels[name].query_keys[query]), []).append((name, query))
    return list(grouped.values())


def fit_normalization(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    by_panel = {name: [] for name in panels}
    for name, query in selected:
        by_panel[name].append(query)
    blocks = []
    for name, queries in by_panel.items():
        if not queries:
            continue
        panel = panels[name]
        rows, weights = candidate_row_weights(
            panel.query_ptr, np.asarray(queries, dtype=np.int64), panel.query_weights,
        )
        blocks.append((panel.scalar[rows], weights))
    first = next(iter(panels.values()))
    return fit_weighted_normalization(blocks, first.feature_names)


def pack(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, weights, lengths = [], [], [], []
    for name, query in selected:
        panel = panels[name]
        left, right = map(int, panel.query_ptr[query:query + 2])
        features.append(panel.scalar[left:right])
        labels.append(panel.labels[left:right])
        weights.append(float(panel.query_weights[query]))
        lengths.append(right - left)
    raw = np.concatenate(features) if features else np.empty((0, len(mean)), dtype=np.float32)
    target = np.concatenate(labels) if labels else np.empty(0, dtype=np.int8)
    ptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return (
        transform_features(raw, mean, scale), target, ptr,
        np.asarray(weights, dtype=np.float32),
    )


def fit_model(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace, seed: int,
) -> tuple[CandidateEvidenceIntegrator, list[float]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    names = next(iter(panels.values())).feature_names
    settings = IntegratorSettings(len(names), names.index(WSE), args.hidden)
    model = CandidateEvidenceIntegrator(CONTROL_ARM, settings).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    groups = grouped_occurrences(panels, selected)
    history = []
    for _ in range(args.epochs):
        order = rng.permutation(len(groups))
        total, mass = 0.0, 0.0
        for start in range(0, len(order), args.batch_queries):
            batch = [occ for index in order[start:start + args.batch_queries] for occ in groups[index]]
            features, labels, ptr, weights = pack(panels, batch, mean, scale)
            tensor = torch.as_tensor(features, dtype=torch.float32, device=args.device)
            logits = model(tensor)
            loss = weighted_listwise_loss(logits, ptr, labels, weights)
            loss = loss + args.anchor_strength * weighted_query_residual_penalty(
                model.residual(tensor), ptr, weights, center_within_query=False,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_mass = float(np.sum(weights))
            total += float(loss.detach().cpu()) * batch_mass
            mass += batch_mass
        history.append(total / mass)
    return model, history


@torch.inference_mode()
def predict(
    model: CandidateEvidenceIntegrator, panels: dict[str, PanelData],
    selected: list[tuple[str, int]], mean: np.ndarray, scale: np.ndarray,
    device: str,
) -> dict[tuple[str, int], np.ndarray]:
    output = {}
    for start in range(0, len(selected), 512):
        batch = selected[start:start + 512]
        features, _, ptr, _ = pack(panels, batch, mean, scale)
        values = model(torch.as_tensor(features, dtype=torch.float32, device=device))
        values = values.cpu().numpy().astype(np.float32)
        for index, occurrence in enumerate(batch):
            output[occurrence] = values[int(ptr[index]):int(ptr[index + 1])]
    return output


def source_panel_view(
    original: PanelData, combined_scores: np.ndarray, candidate_slice: slice,
    method_scores: dict[str, np.ndarray],
) -> tuple[PanelData, dict[str, np.ndarray]]:
    output = {METHOD: combined_scores[candidate_slice]}
    for method in (PARENT, WSE, "official_dreams", "noise_v1", "p2b_noise_v1_frozen"):
        if method in original.scores:
            output[method] = original.scores[method]
    return original, output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds != 5 or args.seed != 20261026 or args.epochs != 20
        or args.batch_queries != 256 or args.hidden != 16
        or args.learning_rate != 1e-3 or args.weight_decay != 1e-4
        or args.anchor_strength != 1e-2 or args.bootstrap_resamples < 100
        or args.device != "cpu" or args.center_residual_penalty
    ):
        raise ValueError("multisource Full23 reuses the exact v1 Additive contract")

    parent = load_checkpoint(args.parent_checkpoint)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_report.get("prediction_ledger", {}).get("sha256")
        != sha256_file(args.enveda_ledger)
        or parent_report.get("checkpoint", {}).get("sha256")
        != sha256_file(args.parent_checkpoint)
    ):
        raise RuntimeError("Enveda ledger/checkpoint are not the released Full23 v1 pair")
    enveda = build_panels(args.enveda_ledger, args.enveda_chem_dir, parent)

    gnps = {}
    stored_parent = {}
    expected_names = tuple(map(str, parent["scalar_feature_names"]))
    with np.load(args.gnps_axis_ledger, allow_pickle=False) as axis:
        for name in PANELS:
            gnps[name], stored_parent[name] = gnps_panel(
                name, args.gnps_score_dir, axis, args.gnps_chem_dir, expected_names,
            )
            gnps[name].scores[PARENT] = stored_parent[name]

    combined, slices = {}, {}
    for name in PANELS:
        combined[name], slices[name] = combine_panels(enveda[name], gnps[name], name)
    unique_queries = set_source_balanced_weights(combined)
    all_occurrences = occurrences(combined)
    fold = np.asarray([
        formula_fold(combined[name].formulas[query], args.folds, args.seed)
        for name, query in all_occurrences
    ], dtype=np.int16)
    oof = {name: np.full(len(panel.labels), np.nan, dtype=np.float32) for name, panel in combined.items()}
    fold_reports = []
    for held_out in range(args.folds):
        train = [occ for index, occ in enumerate(all_occurrences) if fold[index] != held_out]
        valid = [occ for index, occ in enumerate(all_occurrences) if fold[index] == held_out]
        train_unique, train_source_mass = rebalance_selected_source_weights(combined, train)
        mean, scale = fit_normalization(combined, train)
        model, history = fit_model(
            combined, train, mean, scale, args, args.seed + 100 * held_out + 1,
        )
        values = predict(model, combined, valid, mean, scale, args.device)
        for (name, query), prediction in values.items():
            left, right = map(int, combined[name].query_ptr[query:query + 2])
            oof[name][left:right] = prediction
        fold_reports.append({
            "fold": held_out, "training_occurrences": len(train),
            "validation_occurrences": len(valid), "initial_loss": history[0],
            "final_loss": history[-1], "training_unique_queries": train_unique,
            "training_source_mass": train_source_mass,
        })
        print(f"[Full23 multisource] completed global formula fold {held_out + 1}/{args.folds}", flush=True)
    if any(np.any(~np.isfinite(values)) for values in oof.values()):
        raise RuntimeError("multisource OOF predictions are incomplete")

    reports = {source: {} for source in SOURCES}
    prediction_artifact = {}
    for panel_index, name in enumerate(PANELS):
        for source_index, source in enumerate(SOURCES):
            original = enveda[name] if source == "enveda" else gnps[name]
            _, candidate_slice = slices[name][source]
            _, methods = source_panel_view(original, oof[name], candidate_slice, {})
            comparisons = {
                f"{METHOD}_vs_{baseline}": _comparison(
                    methods[METHOD], methods[baseline], original,
                    args.bootstrap_resamples,
                    args.seed + 10_000 + panel_index * 100 + source_index * 10 + offset,
                )
                for offset, baseline in enumerate((PARENT, WSE))
            }
            reports[source][name] = {
                "queries": len(original.query_ptr) - 1,
                "candidate_molecules": len(original.labels),
                "methods": {
                    method: _metric_summary(scores, original)[0]
                    for method, scores in methods.items()
                },
                "comparisons": comparisons,
            }
            prefix = f"{source}__{name}__"
            prediction_artifact[prefix + "query_ptr"] = original.query_ptr
            prediction_artifact[prefix + "labels"] = original.labels
            prediction_artifact[prefix + "formula"] = original.formulas
            for method, scores in methods.items():
                prediction_artifact[prefix + method] = scores

    final_unique, final_source_mass = rebalance_selected_source_weights(
        combined, all_occurrences,
    )
    final_mean, final_scale = fit_normalization(combined, all_occurrences)
    final_model, final_history = fit_model(
        combined, all_occurrences, final_mean, final_scale, args, args.seed + 50_001,
    )
    checkpoint = copy.deepcopy(parent)
    checkpoint["arms"][CONTROL_ARM] = {
        key: value.detach().cpu() for key, value in final_model.state_dict().items()
    }
    checkpoint["scalar_mean"] = final_mean
    checkpoint["scalar_scale"] = final_scale
    checkpoint["training"] = {
        **checkpoint.get("training", {}),
        "source": "source-balanced Enveda plus GNPS consumed development",
        "source_mass": {"enveda": 0.5, "gnps": 0.5},
        "formula_fold_across_sources": True,
        "source_indicator": False,
        "hyperparameter_scan": False,
    }

    report = {
        "status": "FULL23_MULTISOURCE_ADDITIVE_DEVELOPMENT_COMPLETE",
        "method": METHOD,
        "dataset_roles": {
            "enveda": "consumed development formula-OOF",
            "gnps": "consumed development formula-OOF",
        },
        "allow_final_external_claim": False,
        "source_balance": {"enveda": 0.5, "gnps": 0.5},
        "unique_queries": unique_queries,
        "formula_fold_across_sources": True,
        "source_indicator": False,
        "features": list(expected_names),
        "optimization": {
            "folds": args.folds, "seed": args.seed, "epochs": args.epochs,
            "batch_queries": args.batch_queries, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "anchor_strength": args.anchor_strength,
            "center_residual_penalty": False, "hyperparameter_scan": False,
        },
        "folds": fold_reports,
        "final_refit": {
            "initial_loss": final_history[0], "final_loss": final_history[-1],
            "parameters": int(sum(value.numel() for value in final_model.parameters())),
            "unique_queries": final_unique, "source_mass": final_source_mass,
        },
        "panels": reports,
        "claim_boundary": (
            "Both sources have been consumed. OOF estimates multisource development only; "
            "the refit checkpoint requires a genuinely new external collection."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        torch.save(checkpoint, staging / "full23_multisource_additive.pt")
        np.savez_compressed(staging / "candidate_predictions_oof.npz", **prediction_artifact)
        report["artifacts"] = {
            "checkpoint": "full23_multisource_additive.pt",
            "predictions": "candidate_predictions_oof.npz",
            "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            "enveda_ledger_sha256": sha256_file(args.enveda_ledger),
            "gnps_axis_ledger_sha256": sha256_file(args.gnps_axis_ledger),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
