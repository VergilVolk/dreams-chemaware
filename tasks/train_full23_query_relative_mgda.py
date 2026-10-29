#!/usr/bin/env python
"""Train a query-relative, source-robust Full23 candidate scorer.

This is deliberately *not* a residual refit of the released Full23 model.  It
reuses the frozen candidate evidence axis, but starts a new linear scorer from
the WSE ordering.  The seven raw expert scores are standardized inside each
complete query candidate block.  Their rank percentiles and all six ChemAware
V2 fields are retained unchanged.  The three query-constant reliability fields
remain only for axis compatibility; a linear listwise scorer cannot use a
within-query constant to change candidate order.

Enveda and GNPS are both consumed development resources.  A global formula
hash defines strict OOF folds across both sources.  At every optimization step
the two source losses remain separate and their gradients are combined by the
closed-form two-task minimum-norm (MGDA) solution.  No source identifier or
source-weight hyperparameter is available to the model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    CandidateEvidenceIntegrator,
    IntegratorSettings,
    formula_fold,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import PANELS, WSE, PanelData, _comparison, _metric_summary
from train_full23_centered_residual_from_frozen_ledger import PARENT, build_panels, load_checkpoint
from train_full23_multisource_additive import (
    SOURCES,
    combine_panels,
    fit_normalization,
    grouped_occurrences,
    occurrences,
    pack,
    predict,
    rebalance_selected_source_weights,
    set_source_balanced_weights,
)
from evaluate_full23_centered_residual_from_frozen_gnps_ledger import _panel as gnps_panel


METHOD = "full23_query_relative_mgda"
SCHEMA = "full23_query_relative_mgda_v1"
EXPERT_FEATURE_PAIRS = (
    ("noise_v1", "noise_v1_percentile"),
    ("weighted_spectral_entropy", "weighted_spectral_entropy_percentile"),
    ("p2b_noise_v1_frozen", "p2b_noise_v1_frozen_percentile"),
    ("noise_msg_fusion_stack_v1", "noise_msg_fusion_stack_v1_percentile"),
    ("chemaware_stage1_cosine", "chemaware_stage1_percentile"),
    ("chemaware_phasea_cosine", "chemaware_phasea_percentile"),
    ("noise_chemaware_fullmsg_cosine", "noise_chemaware_fullmsg_percentile"),
)
RAW_EXPERT_FEATURES = tuple(raw for raw, _ in EXPERT_FEATURE_PAIRS)
RETAINED_PREFIXES = ("reliability_", "chemaware_v2_")


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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-strength", type=float, default=1e-2)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def query_block_zscore(values: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    """Population z-score within each complete candidate block.

    A constant or singleton expert block is an uninformative exact zero.  The
    transform is label-free and never trims a candidate, so it may be applied
    before a formula-OOF split without leaking outcomes across queries.
    """
    values = np.asarray(values, dtype=np.float64)
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if values.ndim != 1 or ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or ptr[-1] != len(values):
        raise ValueError("query-relative score axis is malformed")
    if np.any(np.diff(ptr) <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("query blocks must be nonempty and finite")
    output = np.zeros(len(values), dtype=np.float32)
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        block = values[left:right]
        centered = block - float(np.mean(block))
        scale = float(np.sqrt(np.mean(np.square(centered))))
        if np.isfinite(scale) and scale >= 1e-8:
            output[left:right] = (centered / scale).astype(np.float32)
    return output


def query_relative_panel(panel: PanelData) -> PanelData:
    """Replace only the seven raw expert columns by query-local z-scores."""
    names = tuple(map(str, panel.feature_names))
    expected = set(RAW_EXPERT_FEATURES)
    missing = sorted(expected - set(names))
    if missing:
        raise RuntimeError(f"Full23 feature contract misses raw experts: {missing}")
    allowed = set(RAW_EXPERT_FEATURES)
    allowed.update(percentile for _, percentile in EXPERT_FEATURE_PAIRS)
    allowed.update(name for name in names if name.startswith(RETAINED_PREFIXES))
    unexpected = sorted(set(names) - allowed)
    if unexpected:
        raise RuntimeError(f"query-relative Full23 received unexpected features: {unexpected}")
    if len(names) != 23 or sum(name.startswith("reliability_") for name in names) != 3:
        raise RuntimeError("query-relative Full23 requires the released 23-feature contract")

    values = np.asarray(panel.scalar, dtype=np.float32).copy()
    renamed = list(names)
    for name in RAW_EXPERT_FEATURES:
        index = names.index(name)
        values[:, index] = query_block_zscore(values[:, index], panel.query_ptr)
        renamed[index] = name + "_query_zscore"
    return PanelData(
        name=panel.name,
        graph=panel.graph,
        scalar=values,
        feature_names=tuple(renamed),
        labels=panel.labels,
        query_ptr=panel.query_ptr,
        formulas=panel.formulas,
        query_keys=panel.query_keys,
        near=panel.near,
        query_weights=panel.query_weights.copy(),
        scores=panel.scores,
    )


def formula_oof_partition(
    panels: dict[str, PanelData], all_occurrences: list[tuple[str, int]],
    held_out: int, folds: int, seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return a strict global formula split and prove formula disjointness."""
    train, valid = [], []
    for occurrence in all_occurrences:
        name, query = occurrence
        fold = formula_fold(str(panels[name].formulas[query]), folds, seed)
        (valid if fold == held_out else train).append(occurrence)
    train_formula = {str(panels[name].formulas[query]) for name, query in train}
    valid_formula = {str(panels[name].formulas[query]) for name, query in valid}
    overlap = train_formula & valid_formula
    if overlap:
        raise RuntimeError(f"formula OOF leaked {len(overlap)} formulas")
    if not train or not valid:
        raise RuntimeError("formula OOF produced an empty partition")
    return train, valid


def two_task_mgda_coefficient(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...],
) -> tuple[float, float, float]:
    """Closed-form minimum-norm coefficient for two task gradients.

    Returns ``(alpha, cosine, squared_norm)`` for
    ``alpha * first + (1-alpha) * second``.  There is no tunable task weight.
    """
    if len(first) != len(second) or not first:
        raise ValueError("MGDA gradient tuples differ or are empty")
    dot_11 = dot_22 = dot_12 = 0.0
    for left, right in zip(first, second, strict=True):
        if left.shape != right.shape:
            raise ValueError("MGDA gradient axes differ")
        dot_11 += float(torch.sum(left.detach() * left.detach()).cpu())
        dot_22 += float(torch.sum(right.detach() * right.detach()).cpu())
        dot_12 += float(torch.sum(left.detach() * right.detach()).cpu())
    denominator = dot_11 + dot_22 - 2.0 * dot_12
    if denominator <= 1e-20:
        alpha = 0.5
    else:
        alpha = float(np.clip((dot_22 - dot_12) / denominator, 0.0, 1.0))
    combined = alpha * alpha * dot_11 + (1.0 - alpha) ** 2 * dot_22
    combined += 2.0 * alpha * (1.0 - alpha) * dot_12
    cosine = dot_12 / math.sqrt(dot_11 * dot_22) if dot_11 > 0 and dot_22 > 0 else 0.0
    return alpha, float(cosine), float(max(combined, 0.0))


def _source_groups(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
) -> dict[str, list[list[tuple[str, int]]]]:
    output: dict[str, list[list[tuple[str, int]]]] = {}
    for source in SOURCES:
        subset = [
            occurrence for occurrence in selected
            if str(panels[occurrence[0]].query_keys[occurrence[1]]).startswith(source + "::")
        ]
        output[source] = grouped_occurrences(panels, subset)
        if not output[source]:
            raise RuntimeError(f"MGDA partition lacks source {source}")
    return output


def _sample_groups(
    groups: list[list[tuple[str, int]]], count: int, rng: np.random.Generator,
) -> list[tuple[str, int]]:
    indices = rng.integers(0, len(groups), size=count)
    return [occurrence for index in indices for occurrence in groups[int(index)]]


def fit_mgda_model(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace, seed: int,
) -> tuple[CandidateEvidenceIntegrator, list[dict[str, float]]]:
    """Fit the WSE-initialized linear scorer with two independent objectives."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    names = next(iter(panels.values())).feature_names
    anchor_name = WSE + "_query_zscore"
    settings = IntegratorSettings(len(names), names.index(anchor_name), hidden=1)
    model = CandidateEvidenceIntegrator(CONTROL_ARM, settings).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    groups = _source_groups(panels, selected)
    steps = max(math.ceil(len(values) / args.batch_queries) for values in groups.values())
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        source_loss_sum = {source: 0.0 for source in SOURCES}
        alphas, cosines = [], []
        for _ in range(steps):
            losses = []
            for source in SOURCES:
                batch = _sample_groups(groups[source], args.batch_queries, rng)
                features, labels, ptr, weights = pack(panels, batch, mean, scale)
                tensor = torch.as_tensor(features, dtype=torch.float32, device=args.device)
                logits = model(tensor)
                loss = weighted_listwise_loss(logits, ptr, labels, weights)
                loss = loss + args.anchor_strength * weighted_query_residual_penalty(
                    model.residual(tensor), ptr, weights, center_within_query=True,
                )
                losses.append(loss)
                source_loss_sum[source] += float(loss.detach().cpu())
            gradients = [
                torch.autograd.grad(loss, parameters, retain_graph=index == 0)
                for index, loss in enumerate(losses)
            ]
            alpha, cosine, _ = two_task_mgda_coefficient(gradients[0], gradients[1])
            optimizer.zero_grad(set_to_none=True)
            for parameter, first, second in zip(parameters, gradients[0], gradients[1], strict=True):
                parameter.grad = alpha * first + (1.0 - alpha) * second
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            alphas.append(alpha)
            cosines.append(cosine)
        history.append({
            "epoch": epoch + 1,
            "enveda_loss": source_loss_sum["enveda"] / steps,
            "gnps_loss": source_loss_sum["gnps"] / steps,
            "mgda_alpha_enveda_mean": float(np.mean(alphas)),
            "mgda_alpha_enveda_min": float(np.min(alphas)),
            "mgda_alpha_enveda_max": float(np.max(alphas)),
            "gradient_cosine_mean": float(np.mean(cosines)),
            "steps": steps,
        })
    return model, history


def _source_methods(
    source: str, original: PanelData, combined_prediction: np.ndarray, candidate_slice: slice,
) -> dict[str, np.ndarray]:
    methods = {METHOD: combined_prediction[candidate_slice]}
    for baseline in (PARENT, WSE, "official_dreams", "noise_v1", "p2b_noise_v1_frozen"):
        if baseline in original.scores:
            methods[baseline] = original.scores[baseline]
    return methods


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds != 5 or args.seed != 20261026 or args.epochs != 20
        or args.batch_queries != 256 or args.learning_rate != 1e-3
        or args.weight_decay != 1e-4 or args.anchor_strength != 1e-2
        or args.bootstrap_resamples < 100 or args.device != "cpu"
    ):
        raise ValueError("query-relative MGDA has one fixed, non-scanned training contract")

    parent = load_checkpoint(args.parent_checkpoint)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_report.get("prediction_ledger", {}).get("sha256") != sha256_file(args.enveda_ledger)
        or parent_report.get("checkpoint", {}).get("sha256") != sha256_file(args.parent_checkpoint)
    ):
        raise RuntimeError("Enveda evidence is not the released Full23 pair")
    enveda = {
        name: query_relative_panel(panel)
        for name, panel in build_panels(args.enveda_ledger, args.enveda_chem_dir, parent).items()
    }

    gnps: dict[str, PanelData] = {}
    expected_parent_names = tuple(map(str, parent["scalar_feature_names"]))
    with np.load(args.gnps_axis_ledger, allow_pickle=False) as axis:
        for name in PANELS:
            panel, frozen_parent = gnps_panel(
                name, args.gnps_score_dir, axis, args.gnps_chem_dir, expected_parent_names,
            )
            panel.scores[PARENT] = frozen_parent
            gnps[name] = query_relative_panel(panel)

    combined: dict[str, PanelData] = {}
    slices: dict[str, dict[str, tuple[slice, slice]]] = {}
    for name in PANELS:
        combined[name], slices[name] = combine_panels(enveda[name], gnps[name], name)
    unique_queries = set_source_balanced_weights(combined)
    all_occurrences = occurrences(combined)
    oof = {name: np.full(len(panel.labels), np.nan, dtype=np.float32) for name, panel in combined.items()}
    fold_reports = []
    for held_out in range(args.folds):
        train, valid = formula_oof_partition(
            combined, all_occurrences, held_out, args.folds, args.seed,
        )
        train_unique, train_source_mass = rebalance_selected_source_weights(combined, train)
        mean, scale = fit_normalization(combined, train)
        model, history = fit_mgda_model(
            combined, train, mean, scale, args, args.seed + held_out * 100 + 1,
        )
        values = predict(model, combined, valid, mean, scale, args.device)
        for (name, query), prediction in values.items():
            left, right = map(int, combined[name].query_ptr[query:query + 2])
            oof[name][left:right] = prediction
        fold_reports.append({
            "fold": held_out,
            "training_occurrences": len(train),
            "validation_occurrences": len(valid),
            "training_unique_queries": train_unique,
            "training_source_mass_for_normalization": train_source_mass,
            "first_epoch": history[0],
            "last_epoch": history[-1],
        })
        print(f"[Full23 query-relative MGDA] completed global formula fold {held_out + 1}/{args.folds}", flush=True)
    if any(np.any(~np.isfinite(values)) for values in oof.values()):
        raise RuntimeError("query-relative MGDA OOF predictions are incomplete")

    reports = {source: {} for source in SOURCES}
    prediction_artifact: dict[str, np.ndarray] = {}
    for panel_index, name in enumerate(PANELS):
        for source_index, source in enumerate(SOURCES):
            original = enveda[name] if source == "enveda" else gnps[name]
            _, candidate_slice = slices[name][source]
            methods = _source_methods(source, original, oof[name], candidate_slice)
            reports[source][name] = {
                "queries": len(original.query_ptr) - 1,
                "candidate_molecules": len(original.labels),
                "methods": {
                    method: _metric_summary(scores, original)[0]
                    for method, scores in methods.items()
                },
                "comparisons": {
                    f"{METHOD}_vs_{baseline}": _comparison(
                        methods[METHOD], methods[baseline], original,
                        args.bootstrap_resamples,
                        args.seed + 10_000 + panel_index * 100 + source_index * 10 + offset,
                    )
                    for offset, baseline in enumerate((PARENT, WSE))
                },
            }
            prefix = f"{source}__{name}__"
            prediction_artifact.update({
                prefix + "query_ptr": original.query_ptr,
                prefix + "labels": original.labels,
                prefix + "query_keys": original.query_keys,
                prefix + "formula": original.formulas,
                prefix + "near": original.near,
                **{prefix + method: values for method, values in methods.items()},
            })

    final_unique, final_source_mass = rebalance_selected_source_weights(combined, all_occurrences)
    final_mean, final_scale = fit_normalization(combined, all_occurrences)
    final_model, final_history = fit_mgda_model(
        combined, all_occurrences, final_mean, final_scale, args, args.seed + 50_001,
    )
    checkpoint = {
        "schema": SCHEMA,
        "base_integrator_schema": CHECKPOINT_SCHEMA,
        "arm": CONTROL_ARM,
        "scalar_feature_names": list(next(iter(combined.values())).feature_names),
        "scalar_mean": final_mean,
        "scalar_scale": final_scale,
        "anchor_index": next(iter(combined.values())).feature_names.index(WSE + "_query_zscore"),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "query_relative_raw_experts": list(RAW_EXPERT_FEATURES),
        "parent_residual": False,
        "initialization": "WSE query-relative z-score only",
        "source_indicator": False,
        "source_weight_grid": False,
        "task_gradient_combination": "closed_form_two_task_MGDA",
    }
    report = {
        "status": "FULL23_QUERY_RELATIVE_MGDA_DEVELOPMENT_COMPLETE",
        "method": METHOD,
        "dataset_roles": {
            "enveda": "consumed development strict global formula-OOF",
            "gnps": "consumed development strict global formula-OOF",
        },
        "allow_final_external_claim": False,
        "strict_formula_oof_across_sources": True,
        "features": list(next(iter(combined.values())).feature_names),
        "query_relative_raw_experts": list(RAW_EXPERT_FEATURES),
        "retained_information": "seven percentiles, three reliability fields, six V2 fields",
        "parent_residual": False,
        "parent_checkpoint_usage": "input contract and comparator only; never initializes an OOF model",
        "initialization": "WSE query-relative z-score",
        "task_objectives": list(SOURCES),
        "task_gradient_combination": "closed-form two-task MGDA per paired minibatch",
        "source_indicator": False,
        "source_weight_grid": False,
        "normalization": "training-fold-only, fixed 0.5/0.5 source mass; not an objective weight",
        "unique_queries": unique_queries,
        "optimization": {
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_queries_per_source": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "anchor_strength": args.anchor_strength,
            "anchor_penalty": "query-centered residual MSE",
            "hyperparameter_scan": False,
        },
        "folds": fold_reports,
        "panels": reports,
        "final_refit": {
            "unique_queries": final_unique,
            "normalization_source_mass": final_source_mass,
            "first_epoch": final_history[0],
            "last_epoch": final_history[-1],
            "parameters": int(sum(value.numel() for value in final_model.parameters())),
        },
        "claim_boundary": (
            "Both sources are consumed development. This is a strict formula-OOF "
            "development estimate and a refit candidate for a genuinely new external source."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / "full23_query_relative_mgda.pt"
        torch.save(checkpoint, checkpoint_path)
        prediction_path = staging / "candidate_predictions_oof.npz"
        np.savez_compressed(prediction_path, **prediction_artifact)
        report["artifacts"] = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
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
