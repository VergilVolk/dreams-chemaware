#!/usr/bin/env python
"""Fit a strict outer-safe Full23-parent query-relative MGDA residual.

For each global molecular-formula outer fold, the Full23 v1 Additive parent is
rebuilt using only Enveda queries outside that fold.  The rebuilt parent may
score its own outer-training rows in sample, because it is only a fixed
initialization for the subsequently fitted residual; neither parent nor
residual ever sees an outer-validation formula.  A zero-initialized linear
delta reads the query-relative 23-feature view and is optimized with separate
Enveda and GNPS listwise losses combined by the closed-form two-task MGDA
solution.

This is a *sequential outer-fit* estimate.  It is not nested stacking: there is
no inner cross-fit of the parent inside an outer-training partition.  The final
all-development refit uses the released Full23 v1 checkpoint as its fixed
parent and fits only the delta on both consumed development sources.  No source
indicator or source-weight grid is exposed to the scorer.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from torch import nn

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    formula_fold,
    predict_from_payload,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import (
    PANELS,
    WSE,
    PanelData,
    _comparison,
    _fit_arm,
    _fit_normalization,
    _metric_summary,
    _occurrences,
    _predict_occurrences,
)
from train_full23_centered_residual_from_frozen_ledger import (
    PARENT,
    build_panels,
    load_checkpoint,
)
from train_full23_multisource_additive import (
    SOURCES,
    combine_panels,
    fit_normalization,
    occurrences,
    pack,
    rebalance_selected_source_weights,
    set_source_balanced_weights,
)
from train_full23_query_relative_mgda import (
    _sample_groups,
    _source_groups,
    formula_oof_partition,
    query_relative_panel,
    two_task_mgda_coefficient,
)
from evaluate_full23_centered_residual_from_frozen_gnps_ledger import (
    _panel as gnps_panel,
)


METHOD = "full23_parent_query_relative_mgda_residual"
SEQUENTIAL_PARENT = "full23_v1_sequential_outer_parent"
SCHEMA = "full23_parent_query_relative_mgda_residual_v1"


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
    # Required by the exact historical Full23 v1 Additive helper.
    parser.set_defaults(center_residual_penalty=False)
    return parser.parse_args()


class ParentPlusLinearDelta(nn.Module):
    """A zero-initialized candidate delta added to immutable parent logits."""

    def __init__(self, features: int):
        super().__init__()
        if features < 1:
            raise ValueError("delta requires at least one feature")
        self.network = nn.Linear(features, 1)
        nn.init.zeros_(self.network.weight)
        nn.init.zeros_(self.network.bias)

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.network.in_features:
            raise ValueError("delta feature axis mismatch")
        return self.network(features).squeeze(-1)

    def forward(self, features: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        if parent.ndim != 1 or parent.shape[0] != features.shape[0]:
            raise ValueError("fixed parent score axis mismatch")
        return parent + self.delta(features)


def outer_parent_training_occurrences(
    enveda: dict[str, PanelData], held_out: int, folds: int, seed: int,
) -> list[tuple[str, int]]:
    """Return Enveda-only parent rows outside the global held-out formula fold."""
    selected = [
        (name, query)
        for name, panel in enveda.items()
        for query in range(len(panel.query_ptr) - 1)
        if formula_fold(str(panel.formulas[query]), folds, seed) != held_out
    ]
    if not selected:
        raise RuntimeError("outer parent partition is empty")
    if any(
        formula_fold(str(enveda[name].formulas[query]), folds, seed) == held_out
        for name, query in selected
    ):
        raise RuntimeError("outer parent saw a held-out formula")
    return selected


def _parent_scores_from_model(
    model: nn.Module,
    raw_by_source: dict[str, dict[str, PanelData]],
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """Score both sources and concatenate them in ``combine_panels`` order."""
    by_source: dict[str, dict[str, np.ndarray]] = {}
    for source in SOURCES:
        panels = raw_by_source[source]
        predicted = _predict_occurrences(
            model, panels, _occurrences(panels), mean, scale, device,
        )
        by_source[source] = {
            name: np.concatenate([
                predicted[(name, query)]
                for query in range(len(panels[name].query_ptr) - 1)
            ]).astype(np.float32)
            for name in PANELS
        }
    return {
        name: np.concatenate((by_source["enveda"][name], by_source["gnps"][name]))
        .astype(np.float32)
        for name in PANELS
    }


def _released_parent_scores(
    parent_payload: dict,
    raw_by_source: dict[str, dict[str, PanelData]],
    device: str,
) -> dict[str, np.ndarray]:
    by_source: dict[str, dict[str, np.ndarray]] = {}
    for source in SOURCES:
        by_source[source] = {}
        for name in PANELS:
            panel = raw_by_source[source][name]
            by_source[source][name] = predict_from_payload(
                parent_payload, panel.feature_names, panel.scalar,
                arm=CONTROL_ARM, device=device,
            )
    return {
        name: np.concatenate((by_source["enveda"][name], by_source["gnps"][name]))
        .astype(np.float32)
        for name in PANELS
    }


def pack_parent(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray, parent_scores: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, ptr, weights = pack(panels, selected, mean, scale)
    parent_blocks = []
    for name, query in selected:
        left, right = map(int, panels[name].query_ptr[query:query + 2])
        parent_blocks.append(parent_scores[name][left:right])
    fixed = np.concatenate(parent_blocks).astype(np.float32) if parent_blocks else np.empty(0, np.float32)
    if len(fixed) != len(labels) or np.any(~np.isfinite(fixed)):
        raise RuntimeError("fixed parent packing drifted")
    return features, labels, ptr, weights, fixed


def fit_delta_mgda(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    parent_scores: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[ParentPlusLinearDelta, list[dict[str, float]]]:
    """Fit only the query-relative delta with two source-specific objectives."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ParentPlusLinearDelta(len(next(iter(panels.values())).feature_names)).to(args.device)
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
                features, labels, ptr, weights, fixed = pack_parent(
                    panels, batch, mean, scale, parent_scores,
                )
                tensor = torch.as_tensor(features, dtype=torch.float32, device=args.device)
                fixed_tensor = torch.as_tensor(fixed, dtype=torch.float32, device=args.device)
                logits = model(tensor, fixed_tensor)
                delta = model.delta(tensor)
                loss = weighted_listwise_loss(logits, ptr, labels, weights)
                loss = loss + args.anchor_strength * weighted_query_residual_penalty(
                    delta, ptr, weights, center_within_query=True,
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


DeltaPayload = tuple[
    torch.Tensor, np.ndarray, np.ndarray, np.ndarray, torch.Tensor,
]


def _delta_payload(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray, parent_scores: dict[str, np.ndarray],
    device: str,
) -> DeltaPayload:
    features, labels, ptr, weights, fixed = pack_parent(
        panels, selected, mean, scale, parent_scores,
    )
    return (
        torch.as_tensor(features, dtype=torch.float32, device=device),
        labels,
        ptr,
        weights,
        torch.as_tensor(fixed, dtype=torch.float32, device=device),
    )


def _delta_objective_components(
    model: ParentPlusLinearDelta,
    payload: DeltaPayload,
    parameters: tuple[torch.nn.Parameter, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, listwise, anchor and shared-L2 objectives.

    The L2 term is placed inside both task objectives.  A plain common-gradient
    step therefore optimizes exactly the regularized objectives used by MGDA;
    no optimizer-side weight decay is allowed to rotate that direction.
    """
    features, labels, ptr, weights, fixed = payload
    logits = model(features, fixed)
    delta = model.delta(features)
    listwise = weighted_listwise_loss(logits, ptr, labels, weights)
    anchor = weighted_query_residual_penalty(
        delta, ptr, weights, center_within_query=True,
    )
    l2 = 0.5 * args.weight_decay * sum(
        torch.sum(torch.square(parameter)) for parameter in parameters
    )
    total = listwise + args.anchor_strength * anchor + l2
    return total, listwise, anchor, l2


def _pareto_backtracking_step(
    parameters: tuple[torch.nn.Parameter, ...],
    direction: tuple[torch.Tensor, ...],
    source_gradients: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    base_losses: tuple[float, float],
    loss_closure: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    learning_rate: float,
    *,
    maximum_norm: float = 5.0,
    maximum_backtracks: int = 12,
    armijo: float = 1e-4,
) -> dict[str, float | int | bool]:
    """Take a common-descent step without Adam/momentum direction rotation."""
    if len(parameters) != len(direction) or any(
        len(gradients) != len(parameters) for gradients in source_gradients
    ):
        raise ValueError("Pareto step gradient axes differ")
    norm_squared = sum(float(torch.sum(value.detach() ** 2).cpu()) for value in direction)
    if not np.isfinite(norm_squared) or norm_squared < 0:
        raise RuntimeError("non-finite Pareto direction norm")
    norm = math.sqrt(norm_squared)
    scale = min(1.0, maximum_norm / norm) if norm > 0 else 1.0
    scaled = tuple(value.detach() * scale for value in direction)
    directional = tuple(
        sum(
            float(torch.sum(gradient.detach() * value).cpu())
            for gradient, value in zip(gradients, scaled, strict=True)
        )
        for gradients in source_gradients
    )
    tolerance = 1e-8 * max(
        1.0,
        *(math.sqrt(sum(float(torch.sum(g.detach() ** 2).cpu()) for g in gradients))
          * max(norm * scale, 1e-12) for gradients in source_gradients),
    )
    if any(value < -tolerance for value in directional):
        raise RuntimeError(
            f"MGDA direction is not common descent: derivatives={directional}"
        )
    if norm <= 1e-20:
        return {
            "accepted": True,
            "stationary": True,
            "backtracks": 0,
            "step_size": 0.0,
            "direction_norm": 0.0,
            "enveda_directional_derivative": directional[0],
            "gnps_directional_derivative": directional[1],
        }

    originals = tuple(parameter.detach().clone() for parameter in parameters)
    for backtracks in range(maximum_backtracks + 1):
        step_size = learning_rate * (0.5 ** backtracks)
        with torch.no_grad():
            for parameter, original, value in zip(
                parameters, originals, scaled, strict=True,
            ):
                parameter.copy_(original - step_size * value)
            trials = tuple(float(loss.detach().cpu()) for loss in loss_closure())
        accepted = all(
            trial <= base - armijo * step_size * derivative
            + 1e-8 * max(1.0, abs(base))
            for trial, base, derivative in zip(
                trials, base_losses, directional, strict=True,
            )
        )
        if accepted:
            return {
                "accepted": True,
                "stationary": False,
                "backtracks": backtracks,
                "step_size": step_size,
                "direction_norm": norm * scale,
                "enveda_directional_derivative": directional[0],
                "gnps_directional_derivative": directional[1],
                "enveda_trial_loss": trials[0],
                "gnps_trial_loss": trials[1],
            }
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)
    return {
        "accepted": False,
        "stationary": False,
        "backtracks": maximum_backtracks + 1,
        "step_size": 0.0,
        "direction_norm": norm * scale,
        "enveda_directional_derivative": directional[0],
        "gnps_directional_derivative": directional[1],
    }


def _fixed_monitor_occurrences(
    groups: list[list[tuple[str, int]]], maximum_groups: int = 256,
) -> list[tuple[str, int]]:
    """Choose a deterministic, source-specific monitor ledger once per fit."""
    if not groups:
        raise ValueError("monitor ledger requires at least one group")
    count = min(maximum_groups, len(groups))
    indices = np.linspace(0, len(groups) - 1, num=count, dtype=np.int64)
    return [occurrence for index in indices for occurrence in groups[int(index)]]


def fit_delta_pareto_sgd(
    panels: dict[str, PanelData], selected: list[tuple[str, int]],
    parent_scores: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[ParentPlusLinearDelta, list[dict[str, float | int]]]:
    """Fit the delta with an actual Pareto-safe common-gradient update.

    Unlike :func:`fit_delta_mgda`, this implementation never passes the MGDA
    direction through Adam, momentum, coordinate preconditioning or decoupled
    weight decay.  A fixed-step SGD proposal is accepted only when both task
    objectives decrease on the exact two source batches used for the gradient;
    otherwise the step is halved, then rejected after a fixed bound.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ParentPlusLinearDelta(
        len(next(iter(panels.values())).feature_names),
    ).to(args.device)
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    groups = _source_groups(panels, selected)
    steps = max(math.ceil(len(values) / args.batch_queries) for values in groups.values())
    rng = np.random.default_rng(seed)
    monitors = {
        source: _delta_payload(
            panels, _fixed_monitor_occurrences(groups[source]), mean, scale,
            parent_scores, args.device,
        )
        for source in SOURCES
    }
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        source_total = {source: 0.0 for source in SOURCES}
        source_listwise = {source: 0.0 for source in SOURCES}
        source_anchor = {source: 0.0 for source in SOURCES}
        alphas: list[float] = []
        cosines: list[float] = []
        backtracks: list[int] = []
        direction_norms: list[float] = []
        enveda_derivatives: list[float] = []
        gnps_derivatives: list[float] = []
        accepted_steps = stationary_steps = 0
        for _ in range(steps):
            payloads = {
                source: _delta_payload(
                    panels,
                    _sample_groups(groups[source], args.batch_queries, rng),
                    mean, scale, parent_scores, args.device,
                )
                for source in SOURCES
            }
            components = {
                source: _delta_objective_components(
                    model, payloads[source], parameters, args,
                )
                for source in SOURCES
            }
            losses = tuple(components[source][0] for source in SOURCES)
            gradients = tuple(
                torch.autograd.grad(loss, parameters, retain_graph=index == 0)
                for index, loss in enumerate(losses)
            )
            alpha, cosine, _ = two_task_mgda_coefficient(
                gradients[0], gradients[1],
            )
            direction = tuple(
                alpha * first + (1.0 - alpha) * second
                for first, second in zip(gradients[0], gradients[1], strict=True)
            )

            def current_losses() -> tuple[torch.Tensor, torch.Tensor]:
                return tuple(
                    _delta_objective_components(
                        model, payloads[source], parameters, args,
                    )[0]
                    for source in SOURCES
                )  # type: ignore[return-value]

            result = _pareto_backtracking_step(
                parameters,
                direction,
                (gradients[0], gradients[1]),
                tuple(float(loss.detach().cpu()) for loss in losses),
                current_losses,
                args.learning_rate,
            )
            accepted_steps += int(bool(result["accepted"]))
            stationary_steps += int(bool(result["stationary"]))
            backtracks.append(int(result["backtracks"]))
            direction_norms.append(float(result["direction_norm"]))
            enveda_derivatives.append(float(result["enveda_directional_derivative"]))
            gnps_derivatives.append(float(result["gnps_directional_derivative"]))
            alphas.append(alpha)
            cosines.append(cosine)
            for source in SOURCES:
                source_total[source] += float(components[source][0].detach().cpu())
                source_listwise[source] += float(components[source][1].detach().cpu())
                source_anchor[source] += float(components[source][2].detach().cpu())

        monitor_values: dict[str, float] = {}
        with torch.no_grad():
            for source in SOURCES:
                total, listwise, anchor, l2 = _delta_objective_components(
                    model, monitors[source], parameters, args,
                )
                monitor_values.update({
                    f"fixed_{source}_total": float(total.cpu()),
                    f"fixed_{source}_listwise": float(listwise.cpu()),
                    f"fixed_{source}_anchor": float(anchor.cpu()),
                    f"fixed_{source}_shared_l2": float(l2.cpu()),
                })
        history.append({
            "epoch": epoch + 1,
            "enveda_loss": source_total["enveda"] / steps,
            "gnps_loss": source_total["gnps"] / steps,
            "enveda_listwise": source_listwise["enveda"] / steps,
            "gnps_listwise": source_listwise["gnps"] / steps,
            "enveda_anchor": source_anchor["enveda"] / steps,
            "gnps_anchor": source_anchor["gnps"] / steps,
            "mgda_alpha_enveda_mean": float(np.mean(alphas)),
            "mgda_alpha_enveda_min": float(np.min(alphas)),
            "mgda_alpha_enveda_max": float(np.max(alphas)),
            "gradient_cosine_mean": float(np.mean(cosines)),
            "accepted_steps": accepted_steps,
            "stationary_steps": stationary_steps,
            "rejected_steps": steps - accepted_steps,
            "mean_backtracks": float(np.mean(backtracks)),
            "maximum_backtracks": int(np.max(backtracks)),
            "mean_direction_norm": float(np.mean(direction_norms)),
            "minimum_enveda_directional_derivative": float(np.min(enveda_derivatives)),
            "minimum_gnps_directional_derivative": float(np.min(gnps_derivatives)),
            "steps": steps,
            **monitor_values,
        })
    return model, history


@torch.inference_mode()
def predict_delta(
    model: ParentPlusLinearDelta, panels: dict[str, PanelData],
    selected: list[tuple[str, int]], parent_scores: dict[str, np.ndarray],
    mean: np.ndarray, scale: np.ndarray, device: str,
) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    for start in range(0, len(selected), 512):
        batch = selected[start:start + 512]
        features, _, ptr, _, fixed = pack_parent(
            panels, batch, mean, scale, parent_scores,
        )
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        values = model(
            tensor, torch.as_tensor(fixed, dtype=torch.float32, device=device),
        ).cpu().numpy().astype(np.float32)
        for index, occurrence in enumerate(batch):
            output[occurrence] = values[int(ptr[index]):int(ptr[index + 1])]
    return output


def _source_methods(
    original: PanelData, combined_scores: np.ndarray, combined_parent: np.ndarray,
    candidate_slice: slice,
) -> dict[str, np.ndarray]:
    methods = {
        METHOD: combined_scores[candidate_slice],
        SEQUENTIAL_PARENT: combined_parent[candidate_slice],
    }
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
        or args.batch_queries != 256 or args.hidden != 16
        or args.learning_rate != 1e-3 or args.weight_decay != 1e-4
        or args.anchor_strength != 1e-2 or args.bootstrap_resamples < 100
        or args.device != "cpu" or args.center_residual_penalty
    ):
        raise ValueError("parent plus query-relative MGDA uses one fixed contract")

    parent_payload = load_checkpoint(args.parent_checkpoint)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_report.get("prediction_ledger", {}).get("sha256")
        != sha256_file(args.enveda_ledger)
        or parent_report.get("checkpoint", {}).get("sha256")
        != sha256_file(args.parent_checkpoint)
    ):
        raise RuntimeError("Enveda evidence is not the released Full23 v1 pair")

    enveda_raw = build_panels(args.enveda_ledger, args.enveda_chem_dir, parent_payload)
    gnps_raw: dict[str, PanelData] = {}
    expected_parent_names = tuple(map(str, parent_payload["scalar_feature_names"]))
    with np.load(args.gnps_axis_ledger, allow_pickle=False) as axis:
        for name in PANELS:
            panel, frozen_parent = gnps_panel(
                name, args.gnps_score_dir, axis, args.gnps_chem_dir,
                expected_parent_names,
            )
            panel.scores[PARENT] = frozen_parent
            gnps_raw[name] = panel
    raw_by_source = {"enveda": enveda_raw, "gnps": gnps_raw}
    enveda_qrel = {name: query_relative_panel(panel) for name, panel in enveda_raw.items()}
    gnps_qrel = {name: query_relative_panel(panel) for name, panel in gnps_raw.items()}

    combined: dict[str, PanelData] = {}
    slices: dict[str, dict[str, tuple[slice, slice]]] = {}
    for name in PANELS:
        combined[name], slices[name] = combine_panels(enveda_qrel[name], gnps_qrel[name], name)
    unique_queries = set_source_balanced_weights(combined)
    all_occurrences = occurrences(combined)
    oof = {name: np.full(len(panel.labels), np.nan, np.float32) for name, panel in combined.items()}
    parent_oof = {
        name: np.full(len(panel.labels), np.nan, np.float32) for name, panel in combined.items()
    }
    fold_reports = []
    for held_out in range(args.folds):
        train, valid = formula_oof_partition(
            combined, all_occurrences, held_out, args.folds, args.seed,
        )
        parent_train = outer_parent_training_occurrences(
            enveda_raw, held_out, args.folds, args.seed,
        )
        parent_train_formulas = {
            str(enveda_raw[name].formulas[query]).strip() for name, query in parent_train
        }
        valid_formulas = {
            str(combined[name].formulas[query]).strip() for name, query in valid
        }
        if parent_train_formulas & valid_formulas:
            raise RuntimeError("outer validation formula leaked into parent fitting")

        # Exact Full23 v1 Additive contract.  CONTROL_ARM is arm index one in
        # the released trainer, hence the historical seed + 100*fold + 1.
        parent_mean, parent_scale = _fit_normalization(enveda_raw, parent_train)
        parent_model, parent_history = _fit_arm(
            CONTROL_ARM, enveda_raw, parent_train, parent_mean, parent_scale,
            args, args.seed + 100 * held_out + 1,
        )
        fold_parent_scores = _parent_scores_from_model(
            parent_model, raw_by_source, parent_mean, parent_scale, args.device,
        )

        train_unique, train_source_mass = rebalance_selected_source_weights(combined, train)
        delta_mean, delta_scale = fit_normalization(combined, train)
        delta_model, delta_history = fit_delta_mgda(
            combined, train, fold_parent_scores, delta_mean, delta_scale,
            args, args.seed + 100 * held_out + 51,
        )
        predictions = predict_delta(
            delta_model, combined, valid, fold_parent_scores,
            delta_mean, delta_scale, args.device,
        )
        for (name, query), values in predictions.items():
            left, right = map(int, combined[name].query_ptr[query:query + 2])
            if np.any(np.isfinite(oof[name][left:right])):
                raise RuntimeError("outer validation row was predicted more than once")
            oof[name][left:right] = values
            parent_oof[name][left:right] = fold_parent_scores[name][left:right]
        fold_reports.append({
            "fold": held_out,
            "training_occurrences": len(train),
            "validation_occurrences": len(valid),
            "training_unique_queries": train_unique,
            "training_source_mass_for_delta_normalization": train_source_mass,
            "parent_training_source": "enveda_only",
            "parent_training_occurrences": len(parent_train),
            "parent_training_formulas": len(parent_train_formulas),
            "validation_formulas": len(valid_formulas),
            "parent_seed": args.seed + 100 * held_out + 1,
            "delta_seed": args.seed + 100 * held_out + 51,
            "parent_initial_loss": parent_history[0],
            "parent_final_loss": parent_history[-1],
            "delta_first_epoch": delta_history[0],
            "delta_last_epoch": delta_history[-1],
            "parent_validation_formula_overlap": 0,
        })
        print(
            f"[Full23 parent + query-relative MGDA] completed outer formula fold "
            f"{held_out + 1}/{args.folds}", flush=True,
        )
    if any(np.any(~np.isfinite(values)) for values in oof.values()):
        raise RuntimeError("outer OOF residual predictions are incomplete")
    if any(np.any(~np.isfinite(values)) for values in parent_oof.values()):
        raise RuntimeError("outer OOF parent predictions are incomplete")

    reports = {source: {} for source in SOURCES}
    prediction_artifact: dict[str, np.ndarray] = {}
    for panel_index, name in enumerate(PANELS):
        for source_index, source in enumerate(SOURCES):
            original = enveda_raw[name] if source == "enveda" else gnps_raw[name]
            _, candidate_slice = slices[name][source]
            methods = _source_methods(
                original, oof[name], parent_oof[name], candidate_slice,
            )
            comparisons = {
                f"{METHOD}_vs_{baseline}": _comparison(
                    methods[METHOD], methods[baseline], original,
                    args.bootstrap_resamples,
                    args.seed + 10_000 + panel_index * 100 + source_index * 10 + offset,
                )
                for offset, baseline in enumerate((SEQUENTIAL_PARENT, PARENT, WSE))
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
            prediction_artifact.update({
                prefix + "query_ptr": original.query_ptr,
                prefix + "labels": original.labels,
                prefix + "query_keys": original.query_keys,
                prefix + "formula": original.formulas,
                prefix + "near": original.near,
                **{prefix + method: values for method, values in methods.items()},
            })

    # Final development refit: the released all-Enveda Full23 v1 checkpoint is
    # the immutable parent; only the query-relative delta sees both sources.
    final_parent_scores = _released_parent_scores(
        parent_payload, raw_by_source, args.device,
    )
    final_unique, final_source_mass = rebalance_selected_source_weights(
        combined, all_occurrences,
    )
    final_mean, final_scale = fit_normalization(combined, all_occurrences)
    final_delta, final_history = fit_delta_mgda(
        combined, all_occurrences, final_parent_scores, final_mean, final_scale,
        args, args.seed + 50_051,
    )
    checkpoint = {
        "schema": SCHEMA,
        "parent_integrator_schema": CHECKPOINT_SCHEMA,
        "parent_arm": CONTROL_ARM,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "parent_payload": copy.deepcopy(parent_payload),
        "delta_feature_names": list(next(iter(combined.values())).feature_names),
        "delta_mean": final_mean,
        "delta_scale": final_scale,
        "delta_state_dict": {
            key: value.detach().cpu() for key, value in final_delta.state_dict().items()
        },
        "delta_semantics": "parent_score + zero_initialized_linear_query_relative_delta",
        "training": {
            "sources": list(SOURCES),
            "source_indicator": False,
            "source_weight_grid": False,
            "task_gradient_combination": "closed_form_two_task_MGDA",
            "query_centered_delta_penalty": True,
            "final_parent": "released Full23 v1 all-Enveda refit checkpoint",
        },
    }
    report = {
        "status": "FULL23_PARENT_QUERY_RELATIVE_MGDA_RESIDUAL_DEVELOPMENT_COMPLETE",
        "method": METHOD,
        "dataset_roles": {
            "enveda": "consumed development strict global formula outer-fit",
            "gnps": "consumed development strict global formula outer-fit",
        },
        "allow_final_external_claim": False,
        "evaluation_design": "sequential outer-fit; not nested stacking",
        "strict_global_formula_outer_folds": True,
        "parent_outer_fit": (
            "each outer parent is rebuilt only on Enveda formulas outside that outer fold; "
            "outer-training predictions may be in sample; outer-validation formulas are unseen"
        ),
        "delta_outer_fit": (
            "zero-initialized linear query-relative delta trained on both outer-training "
            "sources with separate losses and closed-form two-task MGDA"
        ),
        "source_indicator": False,
        "source_weight_grid": False,
        "features": list(next(iter(combined.values())).feature_names),
        "unique_queries": unique_queries,
        "optimization": {
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_queries_per_source": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "delta_anchor_strength": args.anchor_strength,
            "delta_anchor_penalty": "query-centered delta MSE",
            "hyperparameter_scan": False,
        },
        "folds": fold_reports,
        "panels": reports,
        "final_refit": {
            "parent": "released Full23 v1 checkpoint",
            "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            "unique_queries": final_unique,
            "normalization_source_mass": final_source_mass,
            "delta_first_epoch": final_history[0],
            "delta_last_epoch": final_history[-1],
            "delta_parameters": int(sum(value.numel() for value in final_delta.parameters())),
        },
        "claim_boundary": (
            "Both sources are consumed development. Outer-fold estimates are sequential "
            "outer-fit, not nested stacking; the final refit requires a genuinely new source."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / "full23_parent_query_relative_mgda_residual.pt"
        torch.save(checkpoint, checkpoint_path)
        prediction_path = staging / "candidate_predictions_outer_oof.npz"
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
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
