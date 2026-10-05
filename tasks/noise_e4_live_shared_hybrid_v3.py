"""Scientifically bounded E4 plus later-action direct fine-tuning primitives.

V3 restores the complete live shared-ranking relation for later actions:
clean query, action query, positive references and negative references are all
encoded by, and differentiated through, the same shared encoder.  The module
also provides a query-equal scheduler and an optimizer bridge whose E4 and
action AdamW moments are genuinely independent.

This file deliberately does not select actions from held outcomes and does not
define an embedding or margin teacher.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from noise_corrected_update_arbitration_v10 import (
    compose_safe_exact_corrective_updates_by_group,
)


@dataclass(frozen=True)
class QueryEqualActionPlanV3:
    """One action per query opportunity; batches never repeat a query."""

    epoch_steps: tuple[tuple[tuple[int, ...], ...], ...]
    action_count: int
    query_count: int
    views_per_query_per_epoch: int
    optimizer_steps_per_epoch: int
    active_steps_per_epoch: int
    physical_action_exposures: int
    unique_actions_exposed: int
    maximum_actions_per_query: int

    def audit_manifest(self) -> dict[str, object]:
        return {
            "scheduler": "query_equal_one_action_per_query_opportunity_v3",
            "action_count": int(self.action_count),
            "query_count": int(self.query_count),
            "views_per_query_per_epoch": int(self.views_per_query_per_epoch),
            "optimizer_steps_per_epoch": int(self.optimizer_steps_per_epoch),
            "active_steps_per_epoch": int(self.active_steps_per_epoch),
            "zero_action_steps_per_epoch": int(
                self.optimizer_steps_per_epoch - self.active_steps_per_epoch
            ),
            "physical_action_exposures": int(self.physical_action_exposures),
            "unique_actions_exposed": int(self.unique_actions_exposed),
            "all_unique_actions_exposed": bool(
                self.unique_actions_exposed == self.action_count
            ),
            "unique_action_coverage_fraction": float(
                self.unique_actions_exposed / self.action_count
            ),
            "maximum_actions_per_query": int(self.maximum_actions_per_query),
            "same_query_actions_never_share_an_optimizer_step": True,
            "query_dose_equal_within_every_epoch": True,
            "action_rows_are_coverage_first_within_query": True,
            "source_families_are_interleaved_not_averaged": True,
        }


def _family_interleaved_order(
    indices: Sequence[int],
    source_families: Sequence[str],
    rng: np.random.Generator,
) -> list[int]:
    by_family: dict[str, list[int]] = {}
    for index in indices:
        by_family.setdefault(str(source_families[int(index)]), []).append(int(index))
    families = sorted(by_family)
    rng.shuffle(families)
    for family in families:
        rng.shuffle(by_family[family])
    output: list[int] = []
    while any(by_family.values()):
        for family in families:
            if by_family[family]:
                output.append(by_family[family].pop())
    return output


def build_query_equal_action_plan_v3(
    action_ids: Sequence[str],
    query_indices: Sequence[int],
    source_families: Sequence[str],
    *,
    epochs: int,
    views_per_query_per_epoch: int,
    actions_per_step: int,
    optimizer_steps_per_epoch: int,
    seed: int,
) -> QueryEqualActionPlanV3:
    """Build a query-equal schedule without averaging same-query actions.

    Every query receives the same number of action opportunities in every
    epoch.  Each opportunity contains exactly one action for that query.  The
    optimizer may batch actions from different queries, exactly like a normal
    contrastive minibatch, but two actions for the same query never share a
    step.  Rows cycle coverage-first within each query; source/family labels
    only determine the interleaved order and never determine loss weights.
    """
    if not (
        len(action_ids) == len(query_indices) == len(source_families)
        and len(action_ids) > 0
    ):
        raise ValueError("action schedule ledgers must be non-empty and aligned")
    if any(not str(value) for value in action_ids):
        raise ValueError("action identifiers must be non-empty")
    if len(set(map(str, action_ids))) != len(action_ids):
        raise ValueError("action identifiers must be unique")
    if epochs < 1 or views_per_query_per_epoch < 1 or actions_per_step < 1:
        raise ValueError("schedule dimensions must be positive")

    by_query: dict[int, list[int]] = {}
    for index, query in enumerate(query_indices):
        by_query.setdefault(int(query), []).append(index)
    queries = sorted(by_query)
    batches_per_round = math.ceil(len(queries) / actions_per_step)
    active_steps = views_per_query_per_epoch * batches_per_round
    if active_steps > optimizer_steps_per_epoch:
        raise ValueError(
            "query-equal action schedule exceeds the historical E4 step budget"
        )

    rng = np.random.default_rng(seed)
    action_order = {
        query: _family_interleaved_order(by_query[query], source_families, rng)
        for query in queries
    }
    cursors = {query: 0 for query in queries}
    exposed: set[int] = set()
    epoch_steps: list[tuple[tuple[int, ...], ...]] = []
    for _epoch in range(epochs):
        active_batches: list[tuple[int, ...]] = []
        for _view in range(views_per_query_per_epoch):
            round_queries = list(queries)
            rng.shuffle(round_queries)
            for left in range(0, len(round_queries), actions_per_step):
                batch_queries = round_queries[left:left + actions_per_step]
                if len(set(batch_queries)) != len(batch_queries):
                    raise RuntimeError("one action step repeats a query")
                batch: list[int] = []
                for query in batch_queries:
                    order = action_order[query]
                    position = cursors[query]
                    action_index = order[position % len(order)]
                    cursors[query] = position + 1
                    batch.append(action_index)
                    exposed.add(action_index)
                active_batches.append(tuple(batch))

        # Spread action-bearing batches across the unchanged E4 step count.
        positions = [
            min(
                optimizer_steps_per_epoch - 1,
                int(math.floor((index + 0.5) * optimizer_steps_per_epoch / active_steps)),
            )
            for index in range(active_steps)
        ]
        if len(set(positions)) != len(positions):
            raise RuntimeError("action schedule step placement collided")
        steps: list[tuple[int, ...]] = [tuple() for _ in range(optimizer_steps_per_epoch)]
        for position, batch in zip(positions, active_batches):
            steps[position] = batch
        epoch_steps.append(tuple(steps))

    plan = QueryEqualActionPlanV3(
        epoch_steps=tuple(epoch_steps),
        action_count=len(action_ids),
        query_count=len(queries),
        views_per_query_per_epoch=views_per_query_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        active_steps_per_epoch=active_steps,
        physical_action_exposures=(
            epochs * views_per_query_per_epoch * len(queries)
        ),
        unique_actions_exposed=len(exposed),
        maximum_actions_per_query=max(map(len, by_query.values())),
    )

    for epoch in plan.epoch_steps:
        counts = {query: 0 for query in queries}
        for batch in epoch:
            observed_queries = [int(query_indices[index]) for index in batch]
            if len(observed_queries) != len(set(observed_queries)):
                raise RuntimeError("same-query actions were mixed before optimization")
            for query in observed_queries:
                counts[query] += 1
        if set(counts.values()) != {views_per_query_per_epoch}:
            raise RuntimeError("query action dose is not equal within an epoch")
    return plan


def live_shared_e4_action_objective_v3(
    encoded: torch.Tensor,
    layouts: Sequence[Mapping[str, object]],
    anchor_embeddings: Sequence[torch.Tensor],
    official_margins: Sequence[float],
    *,
    rank_margin: float,
    temperature: float,
    margin_floor_slack: float,
    lambda_clean_rank: float,
    lambda_aug_rank: float,
    lambda_consistency: float,
    lambda_margin_floor: float,
    lambda_preserve: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Complete query-equal E4 objective with all four roles live.

    ``anchor_embeddings[i]`` is a fixed preservation target ordered as clean,
    positives, negatives for layout ``i``.  It is not a teacher for the action
    spectrum and never replaces a live positive or negative embedding.
    """
    if not layouts or not (
        len(layouts) == len(anchor_embeddings) == len(official_margins)
    ):
        raise ValueError("live shared E4 objective ledgers must align")
    clean_rank_terms: list[torch.Tensor] = []
    action_rank_terms: list[torch.Tensor] = []
    consistency_terms: list[torch.Tensor] = []
    floor_terms: list[torch.Tensor] = []
    preserve_terms: list[torch.Tensor] = []
    clean_margins: list[torch.Tensor] = []
    action_margins: list[torch.Tensor] = []
    for layout, anchor, official_margin in zip(
        layouts, anchor_embeddings, official_margins
    ):
        clean = encoded[int(layout["clean"])]
        action = encoded[int(layout["action"])]
        positive_indices = list(map(int, layout["positive"]))
        negative_indices = list(map(int, layout["negative"]))
        if not positive_indices or not negative_indices:
            raise RuntimeError("E4 action objective requires positive and negative spectra")
        positives = encoded[positive_indices]
        negatives = encoded[negative_indices]
        clean_margin = torch.max(positives @ clean) - torch.max(negatives @ clean)
        action_margin = torch.max(positives @ action) - torch.max(negatives @ action)
        clean_margins.append(clean_margin)
        action_margins.append(action_margin)
        clean_rank_terms.append(F.softplus((rank_margin - clean_margin) / temperature))
        # Never gate the action anchor. Softplus supplies its own smooth decay.
        action_rank_terms.append(F.softplus((rank_margin - action_margin) / temperature))
        consistency_terms.append(1.0 - torch.sum(clean * action))
        floor_terms.append(F.relu(float(official_margin) - margin_floor_slack - clean_margin))

        current = torch.cat((
            clean.unsqueeze(0), positives, negatives,
        ), dim=0)
        if anchor.shape != current.shape:
            raise RuntimeError(
                f"preservation target shape {tuple(anchor.shape)} != {tuple(current.shape)}"
            )
        preserve_terms.append(
            (1.0 - torch.sum(current * anchor.detach(), dim=1)).mean()
        )

    clean_rank = torch.stack(clean_rank_terms).mean()
    action_rank = torch.stack(action_rank_terms).mean()
    consistency = torch.stack(consistency_terms).mean()
    floor = torch.stack(floor_terms).mean()
    preserve = torch.stack(preserve_terms).mean()
    clean_margin_value = torch.stack(clean_margins)
    action_margin_value = torch.stack(action_margins)
    loss = (
        lambda_clean_rank * clean_rank
        + lambda_aug_rank * action_rank
        + lambda_consistency * consistency
        + lambda_margin_floor * floor
        + lambda_preserve * preserve
    )
    return loss, {
        "action_clean_rank": float(clean_rank.detach()),
        "action_aug_rank": float(action_rank.detach()),
        "action_consistency": float(consistency.detach()),
        "action_margin_floor": float(floor.detach()),
        "action_preserve": float(preserve.detach()),
        "action_clean_margin": float(clean_margin_value.mean().detach()),
        "action_aug_margin": float(action_margin_value.mean().detach()),
        "action_aug_rank_active_fraction": 1.0,
        "action_reference_gradient_detached": False,
        "query_equal_reduction": True,
        "complete_historical_e4_terms_present": True,
    }


def _clone_gradients(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _set_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("parameter and gradient ledgers differ")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone()


def _norm(values: Sequence[torch.Tensor | None], positions: Sequence[int] | None = None) -> float:
    selected = range(len(values)) if positions is None else positions
    total = sum(
        float(torch.sum(values[index].detach().float() ** 2).double())
        for index in selected if values[index] is not None
    )
    return math.sqrt(total)


def _dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    positions: Sequence[int],
) -> float:
    return sum(
        float(torch.sum(left[index].detach().float() * right[index].detach().float()).double())
        for index in positions
        if left[index] is not None and right[index] is not None
    )


def _add(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    scale: float = 1.0,
) -> list[torch.Tensor | None]:
    output: list[torch.Tensor | None] = []
    for base, action in zip(left, right):
        if base is None and action is None:
            output.append(None)
        elif base is None:
            output.append(float(scale) * action.detach())
        elif action is None:
            output.append(base.detach().clone())
        else:
            output.append(base.detach() + float(scale) * action.detach())
    return output


@dataclass(frozen=True)
class SeparatedE4ActionStepV3:
    action_active: bool
    action_rejected_as_opposed: bool
    e4_gradient_norm_before_clip: float
    action_gradient_norm_before_clip: float
    e4_clip_retention: float
    action_clip_retention: float
    natural_action_fraction_by_group: dict[str, float]
    final_action_fraction_by_group: dict[str, float]
    retained_action_scale_by_group: dict[str, float]
    e4_projection_retention_by_group: dict[str, float]
    final_to_e4_update_norm_ratio_by_group: dict[str, float]
    zero_action_exact_e4_equivalence: bool


class SeparatedE4ActionInjectorV3:
    """Compose independent E4-only and later-action-stream AdamW updates.

    The live optimizer's moments receive only the complete historical E4 base
    gradient.  A separate zero-weight-decay AdamW state receives only the full
    live-shared later-action E4-stream gradient.  The two independently
    materialised AdamW displacements are composed with the already validated
    exact-attribution kernel: every active parameter group receives the
    requested action fraction, while retaining the requested projection of the
    historical E4 displacement and respecting the total-norm ceiling.

    This is intentionally a *target* fraction, not a maximum.  Treating 0.25
    as a ceiling silently reintroduced the original signal-loss bug whenever
    the natural action displacement was smaller than the E4 displacement.
    """

    def __init__(
        self,
        optimizer: torch.optim.AdamW,
        parameters: Sequence[torch.nn.Parameter],
        *,
        target_action_fraction: float = 0.25,
        minimum_e4_projection_retention: float = 0.90,
        maximum_update_norm_ratio: float = 1.50,
    ) -> None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("V3 requires an AdamW E4 optimizer")
        if not 0 < target_action_fraction < 1:
            raise ValueError("target action fraction must lie in (0, 1)")
        if not 0 <= minimum_e4_projection_retention <= 1:
            raise ValueError("minimum E4 projection retention must lie in [0, 1]")
        if maximum_update_norm_ratio < 1:
            raise ValueError("maximum update norm ratio must be at least one")
        self.optimizer = optimizer
        self.parameters = list(parameters)
        position = {id(parameter): index for index, parameter in enumerate(self.parameters)}
        if len(position) != len(self.parameters) or not self.parameters:
            raise ValueError("V3 parameters must be non-empty and unique")
        self.group_positions: dict[str, list[int]] = {}
        action_groups: list[dict[str, object]] = []
        self.action_parameters = [
            torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
            for parameter in self.parameters
        ]
        for group_index, group in enumerate(optimizer.param_groups):
            name = str(group.get("group_name", f"group_{group_index}"))
            positions = [position[id(parameter)] for parameter in group["params"]]
            if name in self.group_positions:
                raise ValueError(f"duplicate optimizer group {name}")
            self.group_positions[name] = positions
            options = {key: value for key, value in group.items() if key != "params"}
            options["weight_decay"] = 0.0
            options["params"] = [self.action_parameters[index] for index in positions]
            action_groups.append(options)
        observed = [index for values in self.group_positions.values() for index in values]
        if len(observed) != len(self.parameters) or set(observed) != set(range(len(self.parameters))):
            raise ValueError("optimizer groups do not partition V3 parameters")
        self.action_optimizer = torch.optim.AdamW(action_groups)
        self.target_action_fraction = float(target_action_fraction)
        self.minimum_e4_projection_retention = float(
            minimum_e4_projection_retention
        )
        self.maximum_update_norm_ratio = float(maximum_update_norm_ratio)
        self._action_gradients: list[torch.Tensor | None] | None = None

    def audit_manifest(self) -> dict[str, object]:
        return {
            "injector": "separated_e4_action_adamw_v3",
            "e4_optimizer_moments_receive_only_e4_gradient": True,
            "action_optimizer_moments_receive_only_later_full_e4_stream_gradient": True,
            "action_optimizer_weight_decay": 0.0,
            "e4_and_action_gradients_clipped_independently": True,
            "exact_action_fraction_enforced_groupwise": True,
            "action_update_may_be_scaled_up_or_down_to_reach_target": True,
            "target_action_fraction": self.target_action_fraction,
            "minimum_e4_projection_retention": (
                self.minimum_e4_projection_retention
            ),
            "maximum_update_norm_ratio": self.maximum_update_norm_ratio,
            "zero_action_is_ordinary_e4_adamw": True,
        }

    def capture_action_gradient_(self, *, allow_zero: bool = False) -> None:
        if self._action_gradients is not None:
            raise RuntimeError("action gradient was captured twice")
        gradients = _clone_gradients(self.parameters)
        if _norm(gradients) <= 0 and not allow_zero:
            raise RuntimeError("live-shared E4 action gradient is zero")
        self._action_gradients = gradients

    def _clip(
        self,
        gradients: Sequence[torch.Tensor | None],
        maximum: float,
    ) -> tuple[list[torch.Tensor | None], float, float]:
        _set_gradients(self.parameters, gradients)
        raw_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, maximum))
        if not math.isfinite(raw_norm):
            raise RuntimeError("non-finite gradient norm")
        return (
            _clone_gradients(self.parameters),
            raw_norm,
            min(1.0, maximum / (raw_norm + 1e-6)),
        )

    def _action_updates(
        self,
        before: Sequence[torch.Tensor],
        gradients: Sequence[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        for live, shadow in zip(before, self.action_parameters):
            shadow.data.copy_(live)
        _set_gradients(self.action_parameters, gradients)
        self.action_optimizer.step()
        self.action_optimizer.zero_grad(set_to_none=True)
        return [
            None if gradient is None else live - shadow.detach()
            for live, shadow, gradient in zip(before, self.action_parameters, gradients)
        ]

    def step_(self, *, maximum_gradient_norm: float) -> SeparatedE4ActionStepV3:
        if self._action_gradients is None:
            raise RuntimeError("action gradient was not captured")
        if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm <= 0:
            raise ValueError("maximum gradient norm must be finite and positive")
        e4_raw = _clone_gradients(self.parameters)
        action_raw = self._action_gradients
        if _norm(e4_raw) <= 0:
            raise RuntimeError("historical E4 gradient is zero")
        e4_clipped, e4_norm, e4_retention = self._clip(e4_raw, maximum_gradient_norm)
        action_norm = _norm(action_raw)
        before = [parameter.detach().clone() for parameter in self.parameters]

        if action_norm <= 0:
            _set_gradients(self.parameters, e4_clipped)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._action_gradients = None
            groups = sorted(self.group_positions)
            return SeparatedE4ActionStepV3(
                action_active=False,
                action_rejected_as_opposed=False,
                e4_gradient_norm_before_clip=e4_norm,
                action_gradient_norm_before_clip=0.0,
                e4_clip_retention=e4_retention,
                action_clip_retention=1.0,
                natural_action_fraction_by_group={name: 0.0 for name in groups},
                final_action_fraction_by_group={name: 0.0 for name in groups},
                retained_action_scale_by_group={name: 0.0 for name in groups},
                e4_projection_retention_by_group={name: 1.0 for name in groups},
                final_to_e4_update_norm_ratio_by_group={name: 1.0 for name in groups},
                zero_action_exact_e4_equivalence=True,
            )

        action_clipped, _, action_retention = self._clip(
            action_raw, maximum_gradient_norm,
        )
        action_updates = self._action_updates(before, action_clipped)
        _set_gradients(self.parameters, e4_clipped)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        e4_updates = [
            None if gradient is None else left - parameter.detach()
            for left, parameter, gradient in zip(before, self.parameters, e4_clipped)
        ]

        natural_combined = _add(e4_updates, action_updates)
        composition = compose_safe_exact_corrective_updates_by_group(
            natural_combined,
            e4_updates,
            e4_updates,
            self.group_positions,
            target_attributable_fraction=self.target_action_fraction,
            minimum_protective_component_retention=(
                self.minimum_e4_projection_retention
            ),
            materialize_updates=True,
        )
        if not composition.all_groups_target_reached:
            raise RuntimeError("V3 exact action fraction was not reached")
        if not composition.all_groups_protective_floor_enforced:
            raise RuntimeError("V3 E4 projection floor was not enforced")

        final_updates = composition.updates
        natural = {
            name: float(report.original_attributable_fraction)
            for name, report in composition.parameter_groups.items()
        }
        final_fraction = {
            name: float(report.final_attributable_fraction)
            for name, report in composition.parameter_groups.items()
        }
        scales = {
            name: float(report.action_gain)
            for name, report in composition.parameter_groups.items()
        }
        projections: dict[str, float] = {}
        ratios: dict[str, float] = {}
        for name, positions in self.group_positions.items():
            base_sq = _dot(e4_updates, e4_updates, positions)
            if base_sq <= 0:
                raise RuntimeError("E4 update is zero in one optimizer group")
            projections[name] = _dot(final_updates, e4_updates, positions) / base_sq
            ratios[name] = _norm(final_updates, positions) / math.sqrt(base_sq)
            if (
                abs(final_fraction[name] - self.target_action_fraction) > 1e-6
            ):
                raise RuntimeError("V3 exact action fraction drifted")
            if ratios[name] > self.maximum_update_norm_ratio + 1e-6:
                raise RuntimeError("V3 total update norm cap failed")
            if (
                projections[name] + 1e-6
                < self.minimum_e4_projection_retention
            ):
                raise RuntimeError("V3 consumed the E4 protective direction")

        for parameter, left, update in zip(self.parameters, before, final_updates):
            parameter.data.copy_(left if update is None else left - update)
        self._action_gradients = None
        return SeparatedE4ActionStepV3(
            action_active=True,
            action_rejected_as_opposed=False,
            e4_gradient_norm_before_clip=e4_norm,
            action_gradient_norm_before_clip=action_norm,
            e4_clip_retention=e4_retention,
            action_clip_retention=action_retention,
            natural_action_fraction_by_group=natural,
            final_action_fraction_by_group=final_fraction,
            retained_action_scale_by_group=scales,
            e4_projection_retention_by_group=projections,
            final_to_e4_update_norm_ratio_by_group=ratios,
            zero_action_exact_e4_equivalence=False,
        )


def summarize_separated_e4_action_steps_v3(
    steps: Sequence[SeparatedE4ActionStepV3],
    *,
    target_action_fraction: float,
    minimum_e4_projection_retention: float,
    maximum_update_norm_ratio: float,
) -> dict[str, object]:
    if not steps:
        return {"enabled": False, "steps": 0, "gate_passed": False}
    groups = sorted(steps[0].final_action_fraction_by_group)
    active = [step for step in steps if step.action_active]
    zero = [step for step in steps if not step.action_active]
    if any(sorted(step.final_action_fraction_by_group) != groups for step in steps):
        raise RuntimeError("V3 optimizer groups changed during training")
    report: dict[str, object] = {
        "enabled": True,
        "steps": len(steps),
        "action_active_steps": len(active),
        "zero_action_steps": len(zero),
        "opposed_action_rejected_steps": sum(
            step.action_rejected_as_opposed for step in active
        ),
        "target_action_fraction": float(target_action_fraction),
        "minimum_e4_projection_retention": float(
            minimum_e4_projection_retention
        ),
        "maximum_update_norm_ratio": float(maximum_update_norm_ratio),
        "exact_action_fraction_reached": all(
            abs(step.final_action_fraction_by_group[name] - target_action_fraction)
            <= 1e-6
            for step in active for name in groups
        ),
        "zero_action_exact_e4": all(
            step.zero_action_exact_e4_equivalence for step in zero
        ),
        "group_final_action_fraction_p10": {
            name: (
                float(np.quantile([
                    step.final_action_fraction_by_group[name] for step in active
                ], 0.10)) if active else 0.0
            )
            for name in groups
        },
        "group_final_action_fraction_p50": {
            name: (
                float(np.quantile([
                    step.final_action_fraction_by_group[name] for step in active
                ], 0.50)) if active else 0.0
            )
            for name in groups
        },
        "group_minimum_e4_projection_retention": {
            name: min(
                step.e4_projection_retention_by_group[name] for step in steps
            )
            for name in groups
        },
        "group_maximum_final_to_e4_norm_ratio": {
            name: max(
                step.final_to_e4_update_norm_ratio_by_group[name] for step in steps
            )
            for name in groups
        },
    }
    report["action_transmission_nonzero"] = bool(
        active
        and int(report["opposed_action_rejected_steps"]) < len(active)
        and all(
            float(value) > 0.0
            for value in report["group_final_action_fraction_p50"].values()
        )
    )
    report["gate_passed"] = bool(
        report["exact_action_fraction_reached"]
        and report["zero_action_exact_e4"]
        and report["action_transmission_nonzero"]
        and all(
            abs(
                step.final_action_fraction_by_group[name]
                - target_action_fraction
            ) <= 1e-6
            for step in active for name in groups
        )
        and all(
            step.e4_projection_retention_by_group[name] + 1e-6
            >= minimum_e4_projection_retention
            for step in steps for name in groups
        )
        and all(
            step.final_to_e4_update_norm_ratio_by_group[name]
            <= maximum_update_norm_ratio + 1e-6
            for step in steps for name in groups
        )
    )
    return report
