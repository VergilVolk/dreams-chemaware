"""Core objectives and scheduling for loss-aware direct action fine-tuning.

This module deliberately contains no dataset or model code.  It makes the
three action semantics independently testable:

* corrective actions transfer a bounded margin improvement to the clean view
  and also train the real action view in the shared query/reference geometry;
* robustness actions train only their own view and cannot reward the clean
  view as if they were corrective;
* harmful actions select clean boundaries that need protection, while the
  harmful view is detached and is never imitated.

Every reduction is query-equal and family-equal.  The scheduler keeps every
optimizer step action-active while covering every protective microbatch once,
which removes the long protect-only tail in the v2 trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Generic, Sequence, TypeVar

import torch
import torch.nn.functional as F

from noise_corrected_transfer_objective_v4 import (
    v3_hard_capped_transfer_delta,
    v4_mass_neutral_transfer_delta,
)


T = TypeVar("T")


@dataclass(frozen=True)
class BalancedStep(Generic[T]):
    """One action-active optimizer step and its protective microbatches."""

    corrective: T
    corrective_scale: float
    protective: tuple[T, ...]


@dataclass(frozen=True)
class V3ScheduleGeometry:
    """Integer-only batch geometry for one direct-v3 training epoch.

    The auxiliary and protective panels can require more optimizer steps than
    the natural corrective packing.  In that case the *same ordered corrective
    queries* are split into a few more non-empty batches before recycling.  No
    query or action is added, dropped, or split across forwards.
    """

    corrective_queries: int
    robust_queries: int
    harmful_queries: int
    protective_queries: int
    registered_corrective_batch_size: int
    registered_protective_batch_size: int
    original_corrective_batches: int
    cap_safe_corrective_batches: int
    corrective_batches_added_by_cap_safe_repartition: int
    robust_batches: int
    harmful_batches: int
    auxiliary_batches: int
    protective_batches: int
    auxiliary_required_optimizer_steps: int
    protective_required_optimizer_steps: int
    required_optimizer_steps: int
    original_corrective_recycle_factor: float
    effective_corrective_recycle_factor: float
    configured_maximum_corrective_recycle_factor: float
    minimum_cap_safe_corrective_batch_size: int
    maximum_cap_safe_corrective_batch_size: int
    minimum_size_corrective_batches: int
    maximum_size_corrective_batches: int
    cap_safe_repartition_required: bool
    cap_safe_repartition_feasible: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CorrectiveActionResult:
    loss: torch.Tensor
    margin_transfer: torch.Tensor
    payload_rank: torch.Tensor
    payload_safety: torch.Tensor
    queries: int
    actions: int
    considered_transfer_edges: int
    active_transfer_edges: int
    capped_transfer_edges: int
    active_safety_edges: int
    transfer_edge_diagnostics: dict[str, dict[str, int]]


@dataclass(frozen=True)
class RobustActionResult:
    loss: torch.Tensor
    payload_rank: torch.Tensor
    safety_floor: torch.Tensor
    queries: int
    actions: int


@dataclass(frozen=True)
class HarmfulActionResult:
    loss: torch.Tensor
    boundary_rank: torch.Tensor
    baseline_floor: torch.Tensor
    queries: int
    actions: int
    active_damage_edges: int


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def v3_schedule_geometry(
    *,
    corrective_queries: int,
    robust_queries: int,
    harmful_queries: int,
    protective_queries: int,
    corrective_batch_size: int,
    protective_batch_size: int,
    maximum_auxiliary_microbatches_per_step: int,
    maximum_protective_microbatches_per_step: int,
    maximum_corrective_recycle_factor: float,
) -> V3ScheduleGeometry:
    """Return the unique cap-safe direct-v3 schedule before GPU work.

    Counts and capacity limits determine this geometry completely.  When the
    natural corrective packing would exceed the registered recycle cap, the
    function asks for the smallest larger number of corrective batches.  It
    never changes the optimizer-step count or any auxiliary/protective panel.
    """
    counts = {
        "corrective_queries": corrective_queries,
        "robust_queries": robust_queries,
        "harmful_queries": harmful_queries,
        "protective_queries": protective_queries,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if corrective_queries < 1:
        raise ValueError("at least one corrective query is required")
    capacities = {
        "corrective_batch_size": corrective_batch_size,
        "protective_batch_size": protective_batch_size,
        "maximum_auxiliary_microbatches_per_step": (
            maximum_auxiliary_microbatches_per_step
        ),
        "maximum_protective_microbatches_per_step": (
            maximum_protective_microbatches_per_step
        ),
    }
    for name, value in capacities.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    recycle_cap = float(maximum_corrective_recycle_factor)
    if not math.isfinite(recycle_cap) or recycle_cap < 1.0:
        raise ValueError("maximum corrective recycle factor must be finite and at least one")

    original_corrective_batches = _ceil_div(
        corrective_queries, corrective_batch_size,
    )
    robust_batches = _ceil_div(robust_queries, corrective_batch_size)
    harmful_batches = _ceil_div(harmful_queries, corrective_batch_size)
    auxiliary_batches = robust_batches + harmful_batches
    protective_batches = _ceil_div(protective_queries, protective_batch_size)
    auxiliary_steps = _ceil_div(
        auxiliary_batches, maximum_auxiliary_microbatches_per_step,
    )
    protective_steps = _ceil_div(
        protective_batches, maximum_protective_microbatches_per_step,
    )
    required_steps = max(
        1, original_corrective_batches, auxiliary_steps, protective_steps,
    )

    cap_safe_corrective_batches = max(
        original_corrective_batches,
        int(math.ceil(required_steps / recycle_cap)),
    )
    # Guard the floating division at arbitrary user-supplied caps without
    # weakening the comparison with a tolerance.
    while required_steps / cap_safe_corrective_batches > recycle_cap:
        cap_safe_corrective_batches += 1
    while (
        cap_safe_corrective_batches > original_corrective_batches
        and required_steps / (cap_safe_corrective_batches - 1) <= recycle_cap
    ):
        cap_safe_corrective_batches -= 1
    if cap_safe_corrective_batches > corrective_queries:
        raise RuntimeError(
            "v3 corrective schedule cannot satisfy the frozen recycle cap even "
            "with one complete query per batch: "
            f"queries={corrective_queries} steps={required_steps} "
            f"cap={recycle_cap} required_batches={cap_safe_corrective_batches}"
        )

    minimum_batch_size, maximum_size_batches = divmod(
        corrective_queries, cap_safe_corrective_batches,
    )
    maximum_batch_size = minimum_batch_size + int(maximum_size_batches > 0)
    minimum_size_batches = cap_safe_corrective_batches - maximum_size_batches
    effective_factor = required_steps / cap_safe_corrective_batches
    if effective_factor > recycle_cap:
        raise RuntimeError("internal cap-safe corrective schedule calculation failed")
    return V3ScheduleGeometry(
        corrective_queries=corrective_queries,
        robust_queries=robust_queries,
        harmful_queries=harmful_queries,
        protective_queries=protective_queries,
        registered_corrective_batch_size=corrective_batch_size,
        registered_protective_batch_size=protective_batch_size,
        original_corrective_batches=original_corrective_batches,
        cap_safe_corrective_batches=cap_safe_corrective_batches,
        corrective_batches_added_by_cap_safe_repartition=(
            cap_safe_corrective_batches - original_corrective_batches
        ),
        robust_batches=robust_batches,
        harmful_batches=harmful_batches,
        auxiliary_batches=auxiliary_batches,
        protective_batches=protective_batches,
        auxiliary_required_optimizer_steps=auxiliary_steps,
        protective_required_optimizer_steps=protective_steps,
        required_optimizer_steps=required_steps,
        original_corrective_recycle_factor=(
            required_steps / original_corrective_batches
        ),
        effective_corrective_recycle_factor=effective_factor,
        configured_maximum_corrective_recycle_factor=recycle_cap,
        minimum_cap_safe_corrective_batch_size=minimum_batch_size,
        maximum_cap_safe_corrective_batch_size=maximum_batch_size,
        minimum_size_corrective_batches=minimum_size_batches,
        maximum_size_corrective_batches=maximum_size_batches,
        cap_safe_repartition_required=(
            cap_safe_corrective_batches != original_corrective_batches
        ),
        cap_safe_repartition_feasible=True,
    )


def cap_safe_corrective_repartition(
    batches: Sequence[Sequence[T]],
    target_batches: int,
) -> list[list[T]]:
    """Evenly repartition one ordered corrective cycle without RNG use.

    Integer interpolation spreads smaller batches through the epoch instead of
    concentrating them in a tail.  Elements retain their exact flattened
    order, and each element remains wholly inside one forward batch.
    """
    if not batches or any(not batch for batch in batches):
        raise ValueError("corrective repartition requires non-empty batches")
    if isinstance(target_batches, bool) or not isinstance(target_batches, int):
        raise ValueError("target corrective batch count must be an integer")
    if target_batches < len(batches):
        raise ValueError("cap-safe corrective repartition cannot merge original batches")
    if target_batches == len(batches):
        return [list(batch) for batch in batches]
    flattened = [value for batch in batches for value in batch]
    if target_batches > len(flattened):
        raise ValueError("cap-safe corrective repartition cannot split a query")
    count = len(flattened)
    return [
        flattened[index * count // target_batches:
                  (index + 1) * count // target_batches]
        for index in range(target_batches)
    ]


def symmetric_live_action_consistency(
    clean_embedding: Sequence[torch.Tensor],
    action_embedding: Sequence[Sequence[torch.Tensor]],
    action_group: Sequence[Sequence[str]],
    *,
    query_weight: Sequence[float] | None = None,
    action_block_weight: Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    """E8-style live clean/action consistency without a teacher target.

    Both views are outputs of the same trainable encoder and both receive
    gradient.  The reduction is query-, mechanism- and family-equal, so adding
    more recipes cannot multiply an identity's optimizer dose.
    """
    if not clean_embedding or not (
        len(clean_embedding) == len(action_embedding) == len(action_group)
    ):
        raise ValueError("clean/action consistency lists must align")
    if action_block_weight is not None and len(action_block_weight) != len(action_group):
        raise ValueError("action block weights must align with consistency queries")
    terms = []
    for query_index, (clean, actions, groups) in enumerate(zip(
        clean_embedding, action_embedding, action_group,
    )):
        if not actions or len(actions) != len(groups):
            raise ValueError(f"query {query_index} consistency layout is malformed")
        if clean.ndim != 1 or not torch.isfinite(clean).all():
            raise ValueError(f"query {query_index} clean embedding is malformed")
        stack = torch.stack(list(actions))
        if stack.ndim != 2 or stack.shape[1:] != clean.shape or not torch.isfinite(stack).all():
            raise ValueError(f"query {query_index} action embeddings are malformed")
        block_weight = (
            None if action_block_weight is None else action_block_weight[query_index]
        )
        terms.append(_family_equal_mean(
            1.0 - stack @ clean, groups, block_weight=block_weight,
        ))
    return _query_weighted_mean(terms, query_weight)


def corrective_recycle_scale(
    step: BalancedStep[object],
    *,
    full_dose: bool,
    full_dose_epoch_multiplier: float = 1.0,
) -> float:
    """Choose identity-equalized full-dose or normalized epoch replay.

    ``step.corrective_scale`` is reciprocal to the number of times its batch is
    scheduled.  Multiplying it by the mean recycle factor gives every original
    batch the same cumulative full-dose mass while keeping mean per-step scale
    one even when the final recycle is incomplete.
    """
    multiplier = float(full_dose_epoch_multiplier) if full_dose else 1.0
    scale = multiplier * float(step.corrective_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("corrective recycle scale must be finite and positive")
    return scale


def balanced_action_step_plan(
    corrective_batches: Sequence[T],
    protective_batches: Sequence[T],
    *,
    seed: int,
    maximum_protective_microbatches_per_step: int | None = None,
    minimum_steps: int | None = None,
) -> list[BalancedStep[T]]:
    """Distribute all protective work over action-active optimizer steps.

    Protective batches are shuffled deterministically and divided as evenly
    as possible.  If a maximum protective load is supplied, corrective
    batches are cycled across enough optimizer steps to respect it.  The
    returned ``corrective_scale`` values sum to one for every original
    corrective batch, so this extra scheduling coverage is not extra action
    dose.  Callers accumulate/average protective microbatch gradients before
    the single optimizer update for each step.
    """
    if not corrective_batches:
        raise ValueError("at least one corrective batch is required")
    if (
        maximum_protective_microbatches_per_step is not None
        and maximum_protective_microbatches_per_step < 1
    ):
        raise ValueError("maximum protective microbatches must be positive")
    if minimum_steps is not None and minimum_steps < 1:
        raise ValueError("minimum optimizer steps must be positive")
    corr_order = list(range(len(corrective_batches)))
    risk_order = list(range(len(protective_batches)))
    rng = random.Random(int(seed))
    rng.shuffle(corr_order)
    rng.shuffle(risk_order)

    step_count = len(corr_order)
    if maximum_protective_microbatches_per_step is not None:
        required_steps = (
            len(risk_order) + maximum_protective_microbatches_per_step - 1
        ) // maximum_protective_microbatches_per_step
        step_count = max(step_count, required_steps)
    if minimum_steps is not None:
        step_count = max(step_count, int(minimum_steps))
    scheduled_corr = [corr_order[position % len(corr_order)] for position in range(step_count)]
    repeats = {
        corr_index: scheduled_corr.count(corr_index) for corr_index in corr_order
    }
    allocations: list[list[int]] = [[] for _ in range(step_count)]
    # Round-robin allocation gives counts differing by at most one without
    # duplicating or dropping any protective microbatch.
    for position, risk_index in enumerate(risk_order):
        allocations[position % step_count].append(risk_index)
    return [
        BalancedStep(
            corrective=corrective_batches[corr_index],
            corrective_scale=1.0 / repeats[corr_index],
            protective=tuple(protective_batches[index] for index in allocations[position]),
        )
        for position, corr_index in enumerate(scheduled_corr)
    ]


def _topk_mask(margin: torch.Tensor, topk_negatives: int) -> torch.Tensor:
    if margin.ndim != 1 or margin.numel() < 1 or not torch.isfinite(margin).all():
        raise ValueError("each margin must be a finite non-empty vector")
    if topk_negatives < 1:
        raise ValueError("topk_negatives must be positive")
    take = min(int(topk_negatives), int(margin.numel()))
    indices = torch.topk(-margin.detach(), k=take, largest=True, sorted=False).indices
    mask = torch.zeros_like(margin, dtype=torch.bool)
    mask[indices] = True
    return mask


def _view_complete_topk_mask(
    clean: torch.Tensor,
    actions: Sequence[torch.Tensor],
    topk_negatives: int,
    controls: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return one clean/action/control-complete edge mask per action.

    Selecting edges only from the clean query silently drops a candidate that
    becomes hardest after an intervention.  Every action therefore receives
    the union of the clean Top-k, its own Top-k, and (when present) its paired
    control Top-k.  All masks are discrete detached selections; gradients still
    flow only through the explicitly defined direct objectives.
    """
    if not actions or (controls is not None and len(controls) != len(actions)):
        raise ValueError("action/control lists must be aligned and non-empty")
    clean_mask = _topk_mask(clean, topk_negatives)
    masks = []
    for index, action in enumerate(actions):
        mask = clean_mask | _topk_mask(action, topk_negatives)
        if controls is not None:
            mask = mask | _topk_mask(controls[index], topk_negatives)
        masks.append(mask)
    return torch.stack(masks)


def _query_weighted_mean(
    values: Sequence[torch.Tensor],
    query_weight: Sequence[float] | None,
) -> torch.Tensor:
    stack = torch.stack(list(values))
    if query_weight is None:
        return stack.mean()
    weight = torch.as_tensor(query_weight, device=stack.device, dtype=stack.dtype)
    if (
        weight.shape != stack.shape
        or not torch.isfinite(weight).all()
        or bool(torch.any(weight <= 0))
    ):
        raise ValueError("query weights must be aligned, finite and positive")
    # Weights are frozen globally with mean one.  Do not renormalize inside a
    # minibatch, which would undo inverse-identity weighting stochastically.
    return torch.mean(stack * weight)


def _validate_action_layout(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    action_group: Sequence[Sequence[str]],
    *,
    control_margin: Sequence[Sequence[torch.Tensor]] | None = None,
) -> None:
    if not clean_margin or not (
        len(clean_margin) == len(action_margin) == len(action_group)
    ):
        raise ValueError("clean/action/group query lists must align and be non-empty")
    if control_margin is not None and len(control_margin) != len(clean_margin):
        raise ValueError("control query list must align")
    for query_index, (clean, actions, groups) in enumerate(
        zip(clean_margin, action_margin, action_group)
    ):
        if clean.ndim != 1 or clean.numel() < 1 or not torch.isfinite(clean).all():
            raise ValueError(f"query {query_index} clean margin is malformed")
        if not actions or len(actions) != len(groups):
            raise ValueError(f"query {query_index} action/group layout is malformed")
        controls = None if control_margin is None else control_margin[query_index]
        if controls is not None and len(controls) != len(actions):
            raise ValueError(f"query {query_index} control layout differs")
        for action_index, action in enumerate(actions):
            if action.shape != clean.shape or not torch.isfinite(action).all():
                raise ValueError(f"query {query_index} action {action_index} is malformed")
            if controls is not None:
                control = controls[action_index]
                if control.shape != clean.shape or not torch.isfinite(control).all():
                    raise ValueError(f"query {query_index} control {action_index} is malformed")


def _parse_action_group(group: str) -> tuple[str, str, str]:
    """Parse mechanism::source|family while keeping legacy labels valid."""
    value = str(group)
    mechanism, leaf = value.split("::", 1) if "::" in value else (value, value)
    source, family = leaf.split("|", 1) if "|" in leaf else (leaf, leaf)
    return mechanism, source, family


def _family_equal_mean(
    values: torch.Tensor,
    groups: Sequence[str],
    *,
    block_weight: Sequence[float] | None = None,
) -> torch.Tensor:
    """Reduce action -> source-within-family -> family -> mechanism."""
    if values.ndim != 1 or values.numel() != len(groups) or not groups:
        raise ValueError("values and groups must be aligned non-empty vectors")
    parsed = [_parse_action_group(group) for group in groups]
    weights = None
    if block_weight is not None:
        weights = torch.as_tensor(
            block_weight, device=values.device, dtype=values.dtype,
        )
        if (
            weights.shape != values.shape
            or not torch.isfinite(weights).all()
            or bool(torch.any(weights <= 0))
        ):
            raise ValueError("block weights must be aligned, finite and positive")
    block_terms = []
    for block in sorted({item[0] for item in parsed}):
        family_terms = []
        for family in sorted({item[2] for item in parsed if item[0] == block}):
            source_terms = []
            for source in sorted({
                item[1] for item in parsed
                if item[0] == block and item[2] == family
            }):
                mask = torch.as_tensor(
                    [item == (block, source, family) for item in parsed],
                    device=values.device,
                    dtype=torch.bool,
                )
                source_terms.append(values[mask].mean())
            family_terms.append(torch.stack(source_terms).mean())
        block_term = torch.stack(family_terms).mean()
        if weights is not None:
            block_values = weights[torch.as_tensor(
                [item[0] == block for item in parsed],
                device=values.device,
                dtype=torch.bool,
            )]
            if not torch.allclose(block_values, block_values[:1]):
                raise ValueError("all actions in one mechanism block need one weight")
            block_term = block_term * block_values[0]
        block_terms.append(block_term)
    # With explicit global weights, each mechanism coefficient already
    # contains the 1/M normalization.  Summing preserves equal total epoch
    # mass even when many queries contain only the high-coverage P mechanism.
    return (
        torch.stack(block_terms).sum()
        if weights is not None else torch.stack(block_terms).mean()
    )


def _family_equal_weighted_edges(
    edge_loss: torch.Tensor,
    raw_weight: torch.Tensor,
    groups: Sequence[str],
    *,
    block_weight: Sequence[float] | None = None,
) -> torch.Tensor:
    """Family-equal reduction over an [actions, edges] loss matrix."""
    if edge_loss.ndim != 2 or raw_weight.shape != edge_loss.shape:
        raise ValueError("edge loss and weight must be aligned matrices")
    if edge_loss.shape[0] != len(groups):
        raise ValueError("action groups do not align with edge matrices")
    weights = None
    if block_weight is not None:
        weights = torch.as_tensor(
            block_weight, device=edge_loss.device, dtype=edge_loss.dtype,
        )
        if (
            weights.ndim != 1
            or weights.numel() != edge_loss.shape[0]
            or not torch.isfinite(weights).all()
            or bool(torch.any(weights <= 0))
        ):
            raise ValueError("block weights must align with action rows")
    block_terms = []
    parsed = [_parse_action_group(group) for group in groups]
    eps = torch.finfo(edge_loss.dtype).eps
    for block in sorted({item[0] for item in parsed}):
        family_terms = []
        for family in sorted({item[2] for item in parsed if item[0] == block}):
            source_terms = []
            for source in sorted({
                item[1] for item in parsed
                if item[0] == block and item[2] == family
            }):
                mask = torch.as_tensor(
                    [item == (block, source, family) for item in parsed],
                    device=edge_loss.device,
                    dtype=torch.bool,
                )
                leaf_weight = raw_weight[mask]
                denominator = leaf_weight.sum()
                if bool(denominator.detach() > 0):
                    source_terms.append(
                        torch.sum(
                            edge_loss[mask] * leaf_weight / denominator.clamp_min(eps)
                        )
                    )
            if source_terms:
                family_terms.append(torch.stack(source_terms).mean())
        if family_terms:
            block_term = torch.stack(family_terms).mean()
            if weights is not None:
                block_values = weights[torch.as_tensor(
                    [item[0] == block for item in parsed],
                    device=edge_loss.device,
                    dtype=torch.bool,
                )]
                if not torch.allclose(block_values, block_values[:1]):
                    raise ValueError("all actions in one mechanism block need one weight")
                block_term = block_term * block_values[0]
            block_terms.append(block_term)
    if not block_terms:
        return edge_loss.sum() * 0.0
    return (
        torch.stack(block_terms).sum()
        if weights is not None else torch.stack(block_terms).mean()
    )


def _transfer_diagnostic_scopes(group: str, semantic: str) -> tuple[str, ...]:
    mechanism, source, family = _parse_action_group(group)
    return (
        "all",
        f"mechanism={mechanism}",
        f"family={family}",
        f"source={source}",
        f"source_family={source}|{family}",
        f"control_semantic={semantic}",
    )


def _add_transfer_diagnostics(
    output: dict[str, dict[str, int]],
    *,
    group: str,
    semantic: str,
    edge_mask: torch.Tensor,
    clean_gain: torch.Tensor,
    control_gain: torch.Tensor,
    active: torch.Tensor,
    capped: torch.Tensor,
) -> None:
    considered = int(edge_mask.sum().item())
    counts = {
        "considered": considered,
        "action_better_clean": int(((clean_gain > 0) & edge_mask).sum().item()),
        "action_better_control": int(((control_gain > 0) & edge_mask).sum().item()),
        "active": int((active & edge_mask).sum().item()),
        "clean_limited": int(((clean_gain <= control_gain) & edge_mask).sum().item()),
        "control_limited": int(((control_gain < clean_gain) & edge_mask).sum().item()),
        "capped": int((capped & edge_mask).sum().item()),
    }
    for scope in _transfer_diagnostic_scopes(group, semantic):
        target = output.setdefault(scope, {name: 0 for name in counts})
        for name, value in counts.items():
            target[name] += value


def corrective_action_objective(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    control_margin: Sequence[Sequence[torch.Tensor]],
    action_group: Sequence[Sequence[str]],
    *,
    rank_margin: float,
    rank_temperature: float,
    topk_negatives: int,
    margin_transfer_fraction: float,
    margin_transfer_cap: float,
    lambda_margin_transfer: float,
    lambda_payload_rank: float,
    action_safety_slack: float = 0.0,
    lambda_payload_safety: float = 0.0,
    query_weight: Sequence[float] | None = None,
    action_control_semantic: Sequence[Sequence[str]] | None = None,
    action_block_weight: Sequence[Sequence[float]] | None = None,
    transfer_target_allocation: str = "hard_cap",
    maximum_transfer_cap_factor: float = 2.0,
) -> CorrectiveActionResult:
    """Bounded clean transfer plus a genuine gradient through action views.

    The conservative transfer target is detached: actions decide where a
    clean margin may move, but do not act as an embedding teacher.  Payload
    ranking and its one-sided clean-relative safety floor are intentionally
    separate and send gradients through the real action view.  The trainer
    retains shared query/reference rank updates, which the mature E8 factor
    experiment found necessary.
    """
    _validate_action_layout(
        clean_margin, action_margin, action_group, control_margin=control_margin,
    )
    if action_control_semantic is not None and len(action_control_semantic) != len(action_group):
        raise ValueError("control semantics must align with action queries")
    if action_block_weight is not None and len(action_block_weight) != len(action_group):
        raise ValueError("action block weights must align with action queries")
    if (
        rank_temperature <= 0
        or margin_transfer_cap <= 0
        or action_safety_slack < 0
    ):
        raise ValueError(
            "temperatures/cap must be positive and action safety slack non-negative"
        )
    if not 0 <= margin_transfer_fraction <= 1:
        raise ValueError("margin_transfer_fraction must be in [0, 1]")
    if transfer_target_allocation not in {"hard_cap", "mass_neutral_monotone"}:
        raise ValueError("unsupported transfer-target allocation")
    if maximum_transfer_cap_factor < 1:
        raise ValueError("maximum transfer cap factor must be at least one")
    if min(
        lambda_margin_transfer, lambda_payload_rank, lambda_payload_safety,
    ) < 0:
        raise ValueError("loss weights must be non-negative")

    transfer_terms = []
    payload_terms = []
    safety_terms = []
    action_count = 0
    considered_edges = 0
    active_edges = 0
    capped_edges = 0
    active_safety_edges = 0
    transfer_diagnostics: dict[str, dict[str, int]] = {}
    for query_index, (clean, actions, controls, groups) in enumerate(zip(
        clean_margin, action_margin, control_margin, action_group,
    )):
        semantics = (
            list(action_control_semantic[query_index])
            if action_control_semantic is not None else [
                "wrong_identity_direction"
                if str(group).split("::", 1)[0] == "P" else "matched_neutral"
                for group in groups
            ]
        )
        if (
            len(semantics) != len(actions)
            or any(value not in {
                "matched_neutral", "wrong_identity_direction", "clean_fallback",
            } for value in semantics)
        ):
            raise ValueError("action control semantics are malformed")
        block_weight = (
            None if action_block_weight is None else action_block_weight[query_index]
        )
        if block_weight is not None and len(block_weight) != len(actions):
            raise ValueError("action block weights differ from action rows")
        action_stack = torch.stack(list(actions))
        control_stack = torch.stack(list(controls))
        edge_mask = _view_complete_topk_mask(
            clean, actions, topk_negatives, controls,
        )
        clean_gain = F.relu(action_stack.detach() - clean.detach().unsqueeze(0))
        control_gain = F.relu(action_stack.detach() - control_stack.detach())
        # This remains conservative for both meanings of control.  For N/A4,
        # the second term is matched-perturbation specificity; for P it is
        # correct-vs-wrong-direction specificity.  The clean term is always
        # the hard ceiling, so a very bad P control can never inflate transfer.
        delta_uncapped = torch.minimum(clean_gain, control_gain)
        hard_delta = v3_hard_capped_transfer_delta(
            delta_uncapped, edge_mask, hard_cap=margin_transfer_cap,
        )
        active = (hard_delta > 0) & edge_mask
        capped = (delta_uncapped >= float(margin_transfer_cap)) & active
        delta = (
            hard_delta
            if transfer_target_allocation == "hard_cap"
            else v4_mass_neutral_transfer_delta(
                delta_uncapped, active, groups,
                hard_cap=margin_transfer_cap,
                maximum_cap_factor=maximum_transfer_cap_factor,
            )
        )
        target = clean.detach().unsqueeze(0) + float(margin_transfer_fraction) * delta
        transfer_edge = F.smooth_l1_loss(
            clean.unsqueeze(0).expand_as(target),
            target,
            beta=float(rank_temperature),
            reduction="none",
        )
        transfer_terms.append(_family_equal_weighted_edges(
            transfer_edge,
            active.to(transfer_edge.dtype),
            groups,
            block_weight=block_weight,
        ))
        payload_per_action = torch.stack([
            F.softplus((float(rank_margin) - action) / rank_temperature)[mask].mean()
            for action, mask in zip(actions, edge_mask)
        ])
        payload_terms.append(_family_equal_mean(
            payload_per_action, groups, block_weight=block_weight,
        ))
        safety_per_action = torch.stack([
            F.relu(clean.detach() - float(action_safety_slack) - action)[mask].mean()
            for action, mask in zip(actions, edge_mask)
        ])
        safety_terms.append(_family_equal_mean(
            safety_per_action, groups, block_weight=block_weight,
        ))
        for position, (group, semantic) in enumerate(zip(groups, semantics)):
            _add_transfer_diagnostics(
                transfer_diagnostics,
                group=str(group),
                semantic=str(semantic),
                edge_mask=edge_mask[position],
                clean_gain=clean_gain[position],
                control_gain=control_gain[position],
                active=active[position],
                capped=capped[position],
            )
        action_count += len(actions)
        considered_edges += int(edge_mask.sum().item())
        active_edges += int(active.sum().item())
        capped_edges += int(capped.sum().item())
        active_safety_edges += int(sum(
            int(((clean.detach() - float(action_safety_slack) - action) > 0)[mask].sum())
            for action, mask in zip(actions, edge_mask)
        ))

    transfer = _query_weighted_mean(transfer_terms, query_weight)
    payload = _query_weighted_mean(payload_terms, query_weight)
    safety = _query_weighted_mean(safety_terms, query_weight)
    total = (
        float(lambda_margin_transfer) * transfer
        + float(lambda_payload_rank) * payload
        + float(lambda_payload_safety) * safety
    )
    return CorrectiveActionResult(
        loss=total,
        margin_transfer=transfer,
        payload_rank=payload,
        payload_safety=safety,
        queries=len(clean_margin),
        actions=action_count,
        considered_transfer_edges=considered_edges,
        active_transfer_edges=active_edges,
        capped_transfer_edges=capped_edges,
        active_safety_edges=active_safety_edges,
        transfer_edge_diagnostics=transfer_diagnostics,
    )


def robust_action_objective(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    action_group: Sequence[Sequence[str]],
    *,
    rank_margin: float,
    rank_temperature: float,
    topk_negatives: int,
    safety_slack: float,
    lambda_payload_rank: float,
    lambda_safety_floor: float,
    query_weight: Sequence[float] | None = None,
    action_block_weight: Sequence[Sequence[float]] | None = None,
) -> RobustActionResult:
    """Train valid robust views without treating them as corrective targets."""
    _validate_action_layout(clean_margin, action_margin, action_group)
    if rank_temperature <= 0 or safety_slack < 0:
        raise ValueError("temperature must be positive and slack non-negative")
    if min(lambda_payload_rank, lambda_safety_floor) < 0:
        raise ValueError("loss weights must be non-negative")
    rank_terms = []
    safety_terms = []
    action_count = 0
    if action_block_weight is not None and len(action_block_weight) != len(action_group):
        raise ValueError("action block weights must align with robust queries")
    for query_index, (clean, actions, groups) in enumerate(
        zip(clean_margin, action_margin, action_group)
    ):
        block_weight = (
            None if action_block_weight is None else action_block_weight[query_index]
        )
        edge_mask = _view_complete_topk_mask(clean, actions, topk_negatives)
        ranks = torch.stack([
            F.softplus((float(rank_margin) - action) / rank_temperature)[mask].mean()
            for action, mask in zip(actions, edge_mask)
        ])
        floors = torch.stack([
            F.relu(clean.detach() - float(safety_slack) - action)[mask].mean()
            for action, mask in zip(actions, edge_mask)
        ])
        rank_terms.append(_family_equal_mean(
            ranks, groups, block_weight=block_weight,
        ))
        safety_terms.append(_family_equal_mean(
            floors, groups, block_weight=block_weight,
        ))
        action_count += len(actions)
    rank = _query_weighted_mean(rank_terms, query_weight)
    safety = _query_weighted_mean(safety_terms, query_weight)
    total = float(lambda_payload_rank) * rank + float(lambda_safety_floor) * safety
    return RobustActionResult(
        loss=total,
        payload_rank=rank,
        safety_floor=safety,
        queries=len(clean_margin),
        actions=action_count,
    )


def harmful_boundary_objective(
    clean_margin: Sequence[torch.Tensor],
    harmful_margin: Sequence[Sequence[torch.Tensor]],
    baseline_margin: Sequence[torch.Tensor],
    action_group: Sequence[Sequence[str]],
    *,
    rank_margin: float,
    rank_temperature: float,
    topk_negatives: int,
    harmful_damage_slack: float,
    baseline_floor_slack: float,
    lambda_boundary_rank: float,
    lambda_baseline_floor: float,
    query_weight: Sequence[float] | None = None,
    action_block_weight: Sequence[Sequence[float]] | None = None,
) -> HarmfulActionResult:
    """Use harmful contents to select clean boundaries, never to imitate them."""
    _validate_action_layout(clean_margin, harmful_margin, action_group)
    if len(baseline_margin) != len(clean_margin):
        raise ValueError("baseline margins must align with clean margins")
    if rank_temperature <= 0 or min(harmful_damage_slack, baseline_floor_slack) < 0:
        raise ValueError("temperature/slacks are invalid")
    if min(lambda_boundary_rank, lambda_baseline_floor) < 0:
        raise ValueError("loss weights must be non-negative")

    boundary_terms = []
    floor_terms = []
    action_count = 0
    active_edges = 0
    if action_block_weight is not None and len(action_block_weight) != len(action_group):
        raise ValueError("action block weights must align with harmful queries")
    for query_index, (clean, harmful, baseline, groups) in enumerate(
        zip(clean_margin, harmful_margin, baseline_margin, action_group)
    ):
        block_weight = (
            None if action_block_weight is None else action_block_weight[query_index]
        )
        if baseline.shape != clean.shape or not torch.isfinite(baseline).all():
            raise ValueError(f"query {query_index} baseline margin is malformed")
        harmful_stack = torch.stack(list(harmful))
        edge_mask = _view_complete_topk_mask(clean, harmful, topk_negatives)
        damage = F.relu(
            clean.detach().unsqueeze(0)
            - harmful_stack.detach()
            - float(harmful_damage_slack)
        ) * edge_mask
        active_edges += int((damage > 0).sum().item())
        boundary_edge = F.softplus(
            (float(rank_margin) - clean) / rank_temperature
        ).unsqueeze(0).expand_as(damage)
        floor_edge = F.relu(
            baseline.detach() - float(baseline_floor_slack) - clean
        ).unsqueeze(0).expand_as(damage)
        boundary_terms.append(_family_equal_weighted_edges(
            boundary_edge, damage, groups, block_weight=block_weight,
        ))
        floor_terms.append(_family_equal_weighted_edges(
            floor_edge, damage, groups, block_weight=block_weight,
        ))
        action_count += len(harmful)
    boundary = _query_weighted_mean(boundary_terms, query_weight)
    floor = _query_weighted_mean(floor_terms, query_weight)
    total = float(lambda_boundary_rank) * boundary + float(lambda_baseline_floor) * floor
    return HarmfulActionResult(
        loss=total,
        boundary_rank=boundary,
        baseline_floor=floor,
        queries=len(clean_margin),
        actions=action_count,
        active_damage_edges=active_edges,
    )
