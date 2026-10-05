"""Evaluator-aligned objectives for direct noise-action fine-tuning.

The retrieval evaluator scores each candidate *molecule* by the maximum score
over its spectra.  This module keeps that reduction inside the differentiable
objective.  Frozen, out-of-fold action weights may allocate extra training
dose, but the extra dose is paid to the clean molecular-identity boundary;
an action that is already easy therefore cannot make the clean update vanish.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DirectBoundaryResult:
    loss: torch.Tensor
    clean_rank: torch.Tensor
    corrective_clean_rank: torch.Tensor
    action_rank: torch.Tensor
    counterfactual: torch.Tensor
    margin_transfer: torch.Tensor
    action_safety: torch.Tensor
    effective_queries: int
    qualified_actions: int
    unqualified_actions: int


@dataclass(frozen=True)
class DirectTargetResult:
    """Target-only direct injection; matched controls are not a training gate."""

    loss: torch.Tensor
    clean_rank: torch.Tensor
    action_conditioned_clean_rank: torch.Tensor
    action_rank: torch.Tensor
    action_safety: torch.Tensor
    queries: int
    actions: int


def direct_action_to_clean_invariance(
    clean_embedding: Sequence[torch.Tensor],
    action_embedding: Sequence[Sequence[torch.Tensor]],
    action_group: Sequence[Sequence[str]] | None = None,
) -> torch.Tensor:
    """Move clean toward its raw-action representations, one dose per query.

    Action vectors are detached.  This makes the transfer direction explicit:
    action structure updates the deployed clean embedding instead of being
    erased by a symmetric consistency loss.  Action multiplicity is averaged
    within each query.
    """
    if not clean_embedding or len(clean_embedding) != len(action_embedding):
        raise ValueError("clean/action embedding query lists must align and be non-empty")
    groups_by_query = action_group if action_group is not None else [
        ["all"] * len(actions) for actions in action_embedding
    ]
    if len(groups_by_query) != len(action_embedding):
        raise ValueError("action groups must align to action embeddings")
    terms: list[torch.Tensor] = []
    for query_index, (clean, actions, groups) in enumerate(zip(
        clean_embedding, action_embedding, groups_by_query,
    )):
        if clean.ndim != 1 or not torch.isfinite(clean).all() or not actions:
            raise ValueError(f"query {query_index} clean/action embeddings are malformed")
        if any(action.shape != clean.shape or not torch.isfinite(action).all() for action in actions):
            raise ValueError(f"query {query_index} action embedding shape differs")
        if len(groups) != len(actions):
            raise ValueError(f"query {query_index} action groups are misaligned")
        values = [1.0 - torch.sum(clean * action.detach()) for action in actions]
        family_terms = []
        for family in sorted(set(map(str, groups))):
            family_terms.append(torch.stack([
                value for value, group in zip(values, groups) if str(group) == family
            ]).mean())
        terms.append(torch.stack(family_terms).mean())
    return torch.stack(terms).mean()


def direct_target_boundary_objective(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    *,
    rank_margin: float,
    rank_temperature: float,
    topk_negatives: int,
    action_safety_slack: float,
    lambda_clean: float,
    lambda_action_conditioned_clean: float,
    lambda_action_rank: float,
    lambda_action_safety: float,
    action_group: Sequence[Sequence[str]] | None = None,
) -> DirectTargetResult:
    """Inject every legal raw target without distillation or control gating.

    Each query contributes once to every branch.  Multiple valid action
    prefixes are averaged inside the query, so path multiplicity cannot turn
    into optimizer dose.  The action view is trained against the same live
    molecule-max candidate boundary as the clean query.
    """
    if not clean_margin or len(clean_margin) != len(action_margin):
        raise ValueError("clean/action query lists must be equally sized and non-empty")
    if rank_temperature <= 0 or action_safety_slack < 0:
        raise ValueError("rank temperature must be positive and safety slack non-negative")
    if min(
        lambda_clean, lambda_action_conditioned_clean,
        lambda_action_rank, lambda_action_safety,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    clean_terms: list[torch.Tensor] = []
    conditioned_clean_terms: list[torch.Tensor] = []
    action_terms: list[torch.Tensor] = []
    safety_terms: list[torch.Tensor] = []
    action_count = 0
    groups_by_query = action_group if action_group is not None else [
        ["all"] * len(actions) for actions in action_margin
    ]
    if len(groups_by_query) != len(action_margin):
        raise ValueError("action groups must align to action margins")
    for query_index, (clean, actions, groups) in enumerate(zip(
        clean_margin, action_margin, groups_by_query,
    )):
        if clean.ndim != 1 or clean.numel() < 1 or not torch.isfinite(clean).all():
            raise ValueError(f"query {query_index} clean margin is malformed")
        if not actions:
            raise ValueError(f"query {query_index} has no target action")
        if any(action.shape != clean.shape or not torch.isfinite(action).all() for action in actions):
            raise ValueError(f"query {query_index} action margins are malformed")
        if len(groups) != len(actions):
            raise ValueError(f"query {query_index} action groups are misaligned")
        topk = _topk_mask(clean, topk_negatives)
        clean_loss = F.softplus((float(rank_margin) - clean) / rank_temperature)[topk].mean()
        per_action_rank = torch.stack([
            F.softplus((float(rank_margin) - action) / rank_temperature)[topk].mean()
            for action in actions
        ])
        per_action_safety = torch.stack([
            F.relu(clean.detach() - float(action_safety_slack) - action)[topk].mean()
            for action in actions
        ])
        clean_terms.append(clean_loss)
        # One fixed extra clean-boundary dose signals that this query supports
        # a legal action; it is deliberately independent of action count.
        conditioned_clean_terms.append(clean_loss)
        family_rank = []
        family_safety = []
        for family in sorted(set(map(str, groups))):
            mask = torch.as_tensor(
                [str(group) == family for group in groups],
                device=per_action_rank.device, dtype=torch.bool,
            )
            family_rank.append(per_action_rank[mask].mean())
            family_safety.append(per_action_safety[mask].mean())
        action_terms.append(torch.stack(family_rank).mean())
        safety_terms.append(torch.stack(family_safety).mean())
        action_count += len(actions)
    clean_loss = torch.stack(clean_terms).mean()
    conditioned_clean_loss = torch.stack(conditioned_clean_terms).mean()
    action_loss = torch.stack(action_terms).mean()
    safety_loss = torch.stack(safety_terms).mean()
    total = (
        lambda_clean * clean_loss
        + lambda_action_conditioned_clean * conditioned_clean_loss
        + lambda_action_rank * action_loss
        + lambda_action_safety * safety_loss
    )
    return DirectTargetResult(
        loss=total, clean_rank=clean_loss,
        action_conditioned_clean_rank=conditioned_clean_loss,
        action_rank=action_loss, action_safety=safety_loss,
        queries=len(clean_margin), actions=action_count,
    )


def molecule_max_margin_from_scores(
    positive_scores: torch.Tensor,
    negative_molecule_scores: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return evaluator-aligned margins against each negative molecule."""
    if positive_scores.ndim != 1 or positive_scores.numel() < 1:
        raise ValueError("positive_scores must be a non-empty vector")
    if not negative_molecule_scores:
        raise ValueError("at least one negative molecule is required")
    if not torch.isfinite(positive_scores).all():
        raise ValueError("positive_scores contains non-finite values")
    negative_best: list[torch.Tensor] = []
    for index, scores in enumerate(negative_molecule_scores):
        if scores.ndim != 1 or scores.numel() < 1:
            raise ValueError(f"negative molecule {index} must be a non-empty vector")
        if not torch.isfinite(scores).all():
            raise ValueError(f"negative molecule {index} contains non-finite values")
        negative_best.append(torch.max(scores))
    return torch.max(positive_scores) - torch.stack(negative_best)


def molecule_max_margin(
    query_embedding: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_molecule_embeddings: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute cosine/dot-product molecular margins without scalar distillation."""
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a vector")
    if positive_embeddings.ndim != 2 or positive_embeddings.shape[1] != query_embedding.numel():
        raise ValueError("positive_embeddings must be [spectra, embedding_dim]")
    negative_scores: list[torch.Tensor] = []
    for index, embeddings in enumerate(negative_molecule_embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != query_embedding.numel():
            raise ValueError(
                f"negative molecule {index} must be [spectra, embedding_dim]"
            )
        negative_scores.append(embeddings @ query_embedding)
    return molecule_max_margin_from_scores(
        positive_embeddings @ query_embedding, negative_scores,
    )


def _topk_mask(margin: torch.Tensor, topk_negatives: int) -> torch.Tensor:
    if margin.ndim != 1 or margin.numel() < 1:
        raise ValueError("each molecular margin must be a non-empty vector")
    if topk_negatives < 1:
        raise ValueError("topk_negatives must be positive")
    count = min(int(topk_negatives), int(margin.numel()))
    indices = torch.topk(-margin.detach(), k=count, largest=True, sorted=False).indices
    mask = torch.zeros_like(margin, dtype=torch.bool)
    mask[indices] = True
    return mask


def _validate_views(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    control_margin: Sequence[Sequence[torch.Tensor]],
    action_weight: Sequence[torch.Tensor],
) -> None:
    if not clean_margin or not (
        len(clean_margin) == len(action_margin) == len(control_margin) == len(action_weight)
    ):
        raise ValueError("clean/action/control/weight query lists must be equally sized")
    for query_index, (clean, actions, controls, weights) in enumerate(
        zip(clean_margin, action_margin, control_margin, action_weight)
    ):
        if clean.ndim != 1 or clean.numel() < 1 or not torch.isfinite(clean).all():
            raise ValueError(f"query {query_index} clean molecular margin is malformed")
        if len(actions) != len(controls) or weights.ndim != 1 or weights.numel() != len(actions):
            raise ValueError(f"query {query_index} action layout differs")
        if not torch.isfinite(weights).all() or bool(torch.any(weights < 0)):
            raise ValueError(f"query {query_index} action weights must be finite and non-negative")
        for action_index, (action, control) in enumerate(zip(actions, controls)):
            if action.shape != clean.shape or control.shape != clean.shape:
                raise ValueError(
                    f"query {query_index} action {action_index} molecular shape differs"
                )
            if not (torch.isfinite(action).all() and torch.isfinite(control).all()):
                raise ValueError(f"query {query_index} action {action_index} is non-finite")


def direct_boundary_objective(
    clean_margin: Sequence[torch.Tensor],
    action_margin: Sequence[Sequence[torch.Tensor]],
    control_margin: Sequence[Sequence[torch.Tensor]],
    action_weight: Sequence[torch.Tensor],
    *,
    rank_margin: float,
    rank_temperature: float,
    advantage_temperature: float,
    topk_negatives: int,
    action_safety_slack: float,
    lambda_clean: float,
    lambda_corrective_clean: float,
    lambda_action_rank: float,
    lambda_counterfactual: float,
    lambda_action_safety: float,
    action_group: Sequence[Sequence[str]] | None = None,
    margin_transfer_fraction: float = 0.0,
    margin_transfer_cap: float = 0.10,
    lambda_margin_transfer: float = 0.0,
) -> DirectBoundaryResult:
    """Optimize clean molecular boundaries, with actions as bounded curricula.

    ``action_weight`` must come from a frozen training-side qualification rule
    (for example formula-OOF).  Zero-weight actions receive no corrective or
    ranking reward.  They only enter the one-sided safety term.  Within a
    query, action and edge weights sum to one, so duplicating an action cannot
    multiply its training dose.
    """
    _validate_views(clean_margin, action_margin, control_margin, action_weight)
    if rank_temperature <= 0 or advantage_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if action_safety_slack < 0:
        raise ValueError("action_safety_slack must be non-negative")
    lambdas = (
        lambda_clean, lambda_corrective_clean, lambda_action_rank,
        lambda_counterfactual, lambda_action_safety, lambda_margin_transfer,
    )
    if min(lambdas) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0 <= margin_transfer_fraction <= 1 or margin_transfer_cap <= 0:
        raise ValueError("margin transfer fraction/cap is invalid")
    groups_by_query = action_group if action_group is not None else [
        ["all"] * len(actions) for actions in action_margin
    ]
    if len(groups_by_query) != len(action_margin):
        raise ValueError("action groups must align to action margins")

    zero = clean_margin[0].sum() * 0.0
    clean_terms: list[torch.Tensor] = []
    corrective_terms: list[torch.Tensor] = []
    action_terms: list[torch.Tensor] = []
    counterfactual_terms: list[torch.Tensor] = []
    margin_transfer_terms: list[torch.Tensor] = []
    safety_terms: list[torch.Tensor] = []
    effective_queries = 0
    qualified_actions = 0
    unqualified_actions = 0

    for clean, actions, controls, weights, groups in zip(
        clean_margin, action_margin, control_margin, action_weight, groups_by_query,
    ):
        if len(groups) != len(actions):
            raise ValueError("action groups are misaligned within a query")
        topk = _topk_mask(clean, topk_negatives)
        clean_edge_loss = F.softplus((float(rank_margin) - clean) / rank_temperature)
        clean_terms.append(clean_edge_loss[topk].mean())

        qualified = weights.detach() > 0
        qualified_actions += int(qualified.sum().item())
        unqualified_actions += int((~qualified).sum().item())
        raw_edges: list[torch.Tensor] = []
        selected_actions: list[torch.Tensor] = []
        selected_controls: list[torch.Tensor] = []
        selected_groups: list[str] = []
        for action, control, weight, keep, group in zip(actions, controls, weights, qualified, groups):
            if not bool(keep):
                continue
            # Current target/control margins only distribute a fixed per-query
            # dose.  Detachment prevents the router from becoming a teacher.
            advantage_gate = torch.sigmoid(
                (action.detach() - control.detach()) / advantage_temperature
            )
            raw_edges.append(weight.detach() * advantage_gate * topk.to(clean.dtype))
            selected_actions.append(action)
            selected_controls.append(control)
            selected_groups.append(str(group))

        if raw_edges:
            raw = torch.stack(raw_edges)
            # First normalize within action family, then average families.  A
            # 19-cell family cannot dilute a one-cell family merely by having
            # more recipe/dose variants.
            normalized = torch.zeros_like(raw)
            unique_groups = sorted(set(selected_groups))
            for group in unique_groups:
                mask = torch.as_tensor(
                    [value == group for value in selected_groups],
                    device=raw.device, dtype=torch.bool,
                )
                family_raw = raw[mask]
                normalized[mask] = (
                    family_raw / family_raw.sum().clamp_min(torch.finfo(raw.dtype).eps)
                    / len(unique_groups)
                )
            corrective_terms.append(torch.sum(normalized * clean_edge_loss.unsqueeze(0)))
            action_stack = torch.stack(selected_actions)
            control_stack = torch.stack(selected_controls)
            action_edge_loss = F.softplus(
                (float(rank_margin) - action_stack) / rank_temperature
            )
            action_terms.append(torch.sum(normalized * action_edge_loss))
            # The control is a valid random augmentation, not a negative label.
            # It is detached so an apparent advantage cannot be manufactured by
            # explicitly pushing the control down.
            counterfactual_terms.append(torch.sum(
                normalized * F.softplus(
                    -(action_stack - control_stack.detach()) / advantage_temperature
                )
            ))
            conservative_delta = torch.minimum(
                F.relu(action_stack.detach() - clean.detach().unsqueeze(0)),
                F.relu(action_stack.detach() - control_stack.detach()),
            ).clamp_max(float(margin_transfer_cap))
            target = clean.detach().unsqueeze(0) + float(margin_transfer_fraction) * conservative_delta
            margin_transfer_terms.append(torch.sum(
                normalized * F.smooth_l1_loss(
                    clean.unsqueeze(0).expand_as(target), target,
                    beta=float(rank_temperature), reduction="none",
                )
            ))
            effective_queries += 1
        else:
            corrective_terms.append(zero)
            action_terms.append(zero)
            counterfactual_terms.append(zero)
            margin_transfer_terms.append(zero)

        per_action_safety: list[torch.Tensor] = []
        for action in actions:
            deficit = F.relu(clean.detach() - float(action_safety_slack) - action)
            per_action_safety.append(deficit[topk].mean())
        if per_action_safety:
            safety_stack = torch.stack(per_action_safety)
            family_safety = []
            for group in sorted(set(map(str, groups))):
                mask = torch.as_tensor(
                    [str(value) == group for value in groups],
                    device=safety_stack.device, dtype=torch.bool,
                )
                family_safety.append(safety_stack[mask].mean())
            safety_terms.append(torch.stack(family_safety).mean())
        else:
            safety_terms.append(zero)

    clean_loss = torch.stack(clean_terms).mean()
    corrective_loss = torch.stack(corrective_terms).mean()
    action_loss = torch.stack(action_terms).mean()
    counterfactual_loss = torch.stack(counterfactual_terms).mean()
    margin_transfer_loss = torch.stack(margin_transfer_terms).mean()
    safety_loss = torch.stack(safety_terms).mean()
    total = (
        lambda_clean * clean_loss
        + lambda_corrective_clean * corrective_loss
        + lambda_action_rank * action_loss
        + lambda_counterfactual * counterfactual_loss
        + lambda_margin_transfer * margin_transfer_loss
        + lambda_action_safety * safety_loss
    )
    return DirectBoundaryResult(
        loss=total,
        clean_rank=clean_loss,
        corrective_clean_rank=corrective_loss,
        action_rank=action_loss,
        counterfactual=counterfactual_loss,
        margin_transfer=margin_transfer_loss,
        action_safety=safety_loss,
        effective_queries=effective_queries,
        qualified_actions=qualified_actions,
        unqualified_actions=unqualified_actions,
    )


def calibrate_action_scale(
    *,
    action_gradient_norm: float,
    safety_gradient_norm: float,
    target_action_to_safety_ratio: float,
    safety_stream_weight: float = 1.0,
    scale_cap: float = 16.0,
) -> float:
    """Balance branches without the legacy lower clamp at one."""
    values = (
        action_gradient_norm, safety_gradient_norm, target_action_to_safety_ratio,
        safety_stream_weight, scale_cap,
    )
    if not all(torch.isfinite(torch.tensor(float(value))) for value in values):
        raise ValueError("gradient calibration inputs must be finite")
    if action_gradient_norm <= 0 or safety_gradient_norm < 0:
        raise ValueError("gradient norms are invalid")
    if target_action_to_safety_ratio < 0 or safety_stream_weight < 0 or scale_cap <= 0:
        raise ValueError("gradient calibration weights are invalid")
    if safety_gradient_norm == 0:
        # An inactive hinge supplies no reference scale.  Returning zero would
        # silently disable the entire corrective stream at exact initialization.
        return 1.0
    requested = (
        float(target_action_to_safety_ratio)
        * float(safety_stream_weight)
        * float(safety_gradient_norm)
        / float(action_gradient_norm)
    )
    return min(float(scale_cap), max(0.0, requested))
