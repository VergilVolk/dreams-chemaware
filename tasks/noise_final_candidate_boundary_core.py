"""Candidate-edge objectives for direct noise fine-tuning.

The peak action is never converted to an embedding target or one scalar margin.
For each query, every positive-reference x negative-molecule edge remains an
independent training observation.  Target/control differences only route and
weight those real identity-ranking edges; the label itself is always molecular
identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryLossResult:
    loss: torch.Tensor
    clean_boundary: torch.Tensor
    target_boundary: torch.Tensor
    counterfactual: torch.Tensor
    full_clean_rank: torch.Tensor
    active_edge_fraction: torch.Tensor
    mean_advantage: torch.Tensor
    effective_queries: int


def _require_edge_matrices(
    clean: list[torch.Tensor], target: list[torch.Tensor], control: list[torch.Tensor],
) -> None:
    if not clean or not (len(clean) == len(target) == len(control)):
        raise ValueError("clean/target/control edge lists must be equally sized and non-empty")
    for index, (clean_i, target_i, control_i) in enumerate(zip(clean, target, control)):
        if clean_i.ndim != 2 or clean_i.numel() == 0:
            raise ValueError(f"query {index} edge matrix must be non-empty [positive, negative]")
        if clean_i.shape != target_i.shape or clean_i.shape != control_i.shape:
            raise ValueError(f"query {index} edge matrix shapes differ")
        if not (torch.isfinite(clean_i).all() and torch.isfinite(target_i).all()
                and torch.isfinite(control_i).all()):
            raise ValueError(f"query {index} edge matrix contains non-finite values")


def _topk_negative_mask(clean_margin: torch.Tensor, topk_negatives: int) -> torch.Tensor:
    if topk_negatives < 1:
        raise ValueError("topk_negatives must be positive")
    # A molecule is hard when its best positive-reference margin is small.
    molecule_margin = clean_margin.min(dim=0).values
    k = min(int(topk_negatives), int(molecule_margin.numel()))
    indices = torch.topk(-molecule_margin.detach(), k=k, largest=True, sorted=False).indices
    mask = torch.zeros_like(molecule_margin, dtype=torch.bool)
    mask[indices] = True
    return mask.unsqueeze(0).expand_as(clean_margin)


def candidate_boundary_objective(
    clean_margin: list[torch.Tensor],
    target_margin: list[torch.Tensor],
    control_margin: list[torch.Tensor],
    *,
    rank_margin: float,
    rank_temperature: float,
    advantage_temperature: float,
    hard_temperature: float,
    topk_negatives: int,
    lambda_clean: float,
    lambda_target: float,
    lambda_counterfactual: float,
    lambda_full_clean: float,
) -> BoundaryLossResult:
    """Directly optimize candidate edges exposed by a beneficial peak action.

    ``target-control`` is retained as a matrix.  A detached, per-edge positive
    gate allocates clean-query training budget but never supplies a score to
    imitate.  Logistic losses have no already-satisfied 0.01 hinge.
    """
    _require_edge_matrices(clean_margin, target_margin, control_margin)
    if rank_temperature <= 0 or advantage_temperature <= 0 or hard_temperature <= 0:
        raise ValueError("all temperatures must be positive")
    if min(lambda_clean, lambda_target, lambda_counterfactual, lambda_full_clean) < 0:
        raise ValueError("loss weights must be non-negative")

    zero = clean_margin[0].sum() * 0.0
    clean_terms: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    counterfactual_terms: list[torch.Tensor] = []
    full_terms: list[torch.Tensor] = []
    active_fractions: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    effective_queries = 0

    for clean_i, target_i, control_i in zip(clean_margin, target_margin, control_margin):
        advantage = target_i - control_i
        topk = _topk_negative_mask(clean_i, topk_negatives)
        positive = (advantage.detach() > 0) & topk

        # Rank-sensitive weighting concentrates updates on the candidate edges
        # closest to a clean decision change. Positive action advantage affects
        # each edge independently and is never averaged into a query scalar.
        hardness = torch.exp(
            torch.clamp(-clean_i.detach() / hard_temperature, min=-20.0, max=20.0)
        )
        advantage_strength = torch.sigmoid(advantage.detach() / advantage_temperature)
        raw_weight = hardness * advantage_strength * positive.to(clean_i.dtype)
        denominator = raw_weight.sum()
        if float(denominator.detach()) > 0:
            weight = raw_weight / denominator
            clean_terms.append(
                torch.sum(weight * F.softplus((rank_margin - clean_i) / rank_temperature))
            )
            target_terms.append(
                torch.sum(weight * F.softplus((rank_margin - target_i) / rank_temperature))
            )
            # Unlike the old hinge, this remains differentiable for every
            # retained edge and compares target/control without rewarding the
            # control as another positive augmentation.
            # The matched-random spectrum is a direction control, not a
            # rejected identity example.  Stop its gradient: the optimizer may
            # improve the target branch, but may not manufacture an advantage
            # by damaging a chemically valid random augmentation.
            one_sided_advantage = target_i - control_i.detach()
            counterfactual_terms.append(torch.sum(
                weight * F.softplus(-one_sided_advantage / advantage_temperature)
            ))
            effective_queries += 1
        else:
            clean_terms.append(zero)
            target_terms.append(zero)
            counterfactual_terms.append(zero)

        topk_weight = topk.to(clean_i.dtype)
        topk_weight = topk_weight / topk_weight.sum().clamp_min(1.0)
        full_terms.append(
            torch.sum(topk_weight * F.softplus((rank_margin - clean_i) / rank_temperature))
        )
        active_fractions.append(positive.to(clean_i.dtype).mean())
        advantages.append(advantage.detach().mean())

    clean_loss = torch.stack(clean_terms).mean()
    target_loss = torch.stack(target_terms).mean()
    counterfactual_loss = torch.stack(counterfactual_terms).mean()
    full_loss = torch.stack(full_terms).mean()
    total = (
        lambda_clean * clean_loss
        + lambda_target * target_loss
        + lambda_counterfactual * counterfactual_loss
        + lambda_full_clean * full_loss
    )
    return BoundaryLossResult(
        loss=total,
        clean_boundary=clean_loss,
        target_boundary=target_loss,
        counterfactual=counterfactual_loss,
        full_clean_rank=full_loss,
        active_edge_fraction=torch.stack(active_fractions).mean(),
        mean_advantage=torch.stack(advantages).mean(),
        effective_queries=effective_queries,
    )


def candidate_safety_objective(
    current_margin: list[torch.Tensor],
    initial_margin: list[torch.Tensor],
    *,
    slack: float,
    topk_negatives: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Protect each current candidate edge, including candidate switches."""
    if not current_margin or len(current_margin) != len(initial_margin):
        raise ValueError("current/initial safety edge lists must be equally sized and non-empty")
    terms: list[torch.Tensor] = []
    violations: list[torch.Tensor] = []
    for index, (current_i, initial_i) in enumerate(zip(current_margin, initial_margin)):
        if current_i.shape != initial_i.shape or current_i.ndim != 2:
            raise ValueError(f"safety edge matrix {index} is malformed")
        topk = _topk_negative_mask(current_i, topk_negatives)
        deficit = F.relu(initial_i.detach() - float(slack) - current_i)
        selected = deficit[topk]
        terms.append(selected.mean())
        violations.append((selected > 0).to(current_i.dtype).mean())
    loss = torch.stack(terms).mean()
    return loss, {
        "candidate_safety_violation": float(loss.detach()),
        "candidate_safety_active_fraction": float(torch.stack(violations).mean().detach()),
    }
