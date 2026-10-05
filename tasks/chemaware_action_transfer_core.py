"""Mathematical primitives for auditing ChemAware action-to-clean transfer.

The qualified chemical action and the deployable clean spectrum are different
views of the same query.  This module deliberately separates three questions:

1. Which query value is used in the forward retrieval calculation?
2. Which query receives the backward gradient?
3. Are candidate spectra allowed to receive that gradient?

Keeping those choices explicit prevents a useful action view from being
confused with a useful shared-encoder update.  All scores use the deployment
rule: maximum spectrum similarity within each candidate molecule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


TRANSFER_ROUTES = (
    "clean_query_only",
    "action_query_only",
    "action_forward_clean_backward",
    "raw_shared_action",
)


@dataclass(frozen=True)
class RetrievalLoss:
    """One listwise retrieval objective and its observable forward geometry."""

    loss: torch.Tensor
    scores: torch.Tensor
    margin: torch.Tensor


@dataclass(frozen=True)
class RoutedPairLoss:
    """Clean-query boundary chosen by an action view, with no target regression."""

    loss: torch.Tensor
    margin: torch.Tensor
    positive_reference: int
    negative_reference: int


def _validated_ptr(molecule_ptr: Sequence[int] | torch.Tensor, pairs: int) -> list[int]:
    if isinstance(molecule_ptr, torch.Tensor):
        ptr = [int(value) for value in molecule_ptr.detach().cpu().tolist()]
    else:
        ptr = [int(value) for value in molecule_ptr]
    if len(ptr) < 3:
        raise ValueError("retrieval list requires one positive and at least one negative molecule")
    if ptr[0] != 0 or ptr[-1] != pairs:
        raise ValueError("molecule_ptr must span every candidate spectrum")
    if any(right <= left for left, right in zip(ptr[:-1], ptr[1:])):
        raise ValueError("every candidate molecule must contain at least one spectrum")
    return ptr


def straight_through_value(
    forward_value: torch.Tensor, backward_value: torch.Tensor,
) -> torch.Tensor:
    """Use ``forward_value`` numerically while differentiating ``backward_value``.

    Both tensors must describe the same representation space.  This is a
    gradient-routing operator, not a teacher target: no action score, margin or
    embedding is minimized as a regression target.
    """
    if forward_value.shape != backward_value.shape:
        raise ValueError("straight-through values must have identical shapes")
    return backward_value + (forward_value - backward_value).detach()


def molecule_max_scores(
    query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    molecule_ptr: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Return max-over-reference spectrum scores for each candidate molecule."""
    if query.ndim != 1 or candidate_embeddings.ndim != 2:
        raise ValueError("query must be 1-D and candidates must be 2-D")
    if candidate_embeddings.shape[1] != query.shape[0]:
        raise ValueError("query and candidate embedding dimensions do not match")
    ptr = _validated_ptr(molecule_ptr, len(candidate_embeddings))
    pair_scores = candidate_embeddings @ query
    return torch.stack([
        torch.max(pair_scores[left:right]) for left, right in zip(ptr[:-1], ptr[1:])
    ])


def positive_vs_hardest_negative_margin(scores: torch.Tensor) -> torch.Tensor:
    """Deployment-aligned margin; the unique positive molecule is position zero."""
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("scores must be a one-dimensional positive-first candidate list")
    return scores[0] - torch.max(scores[1:])


def routed_listwise_loss(
    clean_query: torch.Tensor,
    action_query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    molecule_ptr: Sequence[int] | torch.Tensor,
    *,
    route: str,
    temperature: float,
) -> RetrievalLoss:
    """Construct one explicit action-to-clean gradient route.

    ``clean_query_only``
        Clean forward value, gradient only through the clean query.
    ``action_query_only``
        Action forward value, gradient only through the action query.
    ``action_forward_clean_backward``
        Action forward value, but its listwise gradient is delivered to the
        clean-query encoder path.  Candidate embeddings are frozen.
    ``raw_shared_action``
        Historical route: action query and every candidate receive gradients.

    Detaching candidates in the first three routes is essential: it prevents a
    117-action bank from changing commonly reused references more strongly than
    it changes the intended query boundary.
    """
    if route not in TRANSFER_ROUTES:
        raise ValueError(f"unknown ChemAware transfer route: {route}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if clean_query.shape != action_query.shape:
        raise ValueError("clean and action query embeddings must align")

    if route == "clean_query_only":
        query = clean_query
        candidates = candidate_embeddings.detach()
    elif route == "action_query_only":
        query = action_query
        candidates = candidate_embeddings.detach()
    elif route == "action_forward_clean_backward":
        query = straight_through_value(action_query, clean_query)
        candidates = candidate_embeddings.detach()
    else:
        query = action_query
        candidates = candidate_embeddings

    scores = molecule_max_scores(query, candidates, molecule_ptr)
    loss = -F.log_softmax(scores / float(temperature), dim=0)[0]
    return RetrievalLoss(
        loss=loss,
        scores=scores,
        margin=positive_vs_hardest_negative_margin(scores),
    )


def action_routed_clean_pair_loss(
    clean_query: torch.Tensor,
    action_query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    molecule_ptr: Sequence[int] | torch.Tensor,
    *,
    temperature: float,
    target_margin: float = 0.0,
) -> RoutedPairLoss:
    """Train a clean query on the real-reference boundary exposed by an action.

    The action selects one positive reference and the hardest negative reference
    using the same candidate list as deployment.  The differentiable loss is
    then evaluated only on the clean query, while candidate references are
    detached.  Consequently, chemical knowledge controls *which boundary is
    trained* but is never used as a score, embedding, or margin target.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(target_margin)) or float(target_margin) < 0:
        raise ValueError("target_margin must be finite and non-negative")
    if clean_query.shape != action_query.shape:
        raise ValueError("clean and action query embeddings must align")
    ptr = _validated_ptr(molecule_ptr, len(candidate_embeddings))
    frozen_candidates = candidate_embeddings.detach()
    with torch.no_grad():
        action_pair_scores = frozen_candidates @ action_query.detach()
        positive_reference = int(torch.argmax(action_pair_scores[ptr[0]:ptr[1]]))
        negative_reference = ptr[1] + int(torch.argmax(action_pair_scores[ptr[1]:]))
    positive_score = torch.sum(
        clean_query * frozen_candidates[positive_reference], dim=-1,
    )
    negative_score = torch.sum(
        clean_query * frozen_candidates[negative_reference], dim=-1,
    )
    margin = positive_score - negative_score
    loss = F.softplus((float(target_margin) - margin) / float(temperature))
    return RoutedPairLoss(
        loss=loss,
        margin=margin,
        positive_reference=positive_reference,
        negative_reference=negative_reference,
    )


Gradient = torch.Tensor | None


def detached_gradients(
    value: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool = False,
) -> list[Gradient]:
    """Differentiate a scalar without mutating ``.grad`` optimizer buffers."""
    gradients = torch.autograd.grad(
        value,
        list(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [gradient.detach() if gradient is not None else None for gradient in gradients]


def gradient_dot(first: Sequence[Gradient], second: Sequence[Gradient]) -> float:
    if len(first) != len(second):
        raise ValueError("gradient collections must align")
    return float(sum(
        torch.sum(left.float() * right.float()).detach()
        for left, right in zip(first, second)
        if left is not None and right is not None
    ))


def gradient_norm(gradients: Sequence[Gradient]) -> float:
    squared = sum(
        float(torch.sum(value.float() * value.float()).detach())
        for value in gradients if value is not None
    )
    return math.sqrt(squared)


def gradient_cosine(first: Sequence[Gradient], second: Sequence[Gradient]) -> float:
    denominator = gradient_norm(first) * gradient_norm(second)
    return gradient_dot(first, second) / denominator if denominator > 0 else float("nan")


def subtract_gradients(
    minuend: Sequence[Gradient], subtrahend: Sequence[Gradient],
) -> list[Gradient]:
    """Return a contrast for auditing; its sign is not assumed to be beneficial."""
    if len(minuend) != len(subtrahend):
        raise ValueError("gradient collections must align")
    output: list[Gradient] = []
    for left, right in zip(minuend, subtrahend):
        if left is None and right is None:
            output.append(None)
        elif left is None:
            output.append(-right)
        elif right is None:
            output.append(left)
        else:
            output.append(left - right)
    return output


def first_order_loss_decrease(
    update_gradient: Sequence[Gradient], validation_loss_gradient: Sequence[Gradient],
) -> float:
    """Predicted validation-loss decrease per unit learning rate.

    For ``theta' = theta - eta * update_gradient``, the first-order loss change
    is ``-eta * <g_validation, g_update>``.  Positive output is therefore good.
    """
    return gradient_dot(update_gradient, validation_loss_gradient)


def first_order_metric_gain(
    update_gradient: Sequence[Gradient], metric_gradient: Sequence[Gradient],
) -> float:
    """Predicted metric increase per unit learning rate; positive is beneficial."""
    return -gradient_dot(update_gradient, metric_gradient)


def scale_gradients(gradients: Iterable[Gradient], scale: float) -> list[Gradient]:
    if not math.isfinite(float(scale)):
        raise ValueError("gradient scale must be finite")
    return [None if value is None else value * float(scale) for value in gradients]
