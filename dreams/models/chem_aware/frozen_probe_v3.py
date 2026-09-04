"""Frozen chemical readout used only to supervise deployable spectrum PEFT.

The ridge map has no trainable parameters.  It is fitted from official DreaMS
identity centroids to training-fold molecule targets and discarded after
training.  Consequently, a chemical loss can only be reduced by changing the
shared spectrum encoder that is used for both queries and references.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FrozenProbeFit:
    input_mean: np.ndarray
    target_mean: np.ndarray
    weight: np.ndarray
    alpha: float
    examples: int


def fit_frozen_ridge_probe(
    inputs: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> FrozenProbeFit:
    """Fit a deterministic centered multi-output ridge map in dual form."""

    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if (
        alpha <= 0 or inputs.ndim != 2 or targets.ndim != 2
        or len(inputs) != len(targets) or len(inputs) < 2
        or not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets))
    ):
        raise ValueError("invalid frozen ridge probe inputs")
    input_mean = inputs.mean(axis=0, keepdims=True)
    target_mean = targets.mean(axis=0, keepdims=True)
    centered_input = np.asarray(inputs - input_mean, dtype=np.float64)
    centered_target = np.asarray(targets - target_mean, dtype=np.float64)
    if len(centered_input) <= centered_input.shape[1]:
        kernel = centered_input @ centered_input.T
        dual = np.linalg.solve(
            kernel + alpha * np.eye(len(kernel), dtype=np.float64), centered_target
        )
        weight = centered_input.T @ dual
    else:
        gram = centered_input.T @ centered_input
        weight = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
            centered_input.T @ centered_target,
        )
    return FrozenProbeFit(
        input_mean=input_mean.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        weight=weight.astype(np.float32),
        alpha=float(alpha),
        examples=int(len(inputs)),
    )


class FrozenChemicalProbe(nn.Module):
    """Non-trainable normalized chemical readout."""

    def __init__(self, fit: FrozenProbeFit):
        super().__init__()
        self.register_buffer("input_mean", torch.from_numpy(fit.input_mean.copy()))
        self.register_buffer("target_mean", torch.from_numpy(fit.target_mean.copy()))
        self.register_buffer("weight", torch.from_numpy(fit.weight.copy()))
        self.alpha = float(fit.alpha)
        self.examples = int(fit.examples)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.weight.shape[0]:
            raise RuntimeError("frozen probe input dimension mismatch")
        prediction = (
            (embeddings.float() - self.input_mean) @ self.weight + self.target_mean
        )
        return F.normalize(prediction, dim=-1, eps=1e-12)


def targeted_probe_listwise_loss(
    query_embeddings: torch.Tensor,
    official_molecule_scores: torch.Tensor,
    molecule_teacher: torch.Tensor,
    query_ptr: torch.Tensor,
    probe: FrozenChemicalProbe,
    margin_threshold: float = 0.01,
    temperature: float = 0.1,
    observable: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chemical candidate loss only for official-wrong or low-margin queries.

    The positive molecule is first in every complete candidate group.  A group
    is active only when its official positive-vs-hardest-negative margin is at
    most ``margin_threshold`` and at least one teacher-observable negative is
    available.  High-margin official-correct queries therefore receive exactly
    zero chemical gradient.
    """

    if margin_threshold < 0 or temperature <= 0:
        raise ValueError("invalid targeted probe threshold/temperature")
    if query_ptr.ndim != 1 or query_ptr.numel() != len(query_embeddings) + 1:
        raise RuntimeError("targeted probe query pointer mismatch")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(official_molecule_scores):
        raise RuntimeError("targeted probe pointer does not span molecule scores")
    if molecule_teacher.ndim != 2 or len(molecule_teacher) != len(official_molecule_scores):
        raise RuntimeError("targeted probe teacher shape mismatch")
    if observable is None:
        observable = torch.ones(
            len(molecule_teacher), device=molecule_teacher.device, dtype=torch.bool
        )
    if observable.shape != (len(molecule_teacher),):
        raise RuntimeError("targeted probe observability shape mismatch")
    if query_weights is not None and (
        query_weights.shape != (len(query_embeddings),) or torch.any(query_weights < 0)
    ):
        raise RuntimeError("invalid targeted probe query weights")

    predicted = probe(query_embeddings)
    teacher = F.normalize(molecule_teacher.float(), dim=-1, eps=1e-12)
    losses: list[torch.Tensor] = []
    active_indices: list[int] = []
    for query_index, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("targeted probe group needs a negative molecule")
        official = official_molecule_scores[left_i:right_i]
        margin = official[0] - torch.max(official[1:])
        group_observable = observable[left_i:right_i].bool()
        if (
            bool((margin <= margin_threshold).detach())
            and bool(group_observable[0])
            and bool(torch.any(group_observable[1:]))
        ):
            selected = torch.nonzero(group_observable, as_tuple=False).flatten()
            # Positive remains first because group_observable[0] is required.
            logits = teacher[left_i:right_i][selected] @ predicted[query_index]
            losses.append(-F.log_softmax(logits / temperature, dim=0)[0])
            active_indices.append(query_index)

    active = torch.zeros(
        len(query_embeddings), device=query_embeddings.device, dtype=torch.bool
    )
    if not losses:
        return query_embeddings.sum() * 0.0, active
    active[torch.tensor(active_indices, device=active.device, dtype=torch.long)] = True
    values = torch.stack(losses)
    if query_weights is None:
        return values.mean(), active
    selected_weights = query_weights[active]
    selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-12)
    return torch.sum(values * selected_weights), active


def targeted_probe_multiview_listwise_loss(
    query_embeddings: torch.Tensor,
    positive_reference_embeddings: torch.Tensor,
    positive_reference_ptr: torch.Tensor,
    official_molecule_scores: torch.Tensor,
    molecule_teacher: torch.Tensor,
    query_ptr: torch.Tensor,
    probe: FrozenChemicalProbe,
    margin_threshold: float = 0.01,
    temperature: float = 0.1,
    observable: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targeted chemical alignment for query and positive reference views.

    Selection is identical to :func:`targeted_probe_listwise_loss` and depends
    only on frozen official margins and teacher observability.  Within each
    selected query, the query spectrum and every reference spectrum of its
    positive molecule are aligned to the same candidate-teacher list.  View
    losses are averaged inside a query before identity-equal query weighting,
    so molecules with more library spectra do not receive more total weight.
    """

    if margin_threshold < 0 or temperature <= 0:
        raise ValueError("invalid targeted multiview probe threshold/temperature")
    if query_ptr.ndim != 1 or query_ptr.numel() != len(query_embeddings) + 1:
        raise RuntimeError("targeted multiview probe query pointer mismatch")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(official_molecule_scores):
        raise RuntimeError("targeted multiview probe pointer does not span molecule scores")
    if (
        positive_reference_embeddings.ndim != 2
        or positive_reference_embeddings.shape[1] != query_embeddings.shape[1]
        or positive_reference_ptr.ndim != 1
        or positive_reference_ptr.numel() != len(query_embeddings) + 1
        or int(positive_reference_ptr[0]) != 0
        or int(positive_reference_ptr[-1]) != len(positive_reference_embeddings)
        or torch.any(positive_reference_ptr[1:] <= positive_reference_ptr[:-1])
    ):
        raise RuntimeError("targeted multiview positive-reference ledger mismatch")
    if molecule_teacher.ndim != 2 or len(molecule_teacher) != len(official_molecule_scores):
        raise RuntimeError("targeted multiview probe teacher shape mismatch")
    if observable is None:
        observable = torch.ones(
            len(molecule_teacher), device=molecule_teacher.device, dtype=torch.bool
        )
    if observable.shape != (len(molecule_teacher),):
        raise RuntimeError("targeted multiview probe observability shape mismatch")
    if query_weights is not None and (
        query_weights.shape != (len(query_embeddings),) or torch.any(query_weights < 0)
    ):
        raise RuntimeError("invalid targeted multiview probe query weights")

    predicted_query = probe(query_embeddings)
    predicted_reference = probe(positive_reference_embeddings)
    teacher = F.normalize(molecule_teacher.float(), dim=-1, eps=1e-12)
    losses: list[torch.Tensor] = []
    active_indices: list[int] = []
    for query_index, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("targeted multiview probe group needs a negative molecule")
        official = official_molecule_scores[left_i:right_i]
        margin = official[0] - torch.max(official[1:])
        group_observable = observable[left_i:right_i].bool()
        if (
            bool((margin <= margin_threshold).detach())
            and bool(group_observable[0])
            and bool(torch.any(group_observable[1:]))
        ):
            selected = torch.nonzero(group_observable, as_tuple=False).flatten()
            target = teacher[left_i:right_i][selected]
            ref_left = int(positive_reference_ptr[query_index])
            ref_right = int(positive_reference_ptr[query_index + 1])
            views = torch.cat((
                predicted_query[query_index:query_index + 1],
                predicted_reference[ref_left:ref_right],
            ), dim=0)
            logits = views @ target.T
            view_losses = -F.log_softmax(logits / temperature, dim=1)[:, 0]
            losses.append(view_losses.mean())
            active_indices.append(query_index)

    active = torch.zeros(
        len(query_embeddings), device=query_embeddings.device, dtype=torch.bool
    )
    if not losses:
        return (
            (query_embeddings.sum() + positive_reference_embeddings.sum()) * 0.0,
            active,
        )
    active[torch.tensor(active_indices, device=active.device, dtype=torch.long)] = True
    values = torch.stack(losses)
    if query_weights is None:
        return values.mean(), active
    selected_weights = query_weights[active]
    selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-12)
    return torch.sum(values * selected_weights), active
