"""Relation-complete Noise objective for a shared DreaMS encoder.

T1 exposes every selected positive-reference/negative-reference relation.
T3 scores the same references exactly as deployment does: max over spectra
inside a molecule, then listwise competition between candidate molecules.
Both terms are averaged first inside the clean/action anchor groups and then
over queries, so extra actions or reference multiplicity cannot increase a
query's training dose.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def proportional_interleave(clean_count: int, action_count: int) -> np.ndarray:
    """Deterministic proportional merge of two training step streams.

    Returns an int8 array of unit kinds (0 = clean batch, 1 = action event)
    in which both streams are spread evenly across the merged timeline: the
    dose share of each stream equals its share of units, action count is a
    schedule quantity rather than a per-query weight, and ties resolve only
    by stream order (clean first at identical keys).
    """
    clean_keys = np.arange(clean_count, dtype=np.float64) / max(clean_count, 1)
    action_keys = (
        np.arange(action_count, dtype=np.float64) / max(action_count, 1) + 1e-9
    )
    order = np.argsort(np.concatenate([clean_keys, action_keys]), kind="stable")
    kinds = np.concatenate([
        np.zeros(clean_count, dtype=np.int8),
        np.ones(action_count, dtype=np.int8),
    ])
    return kinds[order]


def relation_complete_loss(
    anchor_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    molecule_ptr: torch.Tensor,
    *,
    anchor_weights: torch.Tensor | None = None,
    triplet_margin: float = 0.1,
    listwise_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return equal-weight T1 multi-hinge and T3 molecule-max listwise loss.

    ``molecule_ptr`` describes one positive molecule first, followed by one or
    more negative molecules. All tensors are live and receive gradients.
    """
    if anchor_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise RuntimeError("T1/T3 embeddings must be matrices")
    if anchor_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise RuntimeError("anchor/reference embedding dimensions differ")
    if molecule_ptr.ndim != 1 or molecule_ptr.numel() < 3:
        raise RuntimeError("T1/T3 query needs one positive and at least one negative molecule")
    if int(molecule_ptr[0]) != 0 or int(molecule_ptr[-1]) != len(reference_embeddings):
        raise RuntimeError("molecule pointer does not span references")
    if torch.any(molecule_ptr[1:] <= molecule_ptr[:-1]):
        raise RuntimeError("T1/T3 contains an empty candidate molecule")
    if triplet_margin <= 0 or listwise_temperature <= 0:
        raise ValueError("T1/T3 margin and temperature must be positive")

    anchors = F.normalize(anchor_embeddings.float(), dim=1)
    references = F.normalize(reference_embeddings.float(), dim=1)
    if anchor_weights is None:
        weights = torch.full(
            (len(anchors),), 1.0 / len(anchors),
            device=anchors.device, dtype=anchors.dtype,
        )
    else:
        weights = anchor_weights.to(device=anchors.device, dtype=anchors.dtype)
        if weights.shape != (len(anchors),) or torch.any(weights < 0) or float(weights.sum()) <= 0:
            raise RuntimeError("invalid T1/T3 anchor weights")
        weights = weights / weights.sum()
    pair_scores = anchors @ references.T
    molecule_scores = torch.stack([
        torch.max(pair_scores[:, int(left):int(right)], dim=1).values
        for left, right in zip(molecule_ptr[:-1], molecule_ptr[1:])
    ], dim=1)
    # T3: exact molecule-max aggregation followed by complete candidate CE.
    listwise_per_anchor = -F.log_softmax(
        molecule_scores / listwise_temperature, dim=1,
    )[:, 0]
    listwise = torch.sum(weights * listwise_per_anchor)

    positive_right = int(molecule_ptr[1])
    positive_scores = pair_scores[:, :positive_right]
    negative_scores = pair_scores[:, positive_right:]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        raise RuntimeError("T1/T3 lost a positive or negative relation")
    # T1: every selected positive is compared with every selected negative,
    # but references are averaged inside each negative molecule before the
    # molecule means are averaged. A molecule with more replicate spectra
    # therefore cannot silently receive a larger dose.
    molecule_hinges = []
    molecule_active = []
    molecule_negative_means = []
    for left, right in zip(molecule_ptr[1:-1], molecule_ptr[2:]):
        local_negative = pair_scores[:, int(left):int(right)]
        raw = (
            triplet_margin
            + local_negative.unsqueeze(2)
            - positive_scores.unsqueeze(1)
        )
        molecule_hinges.append(torch.sum(weights * torch.relu(raw).mean(dim=(1, 2))))
        molecule_active.append(torch.sum(weights * (raw > 0).float().mean(dim=(1, 2))))
        molecule_negative_means.append(torch.sum(weights * local_negative.mean(dim=1)))
    hinge = torch.stack(molecule_hinges).mean()
    total = 0.5 * hinge + 0.5 * listwise
    return total, {
        "t1_multi_relation_hinge": hinge,
        "t3_molecule_max_listwise": listwise,
        "positive_similarity_mean": torch.sum(weights * positive_scores.mean(dim=1)),
        "negative_similarity_mean": torch.stack(molecule_negative_means).mean(),
        "active_relation_fraction": torch.stack(molecule_active).mean(),
        "anchor_count": torch.as_tensor(
            len(anchor_embeddings), device=total.device, dtype=torch.float32,
        ),
    }


def batch_relation_complete_loss(
    query_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    triplet_margin: float = 0.1,
    listwise_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average complete relation objectives over queries, never over events."""
    if not query_batches:
        raise RuntimeError("empty T1/T3 query batch")
    losses = []
    components: dict[str, list[torch.Tensor]] = {}
    for anchors, references, pointer, anchor_weights in query_batches:
        loss, detail = relation_complete_loss(
            anchors, references, pointer,
            anchor_weights=anchor_weights,
            triplet_margin=triplet_margin,
            listwise_temperature=listwise_temperature,
        )
        losses.append(loss)
        for name, value in detail.items():
            components.setdefault(name, []).append(value)
    return torch.stack(losses).mean(), {
        name: torch.stack(values).mean() for name, values in components.items()
    }
