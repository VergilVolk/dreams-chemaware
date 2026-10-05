"""Mathematical core for orthogonal-action transfer into a shared embedding.

The candidate-conditioned teacher is used only to identify a boundary event.
The deployable student receives one clean spectrum and is trained to repair
that event through its ordinary shared cosine geometry.  No rule or candidate
feature is an input to the student.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EventBatch:
    """Label-oriented boundary events derived from a frozen teacher cache."""

    positive: np.ndarray
    negative: np.ndarray
    role: np.ndarray
    active: np.ndarray


def teacher_boundary_events(
    baseline_rank: np.ndarray,
    proposal_rank: np.ndarray,
    baseline_candidate: np.ndarray,
    proposed_candidate: np.ndarray,
    selected_slot: np.ndarray,
    positive_candidate: np.ndarray,
) -> EventBatch:
    """Orient selected teacher proposals using truth available during training.

    A correction promotes the true candidate over an erroneous official top-1.
    A protection event preserves an official true top-1 against the proposed
    challenger.  Selected proposals that touch neither side of the true-vs-top1
    boundary are assigned exact zero chemical weight.

    Role codes: 0 unsupported, 1 correction, 2 protection.
    """
    baseline_rank = np.asarray(baseline_rank, dtype=np.int64)
    proposal_rank = np.asarray(proposal_rank, dtype=np.int64)
    baseline_candidate = np.asarray(baseline_candidate, dtype=np.int64)
    proposed_candidate = np.asarray(proposed_candidate, dtype=np.int64)
    selected_slot = np.asarray(selected_slot, dtype=np.int64)
    positive_candidate = np.asarray(positive_candidate, dtype=np.int64)
    n = len(baseline_rank)
    if any(len(value) != n for value in (
        baseline_candidate, selected_slot, positive_candidate,
    )):
        raise ValueError("teacher event vectors have incompatible lengths")
    if proposal_rank.shape != proposed_candidate.shape or proposal_rank.shape[0] != n:
        raise ValueError("teacher proposal matrices have incompatible shapes")
    selected = selected_slot >= 0
    if np.any(selected_slot[selected] >= proposal_rank.shape[1]):
        raise ValueError("selected teacher slot is out of range")
    row = np.arange(n)
    chosen = np.full(n, -1, dtype=np.int64)
    chosen_rank = np.full(n, -1, dtype=np.int64)
    chosen[selected] = proposed_candidate[row[selected], selected_slot[selected]]
    chosen_rank[selected] = proposal_rank[row[selected], selected_slot[selected]]
    correction = (
        selected & (baseline_rank > 1) & (chosen == positive_candidate)
        & (chosen_rank == 1) & (baseline_candidate != positive_candidate)
    )
    protection = (
        selected & (baseline_rank == 1) & (baseline_candidate == positive_candidate)
        & (chosen != positive_candidate) & (chosen_rank > 1)
    )
    if np.any(correction & protection):
        raise RuntimeError("a teacher event cannot be correction and protection")
    role = np.zeros(n, dtype=np.int8)
    role[correction] = 1
    role[protection] = 2
    negative = np.full(n, -1, dtype=np.int64)
    negative[correction] = baseline_candidate[correction]
    negative[protection] = chosen[protection]
    positive = np.where(role > 0, positive_candidate, -1).astype(np.int64)
    return EventBatch(
        positive=positive,
        negative=negative,
        role=role,
        active=role > 0,
    )


def logmeanexp_segments(
    pair_score: torch.Tensor,
    segment_ptr: np.ndarray | torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Count-normalized smooth maximum for variable reference replicates.

    Subtracting ``log(n)`` makes a segment of identical replicate scores map
    back to that score exactly, so molecules with many spectra do not receive
    a purely combinatorial advantage.  Evaluation can still use exact max.
    """
    if pair_score.ndim != 1 or temperature <= 0:
        raise ValueError("pair scores must be a vector and temperature positive")
    ptr = torch.as_tensor(segment_ptr, dtype=torch.long, device=pair_score.device)
    if ptr.ndim != 1 or len(ptr) < 2 or int(ptr[0]) != 0 or int(ptr[-1]) != len(pair_score):
        raise ValueError("segment pointer does not partition pair scores")
    if bool(torch.any(ptr[1:] <= ptr[:-1])):
        raise ValueError("every molecule must have at least one reference spectrum")
    output = []
    tau = float(temperature)
    for left, right in zip(ptr[:-1], ptr[1:]):
        values = pair_score[int(left):int(right)]
        normalizer = torch.log(values.new_tensor(float(len(values))))
        output.append(tau * (torch.logsumexp(values / tau, dim=0) - normalizer))
    return torch.stack(output)


def one_sided_huber(shortfall: torch.Tensor, beta: float) -> torch.Tensor:
    """Huber penalty for positive constraint violations; zero above target."""
    if beta <= 0:
        raise ValueError("Huber transition must be positive")
    value = torch.relu(shortfall)
    return torch.where(
        value < beta,
        0.5 * value.square() / float(beta),
        value - 0.5 * float(beta),
    )


def event_calibrated_rankmax_loss(
    student_score: torch.Tensor,
    official_score: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    role: torch.Tensor,
    weight: torch.Tensor,
    target_margin: float = 0.005,
    protection_slack: float = 0.002,
    huber_beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Repair only teacher-identified true decision boundaries.

    For correction event i, the required residual displacement is

        d_i = eps - (s0(q,p) - s0(q,n)).

    The loss requires the student's candidate-centred residual to be at least
    ``d_i``.  Thus a -0.11 official margin requests about +0.115, rather than
    being silently clipped to an ineffective global 0.05.  Protection events
    retain the official positive margin up to ``protection_slack``.
    """
    if student_score.ndim != 2 or official_score.shape != student_score.shape:
        raise ValueError("student and official candidate scores must be matched matrices")
    if valid.shape != student_score.shape:
        raise ValueError("candidate validity mask has the wrong shape")
    batch, candidates = student_score.shape
    vectors = (positive, negative, role, weight)
    if any(value.ndim != 1 or len(value) != batch for value in vectors):
        raise ValueError("event vectors do not match the candidate-score batch")
    if target_margin <= 0 or protection_slack < 0 or huber_beta <= 0:
        raise ValueError("invalid event-margin hyperparameters")
    active = role > 0
    if bool(torch.any((role < 0) | (role > 2))):
        raise ValueError("unknown teacher event role")
    if bool(torch.any(weight < 0)):
        raise ValueError("event weights must be nonnegative")
    if not bool(torch.any(active)):
        zero = student_score.sum() * 0.0
        empty = student_score.new_zeros(batch)
        return zero, {
            "active": active, "official_margin": empty, "student_margin": empty,
            "required_residual": empty, "shortfall": empty,
        }
    row = torch.arange(batch, device=student_score.device)
    safe_positive = torch.clamp(positive, min=0, max=candidates - 1)
    safe_negative = torch.clamp(negative, min=0, max=candidates - 1)
    if bool(torch.any(active & (~valid[row, safe_positive] | ~valid[row, safe_negative]))):
        raise ValueError("active event points to an invalid candidate")
    student_margin = student_score[row, safe_positive] - student_score[row, safe_negative]
    official_margin = (
        official_score[row, safe_positive] - official_score[row, safe_negative]
    ).detach()
    correction_target = student_margin.new_full((batch,), float(target_margin))
    protection_target = torch.maximum(
        correction_target,
        official_margin - float(protection_slack),
    )
    target = torch.where(role == 1, correction_target, protection_target)
    residual = student_margin - official_margin
    required_residual = target - official_margin
    shortfall = required_residual - residual
    per_query = one_sided_huber(shortfall, huber_beta)
    active_weight = weight.to(student_score.dtype) * active.to(student_score.dtype)
    denominator = active_weight.sum()
    if float(denominator.detach()) <= 0:
        zero = student_score.sum() * 0.0
        return zero, {
            "active": active, "official_margin": official_margin,
            "student_margin": student_margin, "required_residual": required_residual,
            "shortfall": shortfall,
        }
    loss = torch.sum(per_query * active_weight) / denominator
    return loss, {
        "active": active,
        "official_margin": official_margin,
        "student_margin": student_margin,
        "required_residual": required_residual,
        "shortfall": shortfall,
    }
