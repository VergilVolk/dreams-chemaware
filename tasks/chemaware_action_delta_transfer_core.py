"""Pure losses for transferring a frozen peak action into a clean embedding.

The chemical target is not an ICEBERG candidate score.  It is the change that
the already-qualified observed-peak action causes in the *official DreaMS*
candidate geometry.  The action view and reference embeddings are detached;
only the live clean-query embedding receives this chemical gradient.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def optimizer_descent_geometry(
    before: list[torch.Tensor],
    backbone_parameters: list[torch.nn.Parameter],
    head_parameters: list[torch.nn.Parameter],
) -> dict[str, dict[str, float]]:
    """Measure chemical-gradient alignment with a realized optimizer step."""
    parameters = [*backbone_parameters, *head_parameters]
    if len(before) != len(parameters):
        raise ValueError("optimizer geometry snapshot and parameters are not aligned")
    head_ids = {id(parameter) for parameter in head_parameters}
    accumulators = {
        "all": [0.0, 0.0, 0.0],
        "backbone": [0.0, 0.0, 0.0],
        "head": [0.0, 0.0, 0.0],
    }
    for initial, parameter in zip(before, parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        descent = initial - parameter.detach()
        values = (
            float(torch.sum(gradient.detach().float() ** 2)),
            float(torch.sum(descent.float() ** 2)),
            float(torch.sum(gradient.detach().float() * descent.float())),
        )
        group = "head" if id(parameter) in head_ids else "backbone"
        for name in ("all", group):
            for index, value in enumerate(values):
                accumulators[name][index] += value
    output: dict[str, dict[str, float]] = {}
    for name, (gradient_sq, update_sq, dot) in accumulators.items():
        gradient_norm = float(np.sqrt(max(gradient_sq, 0.0)))
        update_norm = float(np.sqrt(max(update_sq, 0.0)))
        denominator = gradient_norm * update_norm
        output[name] = {
            "postclip_gradient_norm": gradient_norm,
            "descent_update_norm": update_norm,
            "gradient_update_cosine": float(dot / denominator) if denominator > 0 else 0.0,
            "first_order_descent": float(dot),
        }
    return output


def molecule_max_scores(
    query: torch.Tensor,
    references: torch.Tensor,
    molecule_ptr: np.ndarray,
) -> torch.Tensor:
    """Maximise query-reference cosine within every candidate molecule."""
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if query.ndim != 1 or references.ndim != 2 or references.shape[1] != len(query):
        raise ValueError("query and reference embeddings are not aligned")
    if ptr.ndim != 1 or len(ptr) < 3 or ptr[0] != 0 or ptr[-1] != len(references):
        raise ValueError("molecule_ptr must span at least two candidate molecules")
    if np.any(np.diff(ptr) <= 0):
        raise ValueError("every candidate molecule requires a reference spectrum")
    pair = references @ query
    return torch.stack([
        pair[int(left):int(right)].max()
        for left, right in zip(ptr[:-1], ptr[1:])
    ])


def candidate_center(values: torch.Tensor) -> torch.Tensor:
    """Remove the unidentifiable common score offset from one candidate list."""
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("candidate centering requires a vector of length at least two")
    return values - values.mean()


def action_delta_target(
    official_clean_query: torch.Tensor,
    official_action_query: torch.Tensor,
    official_references: torch.Tensor,
    molecule_ptr: np.ndarray,
) -> torch.Tensor:
    """Frozen candidate-centred DreaMS effect of one qualified peak action."""
    clean = molecule_max_scores(
        official_clean_query.detach(), official_references.detach(), molecule_ptr,
    )
    action = molecule_max_scores(
        official_action_query.detach(), official_references.detach(), molecule_ptr,
    )
    return candidate_center(action - clean).detach()


def clean_inherits_action_delta_loss(
    live_clean_query: torch.Tensor,
    official_clean_query: torch.Tensor,
    official_action_query: torch.Tensor,
    official_references: torch.Tensor,
    molecule_ptr: np.ndarray,
    *,
    alpha: float,
    dose: float = 1.0,
    huber_beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Make the live clean query inherit a bounded frozen action effect.

    The live score displacement is evaluated against frozen official
    references.  Consequently this loss cannot be satisfied by moving the
    candidate references or by learning to recognise the modified action view.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not 0.0 <= float(dose) <= 1.0:
        raise ValueError("dose must lie in [0, 1]")
    if float(huber_beta) <= 0:
        raise ValueError("huber_beta must be positive")
    frozen_reference = official_references.detach()
    baseline = molecule_max_scores(
        official_clean_query.detach(), frozen_reference, molecule_ptr,
    )
    current = molecule_max_scores(live_clean_query, frozen_reference, molecule_ptr)
    student_delta = candidate_center(current - baseline)
    target_delta = action_delta_target(
        official_clean_query, official_action_query, frozen_reference, molecule_ptr,
    )
    target = float(alpha) * float(dose) * target_delta
    loss = F.smooth_l1_loss(student_delta, target, beta=float(huber_beta))
    return loss, {
        "student_delta": student_delta,
        "target_delta": target_delta,
        "dosed_target": target,
    }


def role_calibrated_dose(
    role_code: np.ndarray,
    margin_gain: np.ndarray,
    corrective_rank_code: int,
    corrective_margin_code: int,
) -> np.ndarray:
    """Give rank corrections full dose and calibrate margin-only actions.

    The margin-only scale is derived from the median positive margin gain of
    rank-corrective actions.  No CLI-selected role weight is introduced.
    Unsupported roles remain exact zero.
    """
    role = np.asarray(role_code, dtype=np.int64)
    gain = np.asarray(margin_gain, dtype=np.float64)
    if role.shape != gain.shape or role.ndim != 1:
        raise ValueError("role and margin gain arrays must be aligned vectors")
    rank = role == int(corrective_rank_code)
    margin = role == int(corrective_margin_code)
    positive_rank = gain[rank & (gain > 0)]
    positive_all = gain[(rank | margin) & (gain > 0)]
    calibration = positive_rank if len(positive_rank) else positive_all
    if not len(calibration):
        raise ValueError("qualified actions contain no positive margin gain")
    scale = max(float(np.median(calibration)), 1e-8)
    dose = np.zeros(len(role), dtype=np.float32)
    dose[rank] = 1.0
    dose[margin] = np.clip(gain[margin] / scale, 0.0, 1.0).astype(np.float32)
    if np.any(dose[~(rank | margin)] != 0):
        raise RuntimeError("unsupported action role received nonzero dose")
    return dose
