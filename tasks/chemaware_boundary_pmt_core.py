"""Pure primitives for ChemAware boundary paired-margin transfer (B-PMT).

The chemical action is privileged training-time information.  It contributes
only a non-negative target increment for the unmodified clean retrieval
margin.  Matched controls define an audit statistic and a separate permuted
dose arm; they are never subtracted in the optimizer objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def strict_action_advantage(
    old_margin: np.ndarray,
    correct_margin: np.ndarray,
    control_margins: np.ndarray,
) -> np.ndarray:
    """Return the weakest correct-action advantage over baseline and controls."""
    old = np.asarray(old_margin, dtype=np.float64)
    correct = np.asarray(correct_margin, dtype=np.float64)
    controls = np.asarray(control_margins, dtype=np.float64)
    if old.ndim != 1 or correct.shape != old.shape:
        raise ValueError("old and correct margins must be aligned vectors")
    if controls.ndim != 2 or controls.shape[1] != len(old) or not len(controls):
        raise ValueError("controls must be a nonempty control-by-query matrix")
    if not (
        np.all(np.isfinite(old))
        and np.all(np.isfinite(correct))
        and np.all(np.isfinite(controls))
    ):
        raise ValueError("action margins must be finite")
    contrasts = np.vstack((correct - old, correct[None, :] - controls))
    return np.maximum(np.min(contrasts, axis=0), 0.0).astype(np.float32)


def active_transfer_weights(
    old_margin: np.ndarray,
    action_advantage: np.ndarray,
    action_count: np.ndarray,
    *,
    activation_margin: float,
    advantage_cap: float,
) -> np.ndarray:
    """Gate to current boundaries and scale only strict-positive specificity."""
    margin = np.asarray(old_margin, dtype=np.float64)
    advantage = np.asarray(action_advantage, dtype=np.float64)
    count = np.asarray(action_count, dtype=np.int64)
    if advantage.shape != margin.shape or count.shape != margin.shape:
        raise ValueError("boundary transfer arrays must align")
    if not 0 <= activation_margin <= 0.1 or advantage_cap <= 0:
        raise ValueError("invalid B-PMT activation margin or advantage cap")
    active = (margin <= activation_margin) & (advantage > 0) & (count > 0)
    weight = np.zeros(len(margin), dtype=np.float32)
    weight[active] = np.minimum(advantage[active] / advantage_cap, 1.0)
    return weight


def margin_bin_formula_derangement_indices(
    old_margin: np.ndarray,
    formula: np.ndarray,
    *,
    seed: int,
    bins: int = 4,
) -> np.ndarray:
    """Return source indices for a hardness-matched formula derangement."""
    margin = np.asarray(old_margin, dtype=np.float64)
    group = np.asarray(formula).astype(str)
    if margin.ndim != 1 or group.shape != margin.shape:
        raise ValueError("derangement inputs must be aligned vectors")
    if len(margin) < 2 or bins < 1 or len(np.unique(group)) < 2:
        raise ValueError("formula derangement requires at least two groups")
    order = np.argsort(margin, kind="stable")
    hardness_bin = np.empty(len(margin), dtype=np.int16)
    for bin_id, positions in enumerate(np.array_split(order, min(bins, len(margin)))):
        hardness_bin[positions] = bin_id
    rng = np.random.default_rng(seed)
    best = None
    best_cost = None
    source = np.arange(len(margin))
    for _ in range(20_000):
        proposal = rng.permutation(source)
        if np.any(group[proposal] == group):
            continue
        cost = int(np.sum(np.abs(hardness_bin[proposal] - hardness_bin)))
        if best_cost is None or cost < best_cost:
            best, best_cost = proposal.copy(), cost
            if cost == 0:
                break
    if best is None:
        raise RuntimeError("could not construct a formula-disjoint dose derangement")
    return best.astype(np.int64)


def margin_bin_formula_derangement(
    values: np.ndarray,
    old_margin: np.ndarray,
    formula: np.ndarray,
    *,
    seed: int,
    bins: int = 4,
) -> np.ndarray:
    """Permute doses across formula groups while approximately matching hardness.

    The output has exactly the same dose multiset as the target arm.  A valid
    assignment may not source a dose from the same formula as its destination.
    The mean absolute quantile-bin displacement is minimized over deterministic
    random derangements; failure is explicit instead of silently self-matching.
    """
    dose = np.asarray(values, dtype=np.float64)
    margin = np.asarray(old_margin, dtype=np.float64)
    group = np.asarray(formula).astype(str)
    if dose.ndim != 1 or margin.shape != dose.shape or group.shape != dose.shape:
        raise ValueError("derangement inputs must be aligned vectors")
    source_index = margin_bin_formula_derangement_indices(
        margin, group, seed=seed, bins=bins
    )
    output = dose[source_index].astype(np.float32)
    if not np.allclose(np.sort(output), np.sort(dose)):
        raise RuntimeError("matched-control dose multiset changed")
    return output


def inherited_clean_margin_target(
    initial_margin: torch.Tensor,
    action_advantage: torch.Tensor,
    alpha: float,
    advantage_cap: float,
) -> torch.Tensor:
    """Freeze a clean target that inherits a bounded fraction of action gain."""
    if initial_margin.shape != action_advantage.shape:
        raise ValueError("initial margin and action advantage must align")
    if alpha not in (0.0, 0.25, 0.5) or advantage_cap <= 0:
        raise ValueError("B-PMT only admits preregistered alpha values")
    return initial_margin.detach() + float(alpha) * torch.clamp(
        action_advantage.detach(), min=0.0, max=float(advantage_cap)
    )


def active_margin_transfer_loss(
    current_margin: torch.Tensor,
    target_margin: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Formula weighting is applied outside; zero corrective weights stay zero."""
    if current_margin.shape != target_margin.shape or sample_weight.shape != current_margin.shape:
        raise ValueError("B-PMT loss tensors must align")
    if temperature <= 0 or torch.any(sample_weight < 0):
        raise ValueError("temperature and sample weights are invalid")
    denominator = torch.sum(sample_weight)
    if float(denominator.detach()) <= 0:
        return torch.sum(current_margin * 0.0)
    each = F.softplus((target_margin - current_margin) / float(temperature))
    return torch.sum(each * sample_weight) / denominator
