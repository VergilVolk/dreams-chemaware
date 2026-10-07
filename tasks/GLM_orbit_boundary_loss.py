"""Orbit-boundary unified loss (pure logic, locally testable).

Implements section 3.3-3.4 of UNIFIED_ALGORITHM_POST_GATE1_ADVERSARIAL_AUDIT:

  A. L_rank      = CE( S_theta(q, candidates)/T, true_molecule )
     with S_theta = tau * (logsumexp_{r in R(c)} cos(z_q, z_r)/tau - log|R(c)|)
  B. L_orbit     = symmetric KL between candidate distributions of two
     same-molecule condition views + hinge keeping the true candidate above
     hard negatives in BOTH views
  C. L_boundary  = CE( (S_theta + delta_chem)/T, true_molecule ), where
     delta_chem is a per-candidate margin payload (real chemistry arm) or a
     matched candidate-rotated null payload (null arm)
  D. Protection constraints (constrained optimization):
     L_noise_replay(theta) <= L_noise_replay(theta_N) + eps_N
     implemented as an additive penalty max(0, L - L_ref - eps) per protected
     quantity (noise replay, global replay, near replay); the caller passes
     the replay losses of the current and reference encoders.

Arm factorization: two booleans (use_orbit, use_boundary) select which
terms are active; arms differ ONLY through these booleans and their payload
arrays (per the audit contract - no per-arm code forks).

Discipline fixed a priori (documented, not tunable post hoc):
  temperature T for CE logits, tau for the reference logsumexp, and the
  orbit hinge margin are constructor arguments frozen at build time and
  written into every run report.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OrbitBoundaryConfig:
    tau: float = 0.05          # reference logsumexp temperature
    temperature: float = 1.0   # CE logits temperature (T)
    orbit_hinge_margin: float = 0.1
    lambda_orbit: float = 1.0
    lambda_boundary: float = 1.0
    eps_noise: float = 0.0     # protection tolerances
    eps_global: float = 0.0
    eps_near: float = 0.0

    def as_dict(self) -> dict:
        return dict(tau=self.tau, temperature=self.temperature,
                    orbit_hinge_margin=self.orbit_hinge_margin,
                    lambda_orbit=self.lambda_orbit,
                    lambda_boundary=self.lambda_boundary,
                    eps_noise=self.eps_noise, eps_global=self.eps_global,
                    eps_near=self.eps_near)


def molecule_score(z_q: torch.Tensor, z_refs: torch.Tensor,
                   ref_ptr: torch.Tensor, tau: float) -> torch.Tensor:
    """S_theta for a single query against padded candidate molecules.

    z_q: (D,) query embedding (L2-normalized).
    z_refs: (R_total, D) stacked reference embeddings of all candidates.
    ref_ptr: (C+1,) boundaries of each candidate's reference block.
    Returns (C,) molecule scores: tau*(logsumexp_r(cos/tau) - log|R(c)|).
    """
    cos = z_refs @ z_q                                   # (R_total,)
    C = ref_ptr.shape[0] - 1
    counts = (ref_ptr[1:] - ref_ptr[:-1]).clamp(min=1)
    scores = torch.empty(C, dtype=z_q.dtype)
    for c in range(C):
        block = cos[ref_ptr[c]:ref_ptr[c + 1]] / tau
        scores[c] = tau * (torch.logsumexp(block, dim=0)
                           - torch.log(counts[c].to(z_q.dtype)))
    return scores


def listwise_loss(scores: torch.Tensor, true_idx: int,
                  delta: torch.Tensor | None,
                  temperature: float) -> torch.Tensor:
    logits = scores / temperature
    if delta is not None:
        logits = logits + delta / temperature
    target = torch.full_like(logits, -1e4)
    target[true_idx] = 1e4
    # robust CE independent of target encoding
    logp = F.log_softmax(logits, dim=0)
    return -logp[true_idx]


def orbit_consistency(scores_a: torch.Tensor, scores_b: torch.Tensor,
                      true_idx: int, cfg: OrbitBoundaryConfig
                      ) -> torch.Tensor:
    """Symmetric KL between two condition views + shared-true hinge."""
    pa = F.log_softmax(scores_a / cfg.temperature, dim=0)
    pb = F.log_softmax(scores_b / cfg.temperature, dim=0)
    kl = 0.5 * (F.kl_div(pa, pb, log_target=True, reduction="sum")
                + F.kl_div(pb, pa, log_target=True, reduction="sum"))
    # hinge: true candidate must stay above every hard negative by margin
    others = torch.cat([scores_a[:true_idx], scores_a[true_idx + 1:]])
    hinge = F.relu(cfg.orbit_hinge_margin - (scores_a[true_idx] - others)).sum()
    others_b = torch.cat([scores_b[:true_idx], scores_b[true_idx + 1:]])
    hinge = hinge + F.relu(
        cfg.orbit_hinge_margin - (scores_b[true_idx] - others_b)).sum()
    return kl + hinge


def protection_penalty(current: float, reference: float,
                       eps: float) -> torch.Tensor:
    """max(0, current - reference - eps) as a tensor penalty."""
    return F.relu(torch.tensor(float(current - reference - eps)))


class OrbitBoundaryLoss(torch.nn.Module):
    """Unified loss; arms differ only by use_orbit/use_boundary flags."""

    def __init__(self, cfg: OrbitBoundaryConfig, use_orbit: bool,
                 use_boundary: bool):
        super().__init__()
        self.cfg = cfg
        self.use_orbit = use_orbit
        self.use_boundary = use_boundary

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        scores = molecule_score(batch["z_q"], batch["z_refs"],
                                batch["ref_ptr"], cfg.tau)
        delta = batch.get("delta_chem") if self.use_boundary else None
        l_rank = listwise_loss(scores, batch["true_idx"], delta,
                               cfg.temperature)
        parts = {"loss_rank": l_rank}
        total = l_rank
        if self.use_orbit:
            scores_o = molecule_score(batch["z_q_orbit"], batch["z_refs"],
                                      batch["ref_ptr"], cfg.tau)
            l_orbit = orbit_consistency(scores, scores_o, batch["true_idx"],
                                        cfg)
            parts["loss_orbit"] = l_orbit
            total = total + cfg.lambda_orbit * l_orbit
        if self.use_boundary:
            l_bound = listwise_loss(scores, batch["true_idx"],
                                    batch["delta_chem"], cfg.temperature)
            parts["loss_boundary_dup_check"] = l_bound  # equals l_rank+delta
        pen = 0.0
        for key, eps in (("replay_noise", cfg.eps_noise),
                         ("replay_global", cfg.eps_global),
                         ("replay_near", cfg.eps_near)):
            if key in batch:
                p = protection_penalty(float(batch[key][0]),
                                       float(batch[key][1]), eps)
                pen = pen + p
        parts["protection"] = torch.as_tensor(pen)
        total = total + pen
        parts["loss"] = total
        return parts
