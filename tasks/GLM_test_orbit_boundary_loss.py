"""Synthetic unit tests for GLM_orbit_boundary_loss (local, CPU-only).

Verifies the audit contract's core properties:
  T1 molecule_score is reference-count corrected: a candidate with many
     references does NOT win just by having more chances (unlike raw max).
  T2 listwise CE + delta reduces to plain CE when delta = 0.
  T3 orbit consistency = 0 for identical views, > 0 for shifted views.
  T4 arm factorization: R arm ignores both payloads; O/C arms differ only
     through the booleans.
  T5 protection penalty exact-zero when within tolerance, positive beyond.
  T6 gradients flow to embeddings (backward smoke).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_orbit_boundary_loss import (  # noqa: E402
    OrbitBoundaryConfig, OrbitBoundaryLoss, listwise_loss,
    molecule_score, orbit_consistency, protection_penalty,
)
torch.manual_seed(7)
cfg = OrbitBoundaryConfig()


def make_case(n_cand=4, refs=(2, 5, 2, 2), d=16, true_idx=1):
    z_refs = []
    ptr = [0]
    for c, n in enumerate(refs):
        base = torch.randn(d)
        for _ in range(n):
            z_refs.append(base + 0.05 * torch.randn(d))
        ptr.append(ptr[-1] + n)
    z_refs = torch.stack(z_refs)
    z_refs = z_refs / z_refs.norm(dim=1, keepdim=True)
    # query = noisy view of the TRUE candidate's reference cluster
    z_q = z_refs[ptr[true_idx]] + 0.3 * torch.randn(d)
    z_q = z_q / z_q.norm()
    return z_q, z_refs, torch.tensor(ptr), true_idx


def main() -> None:
    z_q, z_refs, ptr, ti = make_case()
    scores = molecule_score(z_q, z_refs, ptr, cfg.tau)
    assert scores.shape == (4,)

    # T1: reference-count correction - duplicating a candidate's ENTIRE
    # reference multiset leaves S_theta EXACTLY invariant (log|R| cancels
    # the logsumexp shift), while adding a low-scoring reference decreases
    # S_theta (no reward for extra chances) - unlike plain max which never
    # decreases with more references.
    dup = torch.cat([z_refs[ptr[2]:ptr[3]], z_refs[ptr[2]:ptr[3]]])
    z_refs2 = torch.cat([z_refs[ptr[0]:ptr[2]], dup, z_refs[ptr[3]:]])
    ptr2 = ptr.clone(); ptr2[3:] += (ptr[3] - ptr[2])
    scores2 = molecule_score(z_q, z_refs2, ptr2, cfg.tau)
    assert abs(scores2[2] - scores[2]) < 1e-5, "count correction failed"
    low_ref = (z_refs[ptr[2]] * 0.1).unsqueeze(0)
    low_ref = low_ref / low_ref.norm()
    z_refs3 = torch.cat([z_refs[ptr[0]:ptr[3]], low_ref,
                         z_refs[ptr[3]:]])
    ptr3 = ptr.clone(); ptr3[3:] += 1
    scores3 = molecule_score(z_q, z_refs3, ptr3, cfg.tau)
    assert scores3[2] < scores[2], "extra low ref must not raise score"
    print("T1 PASS: multiset duplication invariant; extra low ref lowers "
          "S_theta (count-corrected)")

    # T2
    ce0 = listwise_loss(scores, ti, None, cfg.temperature)
    ce0b = listwise_loss(scores, ti, torch.zeros_like(scores),
                         cfg.temperature)
    assert torch.allclose(ce0, ce0b, atol=1e-6)
    print("T2 PASS: zero delta == plain CE")

    # T3
    same = orbit_consistency(scores, scores.clone(), ti, cfg)
    assert float(same) < 1e-6, "identical views must give ~0"
    shifted_scores = scores.clone()
    shifted_scores[0] += 1.0          # real distribution distortion
    shifted_scores[ti] -= 0.3
    shifted = orbit_consistency(scores, shifted_scores, ti, cfg)
    assert float(shifted) > 0.1, "distorted view must give substantial loss"
    print(f"T3 PASS: orbit 0-for-identical ({float(same):.2e}), "
          f"substantial-for-distorted ({float(shifted):.3f})")

    # T4 arm factorization on a full batch
    batch = {
        "z_q": z_q, "z_refs": z_refs, "ref_ptr": ptr, "true_idx": ti,
        "z_q_orbit": z_refs[ptr[ti]] + 0.05 * torch.randn(16),
        "delta_chem": torch.tensor([0.0, 0.0, 0.5, 0.0]),
    }
    batch["z_q_orbit"] = batch["z_q_orbit"] / batch["z_q_orbit"].norm()
    arm_R = OrbitBoundaryLoss(cfg, use_orbit=False, use_boundary=False)
    arm_O = OrbitBoundaryLoss(cfg, use_orbit=True, use_boundary=False)
    arm_C = OrbitBoundaryLoss(cfg, use_orbit=False, use_boundary=True)
    arm_OC = OrbitBoundaryLoss(cfg, use_orbit=True, use_boundary=True)
    # R ignores payloads: changing delta/orbit must not change R loss
    b2 = dict(batch)
    b2["delta_chem"] = torch.tensor([9.0, 9.0, 9.0, 9.0])
    b2["z_q_orbit"] = torch.randn(16)
    out_R = arm_R(batch)
    out_R2 = arm_R(b2)
    out_O = arm_O(batch)
    out_C = arm_C(batch)
    out_OC = arm_OC(batch)
    assert torch.allclose(out_R["loss"], out_R["loss_rank"])
    assert "loss_orbit" in out_O
    # C arm: nonzero delta on a FALSE candidate must raise the CE
    assert float(out_C["loss_rank"]) > float(out_R["loss_rank"]) + 1e-6
    assert torch.allclose(out_R["loss"], out_R2["loss"])
    print("T4 PASS: arms differ only through their booleans; R ignores "
          "payloads; C-arm CE responds to margins")

    # T5 differentiable protection: gradient flows through `current`
    cur = torch.tensor(1.5, requires_grad=True)
    ref = torch.tensor(1.0)
    p = protection_penalty(cur, ref, 0.2)
    assert float(p) > 0.0
    p.backward()
    assert cur.grad is not None and float(cur.grad) == 1.0
    inside = protection_penalty(torch.tensor(1.0), torch.tensor(1.0), 0.0)
    assert float(inside) == 0.0
    print("T5 PASS: protection penalty differentiable (grad=1 beyond "
          "tolerance), exact-zero within")

    # T6 gradient flow
    z = z_q.clone().requires_grad_(True)
    s = molecule_score(z, z_refs, ptr, cfg.tau)
    loss = listwise_loss(s, ti, None, cfg.temperature)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    print("T6 PASS: gradients flow")

    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
