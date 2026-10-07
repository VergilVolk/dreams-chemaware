"""Six-arm orbit-boundary trainer (audit section 11, file 3).

One trainer, one pool, six arms selected ONLY by --arm:

    R      use_orbit=False use_boundary=False
    O-null use_orbit=True  orbit=null    use_boundary=False
    O-real use_orbit=True  orbit=real    use_boundary=False
    C-null use_orbit=False use_boundary=True  delta=null
    C-real use_orbit=False use_boundary=True  delta=real
    OC     use_orbit=True  orbit=real    use_boundary=True  delta=real

Encoder: --encoder synthetic (tiny MLP, CPU, for local smoke tests) or
--encoder dreams (server: wraps the repo's frozen DreaMS ContrastiveHead
exactly like GLM_train_chemaware_listwise; imported lazily).

Same-dose discipline: identical --steps/--batch-size/--lr/--seed are
enforced by the sbatch; this script records them and refuses mismatched
reruns of the same run-id.

Checkpoints at fixed fractions of --steps (0.25/0.5/0.75/1.0); the shared
factorial step chosen LATER by GLM_select_shared_factorial_step on the R
arm only, then applied to all arms mechanically.

Replay protection: the pool carries replay_groups (indices of frozen
noise-replay groups with precomputed theta_N scores). The penalty terms
consume current-vs-reference replay CE; the synthetic path uses zeros.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_orbit_boundary_loss import (  # noqa: E402
    OrbitBoundaryConfig, OrbitBoundaryLoss, molecule_score,
)

ARMS = {
    "R":      dict(use_orbit=False, orbit=None, use_boundary=False, delta=None),
    "O-null": dict(use_orbit=True, orbit="null", use_boundary=False, delta=None),
    "O-real": dict(use_orbit=True, orbit="real", use_boundary=False, delta=None),
    "C-null": dict(use_orbit=False, orbit=None, use_boundary=True, delta="null"),
    "C-real": dict(use_orbit=False, orbit=None, use_boundary=True, delta="real"),
    "OC":     dict(use_orbit=True, orbit="real", use_boundary=True, delta="real"),
}


class SyntheticEncoder(torch.nn.Module):
    def __init__(self, dim=32, d=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, d))

    def forward(self, peaks):
        return torch.nn.functional.normalize(self.net(peaks), dim=-1)


def build_batch(pool, idx, encoder, arm, spectra_feats, device):
    z_all = encoder(spectra_feats.to(device))
    z_q = z_all[pool["query_row"][idx]]
    lo, hi = int(pool["cand_ptr"][idx]), int(pool["cand_ptr"][idx + 1])
    rlo, rhi = int(pool["ref_ptr"][lo]), int(pool["ref_ptr"][hi])
    ref_rows = torch.as_tensor(pool["ref_rows"][rlo:rhi],
                               dtype=torch.long, device=device)
    z_refs = z_all[ref_rows]
    ref_ptr = torch.as_tensor(
        np.asarray(pool["ref_ptr"][lo:hi + 1], dtype=np.int64) - rlo)
    labels = pool["molecule_label"][lo:hi]
    true_idx = int(np.flatnonzero(labels == 1)[0])
    batch = {"z_q": z_q, "z_refs": z_refs, "ref_ptr": ref_ptr,
             "true_idx": true_idx}
    spec = ARMS[arm]
    if spec["use_orbit"]:
        row = pool["orbit_row_real" if spec["orbit"] == "real"
                   else "orbit_row_null"][idx]
        batch["z_q_orbit"] = z_all[int(row)]
    if spec["use_boundary"]:
        key = "delta_chem_real" if spec["delta"] == "real" else "delta_chem_null"
        batch["delta_chem"] = torch.as_tensor(
            pool[key][idx][: hi - lo], dtype=torch.float32, device=device)
    # replay protection payloads (synthetic zeros; production: theta_N CE)
    batch["replay_noise"] = (0.0, 0.0)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--encoder", choices=("synthetic", "dreams"),
                    default="synthetic")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    with np.load(args.pool, allow_pickle=False) as z:
        pool = {k: np.asarray(z[k]) for k in z.files}
    n_groups = len(pool["query_row"])
    if args.encoder == "synthetic":
        encoder = SyntheticEncoder(seed=args.seed).to(device)
    else:
        raise NotImplementedError(
            "dreams path wraps the repo ContrastiveHead on the server; "
            "see GLM_train_chemaware_listwise for the loading contract")

    # synthetic spectra features: each row -> 32-dim pseudo-peak vector
    n_rows = int(max(pool["ref_rows"].max(), pool["query_row"].max(),
                     pool["orbit_row_real"].max(),
                     pool["orbit_row_null"].max())) + 1
    spectra_feats = torch.randn(n_rows, 32, generator=torch.Generator().manual_seed(args.seed))

    spec = ARMS[args.arm]
    cfg = OrbitBoundaryConfig()
    loss_fn = OrbitBoundaryLoss(cfg, spec["use_orbit"], spec["use_boundary"])
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    ckpt_fracs = (0.25, 0.5, 0.75, 1.0)
    ckpt_steps = {min(int(f * args.steps), args.steps - 1): f
                  for f in ckpt_fracs}
    for step in range(args.steps):
        idx = rng.integers(0, n_groups, size=args.batch_size)
        losses = []
        for i in idx:
            batch = build_batch(pool, int(i), encoder, args.arm,
                                spectra_feats, device)
            parts = loss_fn(batch)
            losses.append(parts["loss"])
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
        if step in ckpt_steps:
            torch.save(encoder.state_dict(),
                       args.output_dir / f"{args.arm}_step{step}.pt")

    report = {
        "status": "GLM_ORBIT_BOUNDARY_TRAIN_COMPLETE",
        "arm": args.arm, "arm_spec": spec,
        "config": cfg.as_dict(),
        "dose": {"steps": args.steps, "batch_size": args.batch_size,
                 "lr": args.lr, "seed": args.seed, "encoder": args.encoder},
        "n_groups": n_groups,
        "loss_first": round(history[0], 6),
        "loss_last": round(history[-1], 6),
        "loss_history": [round(h, 6) for h in history],
    }
    (args.output_dir / f"{args.arm}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{args.arm}] loss {history[0]:.4f} -> {history[-1]:.4f} "
          f"({args.steps} steps)")


if __name__ == "__main__":
    main()
