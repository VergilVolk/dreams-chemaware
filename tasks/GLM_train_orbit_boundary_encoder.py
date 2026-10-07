"""Six-arm orbit-boundary trainer v2 (post adversarial audit 2026-10-07).

Fixes over v1, in the audit's priority order:
  P0-1  --encoder dreams now implements the REAL loading path
          (load_base_model / preprocess_spectrum / native ContrastiveHead,
          mirroring GLM_train_chemaware_listwise); synthetic remains a
          clearly-bannedered logic-test mode.
  P0-2  protection penalties are differentiable: replay CE computed with
          the LIVE encoder vs a frozen theta_N reference constant.
  P0-3  strict train/val isolation: sampling only from ~val_query_mask AND
          the arm's matched denominator (O-arms: orbit_both_matched;
          C-arms: ~chem_null_trivial; R: all train groups); formula-cluster
          disjointness re-asserted at start.
  P0-4  groups lacking the required orbit row are excluded by the same
          eligibility mask; -1 can never reach an index (hard assert).
  P1-6  --expected-pool-sha256 fail-closed; observed hash in every report.

Still NOT done (server): real 2-group GPU forward/backward smoke, role-2
curve extraction, role-3 confirmation, GNPS chain. The sbatch stays
DO-NOT-SUBMIT until those pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_orbit_boundary_loss import (  # noqa: E402
    OrbitBoundaryConfig, OrbitBoundaryLoss, molecule_score, listwise_loss,
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
    """LOGIC-TEST ONLY. Never used by the dreams path."""

    def __init__(self, dim=32, d=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, d))

    def forward(self, peaks):
        return torch.nn.functional.normalize(self.net(peaks), dim=-1)


def load_dreams_encoder(args):
    """Real path: native DreaMS backbone from the frozen official +
    architecture checkpoints (same contract as GLM_train_chemaware_listwise).
    Noise V1 start = official architecture checkpoint replaced by the
    Noise V1 weights file; the hash is pinned by the caller's sbatch."""
    from dreams.models.heads.heads import ContrastiveHead  # noqa: PLC0415
    from train_e1_identity import load_base_model  # noqa: PLC0415
    initialized, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device("cpu"), args.n_highest_peaks)
    state = torch.load(args.noise_v1_checkpoint, map_location="cpu")
    initialized.backbone.load_state_dict(state["backbone"]
                                         if "backbone" in state else state)
    head = ContrastiveHead(initialized.backbone, args.lr, 0.0,
                           triplet_loss_margin=0.1)
    head.head.load_state_dict(initialized.head.state_dict(), strict=True)
    return head


def load_real_spectra(args, pool) -> torch.Tensor:
    """Real spectra input pipeline (P0-1 completion): preprocess every row
    referenced by the pool exactly like the proven listwise trainer
    (h5py row -> preprocess_spectrum -> fixed-shape tensor), stacked once.
    """
    import h5py  # noqa: PLC0415
    from train_e1_identity import preprocess_spectrum  # noqa: PLC0415
    rows_needed = np.unique(np.concatenate((
        np.asarray(pool["query_row"], dtype=np.int64),
        np.asarray(pool["ref_rows"], dtype=np.int64),
        np.asarray(pool["orbit_row_real"], dtype=np.int64),
        np.asarray(pool["orbit_row_null"], dtype=np.int64))))
    rows_needed = rows_needed[rows_needed >= 0]
    feats: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        total = int(len(handle["spectrum"]))
        if np.any((rows_needed < 0) | (rows_needed >= total)):
            raise RuntimeError("pool references an out-of-range HDF5 row")
        for row in rows_needed:
            feats[int(row)] = preprocess_spectrum(
                np.asarray(handle["spectrum"][int(row)]),
                float(handle["precursor_mz"][int(row)]),
                args.n_highest_peaks)
    n_rows = int(rows_needed.max()) + 1
    first = next(iter(feats.values()))
    out = torch.zeros(n_rows, *first.shape, dtype=first.dtype)
    for row, t in feats.items():
        out[row] = t
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible_indices(pool, arm):
    val = pool["val_query_mask"].astype(bool)
    spec = ARMS[arm]
    mask = ~val
    if spec["use_orbit"]:
        mask = mask & pool["orbit_both_matched"].astype(bool)
    if spec["use_boundary"]:
        mask = mask & ~pool["chem_null_trivial"].astype(bool)
    return np.flatnonzero(mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--expected-pool-sha256", required=True)
    ap.add_argument("--encoder", choices=("synthetic", "dreams"),
                    default="synthetic")
    ap.add_argument("--official-checkpoint", type=Path)
    ap.add_argument("--architecture-checkpoint", type=Path)
    ap.add_argument("--noise-v1-checkpoint", type=Path)
    ap.add_argument("--data", type=Path,
                    help="real spectra HDF5 (required for --encoder dreams)")
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--replay-groups", type=int, default=4,
                    help="groups reused as the noise-replay panel")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    observed = sha256_of(args.pool)
    if observed != args.expected_pool_sha256:
        raise RuntimeError(
            f"pool sha drift: expected={args.expected_pool_sha256} "
            f"observed={observed}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    with np.load(args.pool, allow_pickle=False) as z:
        pool = {k: np.asarray(z[k]) for k in z.files}
    for key in ("val_query_mask", "orbit_both_matched", "chem_null_trivial",
                "orbit_row_real", "orbit_row_null", "delta_chem_real",
                "delta_chem_null", "formula_cluster"):
        if key not in pool:
            raise RuntimeError(f"pool lacks required array: {key}")

    # formula-cluster disjointness re-asserted (P0-3)
    val = pool["val_query_mask"].astype(bool)
    tr_cl = set(pool["formula_cluster"][~val].tolist())
    va_cl = set(pool["formula_cluster"][val].tolist())
    if tr_cl & va_cl:
        raise RuntimeError("train/validation formula clusters overlap")

    if args.encoder == "synthetic":
        print("*** LOGIC-TEST MODE (synthetic MLP): not a training result ***")
        encoder = SyntheticEncoder(seed=args.seed).to(device)
        n_rows = int(max(pool["ref_rows"].max(), pool["query_row"].max(),
                         pool["orbit_row_real"].max(),
                         pool["orbit_row_null"].max()) + 1)
        spectra_feats = torch.randn(
            n_rows, 32, generator=torch.Generator().manual_seed(args.seed))
    else:
        for req in (args.official_checkpoint, args.architecture_checkpoint,
                    args.noise_v1_checkpoint, args.data):
            if req is None:
                raise RuntimeError("dreams path requires all checkpoints "
                                   "and --data (real spectra hdf5)")
        encoder = load_dreams_encoder(args).to(device)
        spectra_feats = load_real_spectra(args, pool).to(device)

    eligible = eligible_indices(pool, args.arm)
    if len(eligible) < args.batch_size:
        raise RuntimeError(f"arm {args.arm}: only {len(eligible)} eligible "
                           f"train groups < batch {args.batch_size}")

    # replay reference: frozen theta (init) CE on a fixed replay panel
    replay_ids = eligible[: args.replay_groups]

    spec = ARMS[args.arm]
    cfg = OrbitBoundaryConfig()
    loss_fn = OrbitBoundaryLoss(cfg, spec["use_orbit"], spec["use_boundary"])
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    ckpt_steps = {min(int(f * args.steps), args.steps - 1): f
                  for f in (0.25, 0.5, 0.75, 1.0)}

    def group_batch(i, z_all):
        z_q = z_all[pool["query_row"][i]]
        lo, hi = int(pool["cand_ptr"][i]), int(pool["cand_ptr"][i + 1])
        rlo, rhi = int(pool["ref_ptr"][lo]), int(pool["ref_ptr"][hi])
        z_refs = z_all[torch.as_tensor(pool["ref_rows"][rlo:rhi],
                                       dtype=torch.long)]
        ref_ptr = torch.as_tensor(
            np.asarray(pool["ref_ptr"][lo:hi + 1], np.int64) - rlo)
        true_idx = int(np.flatnonzero(pool["molecule_label"][lo:hi])[0])
        batch = {"z_q": z_q, "z_refs": z_refs, "ref_ptr": ref_ptr,
                 "true_idx": true_idx}
        if spec["use_orbit"]:
            row = int(pool["orbit_row_real" if spec["orbit"] == "real"
                           else "orbit_row_null"][i])
            assert row >= 0, "eligibility mask must prevent -1 indexing"
            batch["z_q_orbit"] = z_all[row]
        if spec["use_boundary"]:
            key = ("delta_chem_real" if spec["delta"] == "real"
                   else "delta_chem_null")
            batch["delta_chem"] = torch.as_tensor(
                pool[key][i][: hi - lo], dtype=torch.float32)
        return batch

    def replay_ce(z_all):
        vals = []
        for i in replay_ids:
            b = group_batch(i, z_all)
            vals.append(listwise_loss(molecule_score(
                b["z_q"], b["z_refs"], b["ref_ptr"], cfg.tau),
                b["true_idx"], None, cfg.temperature))
        return torch.stack(vals).mean()

    with torch.no_grad():
        ref_replay = float(replay_ce(encoder(spectra_feats)))

    for step in range(args.steps):
        idx = rng.choice(eligible, size=args.batch_size, replace=False)
        z_all = encoder(spectra_feats)          # ONE encode per step
        cur_replay = replay_ce(z_all)
        losses = []
        for i in idx:
            batch = group_batch(int(i), z_all)
            batch["replay_noise"] = (cur_replay,
                                     torch.tensor(ref_replay))
            parts = loss_fn(batch)
            losses.append(parts["loss"])
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
        if step in ckpt_steps:
            torch.save({"encoder": encoder.state_dict(), "step": step},
                       args.output_dir / f"{args.arm}_step{step:06d}.ckpt")

    report = {
        "status": "GLM_ORBIT_BOUNDARY_TRAIN_COMPLETE_V2",
        "arm": args.arm, "arm_spec": spec,
        "encoder": args.encoder,
        "pool_sha256": observed,
        "config": cfg.as_dict(),
        "dose": {"steps": args.steps, "batch_size": args.batch_size,
                 "lr": args.lr, "seed": args.seed},
        "n_groups": len(pool["query_row"]),
        "eligible_train_groups": int(len(eligible)),
        "val_groups": int(val.sum()),
        "replay_reference_ce": round(ref_replay, 6),
        "loss_first": round(history[0], 6),
        "loss_last": round(history[-1], 6),
        "loss_history": [round(h, 6) for h in history],
    }
    (args.output_dir / f"{args.arm}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{args.arm}] eligible {len(eligible)} loss {history[0]:.4f} -> "
          f"{history[-1]:.4f} ({args.steps} steps, sha pinned)")


if __name__ == "__main__":
    main()
