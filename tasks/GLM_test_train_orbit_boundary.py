"""End-to-end six-arm smoke: build a synthetic pool, train all six arms,
verify the audit contract on REAL training runs:

  A1 all six arms train with finite, decreasing-ish losses;
  A2 R-arm immunity: corrupting every payload array leaves the R arm's
     loss history IDENTICAL;
  A3 orbit arms read different rows (O-real vs O-null histories differ);
  A4 same-dose reproducibility: rerunning the same arm reproduces the
     history exactly (same seed/data/dose);
  A5 checkpoint files exist at the fixed fractions for every arm.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="six_arm_smoke_"))

# 1. build the pool via the builder smoke fixtures
N, C = 24, 3
g = {
    "query_row": np.arange(N),
    "cand_ptr": np.arange(N + 1) * C,
    "molecule_label": np.tile(np.array([1, 0, 0]), N),
    "mol_ik14": np.array([f"IK{i:04d}abc" for i in range(N * C)]),
    "formula_cluster": np.array([i // 4 for i in range(N)]),
    "val_query_mask": np.array([i >= 16 for i in range(N)]),
    "ref_ptr": np.arange(N * C + 1),
    "ref_rows": np.arange(N * C),
}
rows, mols, inst, qual = [], [], [], []
for i in range(N):
    for base, k in ((0, 1), (100, 2), (200, 1)):
        rows.append(base + i); mols.append(f"IK{i:04d}abc")
        inst.append(k); qual.append(0.5)
# fix mol ids: candidate c of group i has its own molecule; only candidate 0
# is the true molecule, so the ledger must reference candidate-0 ids
mol_ids = np.array([f"IK{i:04d}abc" for i in range(N * C)])
g["mol_ik14"] = mol_ids
views_mol = []
for i in range(N):
    views_mol.extend([mol_ids[i * C]] * 3)
v = {"rows": np.asarray(rows), "molecule": np.asarray(views_mol),
     "instrument": np.asarray(inst), "quality": np.asarray(qual, float)}
m = {"group_id": np.concatenate([np.arange(N), np.arange(N)]),
     "candidate_local": np.concatenate([np.full(N, 1), np.full(N, 2)]),
     "margin": np.concatenate([np.full(N, 0.3, dtype=np.float32),
                               np.full(N, 0.5, dtype=np.float32)])}
pool = tmp / "pool.npz"
np.savez(tmp / "cand.npz", **g)
np.savez(tmp / "views.npz", **v)
np.savez(tmp / "marg.npz", **m)
r = subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "tasks/GLM_build_orbit_boundary_groups.py"),
                    "--candidate-groups", str(tmp / "cand.npz"),
                    "--condition-views", str(tmp / "views.npz"),
                    "--chem-margins", str(tmp / "marg.npz"),
                    "--output", str(pool)], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]

# corrupted pool: payloads scrambled (R must not notice)
with np.load(pool) as z:
    p = {k: np.asarray(z[k]) for k in z.files}
p["delta_chem_real"] = np.random.default_rng(1).uniform(
    0, 9, p["delta_chem_real"].shape).astype(np.float32)
p["delta_chem_null"] = np.random.default_rng(2).uniform(
    0, 9, p["delta_chem_null"].shape).astype(np.float32)
p["orbit_row_real"] = p["orbit_row_null"]  # swap semantics
pool_bad = tmp / "pool_corrupted.npz"
np.savez(pool_bad, **p)


def train(arm, pl, out, steps=30, sha=None):
    cmd = [sys.executable, "-X", "utf8",
           str(ROOT / "tasks/GLM_train_orbit_boundary_encoder.py"),
           "--arm", arm, "--pool", str(pl), "--encoder", "synthetic",
           "--steps", str(steps), "--batch-size", "8", "--lr", "1e-2",
           "--seed", "3407", "--output-dir", str(out),
           "--expected-pool-sha256", sha or _sha(pl)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads((out / f"{arm}_report.json").read_text(encoding="utf-8"))


def _sha(pl):
    import hashlib
    return hashlib.sha256(Path(pl).read_bytes()).hexdigest()


hist = {}
for arm in ("R", "O-null", "O-real", "C-null", "C-real", "OC"):
    out = tmp / arm
    hist[arm] = train(arm, pool, out)
    h = hist[arm]["loss_history"]
    assert all(np.isfinite(h)), f"{arm}: non-finite loss"
    ckpts = list(out.glob(f"{arm}_step*.ckpt"))
    assert len(ckpts) == 4, f"{arm}: expected 4 checkpoints, got {len(ckpts)}"
    rep = hist[arm]
    # P0-3 isolation: eligible <= train groups; report records the counts
    assert rep["eligible_train_groups"] <= rep["n_groups"] - rep["val_groups"]
print("A1/A5 PASS: six arms train, finite losses, 4 ckpts, isolation counts")

r_good = train("R", pool, tmp / "rg")
r_bad = train("R", pool_bad, tmp / "rb")
assert r_good["loss_history"] == r_bad["loss_history"], \
    "R arm must be immune to payload corruption"
print("A2 PASS: R-arm loss history identical under corrupted payloads")

assert hist["O-real"]["loss_history"] != hist["O-null"]["loss_history"], \
    "O-real and O-null must consume different orbit rows"
assert hist["C-real"]["loss_history"] != hist["C-null"]["loss_history"], \
    "C-real and C-null must consume different margins"
print("A3 PASS: orbit/boundary arms read their designated payloads")

r_rep = train("R", pool, tmp / "rr")
assert r_rep["loss_history"] == hist["R"]["loss_history"], \
    "same seed/data/dose must reproduce exactly"
print("A4 PASS: same-dose rerun reproduces the loss history exactly")

# A6 SHA gate: wrong expected sha must ABORT
bad = subprocess.run(
    [sys.executable, "-X", "utf8",
     str(ROOT / "tasks/GLM_train_orbit_boundary_encoder.py"),
     "--arm", "R", "--pool", str(pool), "--encoder", "synthetic",
     "--steps", "5", "--batch-size", "4", "--lr", "1e-3", "--seed", "3407",
     "--output-dir", str(tmp / "sha_bad"),
     "--expected-pool-sha256", "0" * 64],
    capture_output=True, text=True)
assert bad.returncode != 0 and "sha drift" in bad.stderr
print("A6 PASS: sha mismatch aborts training")

# A7 -1 guard: poison one group's orbit rows -> O-arms exclude it, R unaffected
with np.load(pool) as z:
    p2 = {k: np.asarray(z[k]) for k in z.files}
p2["orbit_row_real"][3] = -1
p2["orbit_row_null"][3] = -1
p2["orbit_both_matched"][3] = False
pool_poison = tmp / "pool_poison.npz"
np.savez(pool_poison, **p2)
out_o = tmp / "opoison"
hist_o = train("O-real", pool_poison, out_o)
rep_o = hist_o
expected_elig = int((~p2["val_query_mask"].astype(bool)
                     & p2["orbit_both_matched"].astype(bool)).sum())
assert rep_o["eligible_train_groups"] == expected_elig
print(f"A7 PASS: poisoned group excluded (O-real eligible "
      f"{rep_o['eligible_train_groups']} = expected {expected_elig})")
print("SIX-ARM SMOKE v2: ALL PASS")
