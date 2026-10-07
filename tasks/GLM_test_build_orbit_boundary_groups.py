"""Synthetic smoke test for GLM_build_orbit_boundary_groups.

Fabricates minimal candidate groups, a multi-condition ledger, and a
chemical-margin ledger; runs the builder; verifies the output contract and
integrity flags (I1 true-molecule orbit, I2 condition semantics, I3
rotation multiset, I4 formula-disjoint validation).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="orbit_pools_smoke_"))

N, C = 12, 3          # 12 groups, up to 3 candidates
mol_of_group = []
g = {
    "query_row": np.arange(N),
    "cand_ptr": np.arange(N + 1) * C,
    "molecule_label": np.tile(np.array([1, 0, 0]), N),
    "mol_ik14": np.array([], dtype="<U14"),
    "formula_cluster": np.array([i // 4 for i in range(N)]),
    "val_query_mask": np.array([i >= 8 for i in range(N)]),
    "ref_ptr": np.arange(N * C + 1),
    "ref_rows": np.arange(N * C),
}
iks = []
for i in range(N):
    iks.extend([f"IKTRUE{i:04d}xxxxx", f"IKFALSE{i:04d}a", f"IKFALSE{i:04d}b"])
g["mol_ik14"] = np.asarray(iks)

# condition ledger: each TRUE molecule has 3 views:
#   row i   : instrument 1, quality 0.5   (the group's own query row)
#   row 100+: instrument 2, quality 0.5   (REAL condition view)
#   row 200+: instrument 1, quality 0.5   (NULL same-condition replicate)
rows, mols, inst, qual = [], [], [], []
for i in range(N):
    for base, k in ((0, 1), (100, 2), (200, 1)):
        rows.append(base + i)
        mols.append(f"IKTRUE{i:04d}xxxxx")
        inst.append(k)
        qual.append(0.5)
v = {"rows": np.asarray(rows), "molecule": np.asarray(mols),
     "instrument": np.asarray(inst), "quality": np.asarray(qual, float)}

# chemical margins: false candidates carry two distinct margins (1 -> 0.3,
# 2 -> 0.5) so the candidate-rotated null can actually permute
m = {"group_id": np.concatenate([np.arange(N), np.arange(N)]),
     "candidate_local": np.concatenate([np.full(N, 1), np.full(N, 2)]),
     "margin": np.concatenate([np.full(N, 0.3, dtype=np.float32),
                               np.full(N, 0.5, dtype=np.float32)])}

cand = tmp / "candidate_groups.npz"
views = tmp / "condition_views.npz"
marg = tmp / "chem_margins.npz"
out = tmp / "orbit_pools.npz"
np.savez(cand, **g)
np.savez(views, **v)
np.savez(marg, **m)

proc = subprocess.run(
    [sys.executable, "-X", "utf8",
     str(ROOT / "tasks/GLM_build_orbit_boundary_groups.py"),
     "--candidate-groups", str(cand), "--condition-views", str(views),
     "--chem-margins", str(marg), "--output", str(out)],
    capture_output=True, text=True)
print(proc.stdout[-1500:])
print(proc.stderr[-1500:])
assert proc.returncode == 0, "builder failed"

report = json.loads((tmp / "orbit_pools.json").read_text(encoding="utf-8"))
with np.load(out) as z:
    orbit_real = z["orbit_row_real"]
    orbit_null = z["orbit_row_null"]
    dr = z["delta_chem_real"]
    dn = z["delta_chem_null"]

assert report["checks"]["I1_true_molecule"]
assert report["checks"]["I2_condition_semantics"]
assert report["checks"]["I4_formula_disjoint_val"]
# every group has BOTH view kinds available (synthetic ledger provides both)
assert (orbit_real >= 0).all() and (orbit_null >= 0).all()
# real views cross instrument (rows 100+), null views are same-condition
# replicates (rows 200+)
assert (orbit_real >= 100).all() and (orbit_real < 200).all()
assert (orbit_null >= 200).all()
# margins: real margins 0.3/0.5 on candidates 1/2; null permutes them
assert np.allclose(dr[:, 1], 0.3) and np.allclose(dr[:, 2], 0.5)
swapped = int(((dn[:, 1] == 0.5) & (dn[:, 2] == 0.3)).sum())
kept = int(((dn[:, 1] == 0.3) & (dn[:, 2] == 0.5)).sum())
assert swapped + kept == N
for q in range(N):
    vals_real = sorted(dr[q, [1, 2]])
    vals_null = sorted(dn[q, [1, 2]])
    assert vals_real == vals_null, "I3: null must be a rotation (multiset)"
    assert dr[q, 0] == 0.0 and dn[q, 0] == 0.0, "true candidate margin 0"
print(f"SMOKE PASS: pools built with BOTH orbit rows (real+null), integrity "
      f"I1-I4 verified, rotation permutes ({swapped} swapped / {kept} kept)")
