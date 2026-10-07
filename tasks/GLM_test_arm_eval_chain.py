"""End-to-end smoke of the arm evaluation chain (local, synthetic encoder).

Build pool fixtures -> train a 5-step synthetic R arm -> embed ALL panel
spectra -> merge bundle row (+arm) -> frozen evaluator -> R@1 readout.
Validates the ENTIRE post-training pipeline that server arms will use.
Numbers are pipeline validation only (random-quality encoder).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="arm_eval_smoke_"))

# ---- fixtures (minimal pool via the real builder) ------------------------
N, C = 16, 3
mol_ids = np.array([f"IK{i:04d}abc" for i in range(N * C)])
g = {
    "query_row": np.arange(N, dtype=np.int64),
    "cand_ptr": np.arange(N + 1, dtype=np.int64) * C,
    "molecule_label": np.tile(np.array([1, 0, 0], dtype=np.int8), N),
    "mol_ik14": mol_ids,
    "formula_cluster": np.array([i // 4 for i in range(N)]),
    "val_query_mask": np.array([i >= 12 for i in range(N)]),
    "ref_ptr": np.arange(N * C + 1, dtype=np.int64),
    "ref_rows": np.arange(N * C, dtype=np.int64),
}
rows, mols, inst, qual = [], [], [], []
for i in range(N):
    views_mol = mol_ids[i * C]
    for base, k in ((0, 1), (100, 2), (200, 1)):
        rows.append(base + i); mols.append(views_mol)
        inst.append(k); qual.append(0.5)
v = {"rows": np.asarray(rows), "molecule": np.asarray(mols),
     "instrument": np.asarray(inst), "quality": np.asarray(qual, float)}
m = {"group_id": np.concatenate([np.arange(N)] * 2),
     "candidate_local": np.concatenate([np.full(N, 1), np.full(N, 2)]),
     "margin": np.concatenate([np.full(N, 0.3, np.float32),
                               np.full(N, 0.5, np.float32)])}
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
assert r.returncode == 0, r.stderr[-600:]
sha = json.loads((tmp / "pool.json").read_text(encoding="utf-8"))["pool_sha256"]

# ---- train arm R (5 steps) ------------------------------------------------
arm_dir = tmp / "armR"
r = subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "tasks/GLM_train_orbit_boundary_encoder.py"),
                    "--arm", "R", "--pool", str(pool),
                    "--expected-pool-sha256", sha,
                    "--encoder", "synthetic",
                    "--steps", "5", "--batch-size", "4", "--lr", "1e-3",
                    "--seed", "3407", "--output-dir", str(arm_dir)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-600:]
ckpts = sorted(arm_dir.glob("R_step*.ckpt"))
assert ckpts, "no checkpoint saved"
ckpt = ckpts[-1]

# ---- embed -> ladder (the real chain) -------------------------------------
emb = tmp / "arm_embeddings.npz"
r = subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "tasks/GLM_embed_panel_spectra.py"),
                    "--checkpoint", str(ckpt), "--encoder", "synthetic",
                    "--output", str(emb)], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]
print(r.stdout.strip().splitlines()[-2:])

run_dir = tmp / "run_arm_smoke"
r = subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "tasks/GLM_arm_scores_to_ladder.py"),
                    "--embeddings", str(emb),
                    "--method-name", "arm_R_smoke",
                    "--run-dir", str(run_dir)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stdout[-500:] + r.stderr[-1200:]
print(r.stdout.strip().splitlines()[-2:])

report = json.loads((run_dir / "eval_chain_report.json").read_text(
    encoding="utf-8"))
assert report["status"] == "GLM_ARM_EVAL_CHAIN_COMPLETE"
assert report["n_methods"] == 16
assert 0.0 <= report["identity_recall1"] <= 100.0
print(f"ARM EVAL CHAIN SMOKE PASS: 16-method bundle row added, frozen "
      f"evaluator ran, arm R@1 = {report['identity_recall1']:.2f}% "
      f"(synthetic quality, pipeline validated)")
