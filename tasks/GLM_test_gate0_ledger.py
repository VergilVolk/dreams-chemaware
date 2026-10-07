"""Deterministic test of the Gate-0 complementarity ledger.

Builds a SYNTHETIC chemaware row with KNOWN properties by copying noise_v1
pair scores and applying targeted rank flips:
  - fix K_FIX noise-wrong queries (chem corrects them)
    - half of those also fixed by WSE (overlapping fixes)
    - half fixed by nobody else (independent fixes)
  - break K_BREAK noise-right queries (chem damage)
Then runs the ledger and requires EXACT recovery of every count.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
import os  # noqa: E402
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

K_FIX, K_BREAK = 100, 30
rng = np.random.default_rng(11)

# construct the synthetic chem bundle (identity panel only matters here;
# formula gets an unmodified noise copy named chemaware_synthetic)
g, methods, mol = load_panel("identity_disjoint")
qptr, mptr = g.query_ptr, g.molecule_ptr
rank, _, _ = per_query_stats(g, mol)
correct = rank == 1
mi_n = methods.index("noise_v1")
mi_w = methods.index("weighted_spectral_entropy")
n_ok, w_ok = correct[mi_n], correct[mi_w]

# "independent" pool must match the ledger definition: noise-wrong AND not
# fixed by ANY of the comparison methods (WSE, p2b, entropy_raw, spec2vec,
# denoising), not merely WSE
OTHERS = ("weighted_spectral_entropy", "p2b_noise_v1_frozen",
          "entropy_raw_public", "spec2vec_gnps_public",
          "denoising_search_public")
any_other_fix = np.zeros_like(n_ok)
for o in OTHERS:
    if o in methods:
        any_other_fix |= correct[methods.index(o)]
fix_pool_overlap = np.flatnonzero(~n_ok & any_other_fix)
fix_pool_indep = np.flatnonzero(~n_ok & ~any_other_fix)
break_pool = np.flatnonzero(n_ok)
# independent pool must be large enough for the split
n_indep_target = min(K_FIX // 2, len(fix_pool_indep))
n_overlap_target = K_FIX - n_indep_target
rng.shuffle(fix_pool_overlap); rng.shuffle(fix_pool_indep)
rng.shuffle(break_pool)
fix_q = set(fix_pool_overlap[: n_overlap_target].tolist()
            + fix_pool_indep[: n_indep_target].tolist())
break_q = set(break_pool[:K_BREAK].tolist())

# pair-level construction: copy noise pair scores, boost molecule blocks
with np.load(Path("data/validation/GLM_gnps_article_benchmark_s2v26/"
                  "run15/bundle/method_scores.npz")) as z:
    names = [str(v) for v in z["method_names"]]
    old = {p: np.asarray(z[f"scores_{p}"], np.float32)
           for p in ("identity_disjoint", "formula_disjoint")}
new_scores = {"identity_disjoint":
              np.array(old["identity_disjoint"][mi_n], copy=True),
              "formula_disjoint":
              np.array(old["formula_disjoint"][mi_n], copy=True)}
for q in fix_q:
    lo = int(qptr[q])
    p0, p1 = int(mptr[lo]), int(mptr[lo + 1])
    new_scores["identity_disjoint"][p0:p1] = 1.5
for q in break_q:
    lo = int(qptr[q])
    wrong = lo + 1
    w0, w1 = int(mptr[wrong]), int(mptr[wrong + 1])
    new_scores["identity_disjoint"][w0:w1] = 1.5

bundle = Path("data/validation/GLM_candidate_differential_ledger/"
              "gate0_test_bundle.npz")
assert "chemaware_synthetic" not in names
names2 = names + ["chemaware_synthetic"]
bundle.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(bundle, method_names=np.asarray(names2),
                    scores_identity_disjoint=np.vstack(
                        [old["identity_disjoint"],
                         new_scores["identity_disjoint"][None]]),
                    scores_formula_disjoint=np.vstack(
                        [old["formula_disjoint"],
                         new_scores["formula_disjoint"][None]]))
print(f"synthetic chem row: {len(fix_q)} fixes ({K_FIX//2} overlap-with-WSE"
      f" + {K_FIX//2} independent), {len(break_q)} breaks")

r = subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "tasks/GLM_gate0_complementarity.py"),
                    "--bundle", str(bundle),
                    "--chem-method", "chemaware_synthetic"],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stdout[-600:] + r.stderr[-900:]
print(r.stdout.strip().splitlines()[-6:])

rep = json.loads((ROOT / "data/validation/GLM_candidate_differential_ledger/"
                  "gate0_complementarity.json").read_text(encoding="utf-8"))
e = rep["panels"]["identity_disjoint"]
assert e["G1_intersection"]["noise_wrong"] == int((~n_ok).sum())
assert e["G4_damage"]["chem_breaks_noise_right"] == len(break_q)
# chem fixes = all flip targets that ended correct (construction guarantees)
assert e["G2_conditional_headroom"]["noise_wrong_fixed_by_chem_pct"] == round(
    100 * len(fix_q) / (~n_ok).sum(), 2)
assert e["G3_chem_independent"]["count"] == n_indep_target, \
    f"expected {n_indep_target} independent, got {e['G3_chem_independent']['count']}"
print("GATE-0 LEDGER TEST PASS: exact recovery of constructed fixes "
      f"({len(fix_q)}), independent ({n_indep_target}), breaks ({len(break_q)})")
