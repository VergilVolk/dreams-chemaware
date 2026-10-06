"""Permutation control for the leak-free router.

Question: is the router's +0.5pp gain on 15 methods a real signal or an
artifact of the pipeline? Control: permute the training labels (per-method
correctness) within the training split, retrain the identical HGB router,
evaluate on the untouched test split. If the gain collapses to ~0 under
label permutation, the observed gain is label-driven (real signal).

Splits tested: A (formula -> unseen identity) and B (unseen identity ->
formula), matching router_v3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats, r1  # noqa: E402
from GLM_router_v2 import build_features  # noqa: E402
from GLM_router_v3 import top1_molecule  # noqa: E402

PANELS = ("identity_disjoint", "formula_disjoint")
N_PERM = 5
RNG = np.random.default_rng(20261007)

data = {}
for panel in PANELS:
    g, methods, mol = load_panel(panel)
    rank, gap, top1 = per_query_stats(g, mol)
    data[panel] = dict(graph=g, methods=methods, rank=rank, gap=gap,
                       top1=top1, mol=mol.astype(np.float64),
                       n_mol=np.diff(g.query_ptr).astype(np.float64),
                       mz=np.load(Path("data/validation/"
                                       "GLM_gnps_identity_panel_reconstruction"
                                       ) / f"panel_{panel}.npz")[
                           "query_precursor_mz"].astype(np.float64),
                       t1m=top1_molecule(g, mol))
    data[panel]["X"] = build_features(data[panel], data[panel]["t1m"])

zi = np.load(Path("data/validation/GLM_gnps_identity_panel_reconstruction/"
                  "panel_identity_disjoint.npz"))
zf = np.load(Path("data/validation/GLM_gnps_identity_panel_reconstruction/"
                  "panel_formula_disjoint.npz"))
f_rows = set(map(int, zf["query_row"]))
f_iks = set(map(str, zf["query_ik14"]))
id_rows = zi["query_row"]
id_iks = zi["query_ik14"].astype(str)
unseen = np.array([(int(r) not in f_rows and k not in f_iks)
                   for r, k in zip(id_rows, id_iks)])

splits = {
    "A_train_formula_test_unseen_identity": (
        "formula_disjoint", np.ones(len(zf["query_row"]), dtype=bool),
        "identity_disjoint", unseen),
    "B_train_unseen_identity_test_formula": (
        "identity_disjoint", unseen,
        "formula_disjoint", np.ones(len(zf["query_row"]), dtype=bool)),
}

report = {"status": "GLM_ROUTER_PERMUTATION_CONTROL",
          "n_permutations": N_PERM, "splits": {}}
for name, (tr_p, tr_m, te_p, te_m) in splits.items():
    tr, te = data[tr_p], data[te_p]
    methods = tr["methods"]
    M = tr["rank"].shape[0]
    Xtr, ytr = tr["X"][tr_m], (tr["rank"] == 1).T[tr_m]
    Xte = te["X"][te_m]
    correct_te = (te["rank"] == 1)[:, te_m]
    best_i = int(np.argmax((te["rank"] == 1).mean(axis=1)))
    best_mask = correct_te[best_i]
    N = len(Xte)

    def run_router(ytrain):
        probs = np.zeros((N, M))
        for m_i in range(M):
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                l2_regularization=1.0, random_state=20261007)
            clf.fit(Xtr, ytrain[:, m_i])
            probs[:, m_i] = clf.predict_proba(Xte)[:, 1]
        pick = probs.argmax(axis=1)
        return r1(correct_te[pick, np.arange(N)])

    true_gain = run_router(ytr) - r1(best_mask)
    perm_gains = []
    for p in range(N_PERM):
        yperm = ytr.copy()
        for m_i in range(M):
            yperm[:, m_i] = yperm[RNG.permutation(len(yperm)), m_i]
        perm_gains.append(round(run_router(yperm) - r1(best_mask), 2))
        print(f"  {name} perm {p + 1}/{N_PERM}: "
              f"{perm_gains[-1]:+.2f}pp", flush=True)
    report["splits"][name] = {
        "true_gain_pp": round(true_gain, 2),
        "permutation_gains_pp": perm_gains,
        "permutation_mean": round(float(np.mean(perm_gains)), 3),
    }
    print(f"{name}: true {true_gain:+.2f}pp | perm mean "
          f"{np.mean(perm_gains):+.3f}pp")

out = ROOT / "deliverables/GLM_gnps_article_ladder/run15/router_permutation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
