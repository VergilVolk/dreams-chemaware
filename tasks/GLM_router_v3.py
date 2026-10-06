"""Router v3: leak-free evaluation of supervised method routing.

LEAK DISCOVERED AND FIXED (2026-10-07):
  The identity-disjoint and formula-disjoint panels share 5,237 of the
  formula panel's 5,261 query spectra. "Train on one panel, test on the
  other" therefore memorizes test queries. Router v2's cross-panel gains
  (+2.93/+4.03pp) are contaminated and are retracted.

Leak-free protocol (both directions):
  Split-A: train on ALL formula-panel queries; test on identity-panel
           queries whose SPECTRUM (query_row) and STRUCTURE (query_ik14)
           were never seen in training.
  Split-B: train on those unseen identity queries only; test on the full
           formula panel (every test query unseen).
Secondary: structure-disjoint 50/50 split within the identity panel.

Features are unchanged from v2 (truth-blind at inference):
  per-method gap / batch-pct gap / top1 / batch-pct top1,
  per-method top1 agreement, modal-candidate consensus, n_mol, precursor mz.
Models: per-method HistGradientBoostingClassifier (deploy argmax of
predicted correctness), plus LogisticRegression as a linear control,
plus pair-level learned weighted fusion.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel  # noqa: E402
from GLM_truthblind_fusion_analysis import (  # noqa: E402
    BENCH, load_panel, per_query_stats, r1,
)
from GLM_router_v2 import (  # noqa: E402
    build_features, learned_fusion_weights, fused_ranks,
)

PANELS = ("identity_disjoint", "formula_disjoint")
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def paired_bootstrap(a: np.ndarray, b: np.ndarray) -> list[float]:
    n = len(a)
    idx = RNG.integers(0, n, size=(BOOT_N, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo * 100), 2), round(float(hi * 100), 2)]


def top1_molecule(graph, mol):
    ptr = graph.query_ptr
    M, _ = mol.shape
    out = np.zeros((M, ptr.shape[0] - 1), dtype=np.int32)
    for i in range(ptr.shape[0] - 1):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if hi <= lo:
            continue
        out[:, i] = np.argmax(mol[:, lo:hi], axis=1)
    return out


def masked_fusion_weights(graph, mol, correct, methods, mask):
    """Pair-level logistic restricted to masked (training) queries."""
    ptr = graph.query_ptr
    Xs, ys = [], []
    for i in np.flatnonzero(mask):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if hi <= lo:
            continue
        block = mol[:, lo:hi].astype(np.float64)
        mu = block.mean(axis=1, keepdims=True)
        sd = np.maximum(block.std(axis=1, keepdims=True), 1e-9)
        Xs.append(((block - mu) / sd).T)
        y = np.zeros(block.shape[1])
        y[0] = 1.0  # positive molecule is block-first
        ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return clf.coef_[0]


def main():
    data = {}
    for panel in PANELS:
        graph, methods, mol = load_panel(panel)
        rank, gap, top1 = per_query_stats(graph, mol)
        data[panel] = dict(graph=graph, methods=methods, rank=rank, gap=gap,
                           top1=top1, mol=mol.astype(np.float64),
                           n_mol=np.diff(graph.query_ptr).astype(np.float64),
                           mz=np.load(BENCH / f"panel_{panel}.npz")[
                               "query_precursor_mz"].astype(np.float64),
                           t1m=top1_molecule(graph, mol))
    # features built once per panel
    for panel in PANELS:
        data[panel]["X"] = build_features(data[panel], data[panel]["t1m"])

    # ---- leak-free splits ------------------------------------------------
    zi = np.load(BENCH / "panel_identity_disjoint.npz")
    zf = np.load(BENCH / "panel_formula_disjoint.npz")
    f_rows = set(map(int, zf["query_row"]))
    f_iks = set(map(str, zf["query_ik14"]))
    id_rows = zi["query_row"]
    id_iks = zi["query_ik14"].astype(str)
    unseen = np.array([(int(r) not in f_rows and ik not in f_iks)
                       for r, ik in zip(id_rows, id_iks)])
    print(f"identity queries unseen in formula panel: {unseen.sum()} "
          f"/ {len(unseen)}")

    splits = {
        "A_train_formula_test_unseen_identity": (
            "formula_disjoint", np.ones(len(zf["query_row"]), dtype=bool),
            "identity_disjoint", unseen),
        "B_train_unseen_identity_test_formula": (
            "identity_disjoint", unseen,
            "formula_disjoint", np.ones(len(zf["query_row"]), dtype=bool)),
    }

    # secondary: structure-disjoint 50/50 within identity
    iks = id_iks
    uniq = np.array(sorted(set(iks)))
    half = set(uniq[RNG.permutation(len(uniq))[: len(uniq) // 2]])
    test_c = np.isin(iks, list(half))
    splits["C_identity_struct_split"] = (
        "identity_disjoint", ~test_c, "identity_disjoint", test_c)

    report: dict = {"status": "GLM_ROUTER_V3_LEAKFREE",
                    "retraction": ("Router v2 cross-panel gains were "
                                   "contaminated: panels share 5,237/5,261 "
                                   "formula-panel query spectra; v3 tests on "
                                   "genuinely unseen queries only."),
                    "splits": {}}
    for split_name, (tr_panel, tr_mask, te_panel, te_mask) in splits.items():
        tr, te = data[tr_panel], data[te_panel]
        methods = tr["methods"]
        M = tr["rank"].shape[0]
        Xtr, ytr = tr["X"][tr_mask], (tr["rank"] == 1).T[tr_mask]
        Xte = te["X"][te_mask]
        N = len(Xte)
        correct_te = (te["rank"] == 1)[:, te_mask]
        best_i = int(np.argmax((te["rank"] == 1).mean(axis=1)))
        best_mask = correct_te[best_i]
        near = te["graph"].query_has_near[te_mask]

        res: dict = {"_best_single": {"method": methods[best_i],
                                      "recall1": r1(best_mask)}}
        for model_name, mk in (
            ("logreg", lambda: Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, C=1.0))])),
            ("hgb", lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                l2_regularization=1.0, random_state=20261007))):
            probs = np.zeros((N, M))
            for m_i in range(M):
                clf = mk()
                clf.fit(Xtr, ytr[:, m_i])
                probs[:, m_i] = clf.predict_proba(Xte)[:, 1]
            pick = probs.argmax(axis=1)
            mask = correct_te[pick, np.arange(N)]
            res[f"router_{model_name}"] = {
                "train_panel": tr_panel, "test_panel": te_panel,
                "n_train": int(len(Xtr)), "n_test": int(N),
                "recall1": r1(mask), "near_recall1": r1(mask[near]),
                "delta_vs_best_single_pp": round(r1(mask) - r1(best_mask), 2),
                "ci95_vs_best_single": paired_bootstrap(mask, best_mask),
                "selection_distribution": {
                    methods[m_i]: {"selected": int((pick == m_i).sum()),
                                   "accuracy": r1(mask[pick == m_i])}
                    for m_i in range(M) if (pick == m_i).any()},
            }

        # learned weighted fusion (weights from train rows only)
        w = masked_fusion_weights(tr["graph"], tr["mol"], (tr["rank"] == 1),
                                  methods, tr_mask)
        ranks = fused_ranks(te["graph"], te["mol"], w)
        mask = (ranks == 1)[te_mask]
        res["learned_fusion"] = {
            "train_panel": tr_panel, "n_test": int(N),
            "recall1": r1(mask), "near_recall1": r1(mask[near]),
            "delta_vs_best_single_pp": round(r1(mask) - r1(best_mask), 2),
            "ci95_vs_best_single": paired_bootstrap(mask, best_mask),
            "fusion_weights": {m: round(float(x), 4)
                               for m, x in zip(methods, w)},
        }

        report["splits"][split_name] = res
        print(f"\n=== {split_name} ===")
        for name in res:
            if name.startswith("_"):
                continue
            r = res[name]
            print(f"  {name:18s} {r['recall1']:6.2f}%  "
                  f"delta {r['delta_vs_best_single_pp']:+.2f}pp  "
                  f"CI [{r['ci95_vs_best_single'][0]:+.2f}, "
                  f"{r['ci95_vs_best_single'][1]:+.2f}]  near {r['near_recall1']}")

    out = Path(os.environ.get(
        "GLM_OUT_DIR",
        "deliverables/GLM_gnps_article_ladder/run_local")) / "router_v3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
