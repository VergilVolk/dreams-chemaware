"""Router v2: stronger truth-blind features + learned fusion, cross-panel.

Features per query (all deployment-computable, no truth):
  - per-method raw top1-top2 gap and its batch percentile     (2M)
  - per-method raw top1 score and its batch percentile         (2M)
  - per-method agreement: fraction of other methods whose top1
    candidate molecule is the same as this method's            (M)
  - consensus: fraction of methods backing the modal candidate (1)
  - n candidate molecules, precursor mz                        (2)
Models:
  - per-method LogisticRegression and HistGradientBoostingClassifier
    (predict P(method ranks positive first); deploy argmax)
  - learned weighted fusion: logistic on within-query z-scored
    molecule scores (pair level, weights learned on train panel)
Evaluation: train identity -> test formula, and the reverse, with
paired bootstrap CIs vs the best single method. Frozen strict-rank
semantics throughout (ties count against the positive).
"""
from __future__ import annotations

import json
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
    BENCH, RUN, load_panel, per_query_stats, r1,
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
    """Per (method, query): local index of the top-scoring molecule."""
    ptr = graph.query_ptr
    M, _ = mol.shape
    out = np.zeros((M, ptr.shape[0] - 1), dtype=np.int32)
    for i in range(ptr.shape[0] - 1):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if hi <= lo:
            continue
        block = mol[:, lo:hi]
        out[:, i] = np.argmax(block, axis=1)
    return out


def build_features(d, top1mol):
    gap, top1, N = d["gap"], d["top1"], d["gap"].shape[1]
    M = gap.shape[0]
    gp = np.argsort(np.argsort(gap, axis=1), axis=1) / (N - 1)
    tp = np.argsort(np.argsort(top1, axis=1), axis=1) / (N - 1)
    # agreement: for method m, fraction of others with same top1 molecule
    agree = np.zeros((M, N), dtype=np.float64)
    for i in range(N):
        t = top1mol[:, i]
        counts = np.bincount(t, minlength=M)  # rough cap; may over-alloc
        agree[:, i] = (counts[t] - 1) / (M - 1)
    # consensus: fraction backing the modal candidate
    consensus = np.zeros(N, dtype=np.float64)
    for i in range(N):
        t = top1mol[:, i]
        counts = np.bincount(t)
        consensus[i] = counts.max() / M
    return np.column_stack([
        gap.T.astype(np.float64), gp.T.astype(np.float64),
        top1.T.astype(np.float64), tp.T.astype(np.float64),
        agree.T, consensus[:, None],
        d["n_mol"][:, None], d["mz"][:, None]])


def learned_fusion_weights(graph, mol, correct, methods):
    """Pair-level logistic: label ~ within-query z-scored scores per method."""
    ptr = graph.query_ptr
    M = mol.shape[0]
    Xs, ys = [], []
    for i in range(graph.n_queries):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if hi <= lo:
            continue
        block = mol[:, lo:hi].astype(np.float64)
        mu = block.mean(axis=1, keepdims=True)
        sd = np.maximum(block.std(axis=1, keepdims=True), 1e-9)
        Xs.append(((block - mu) / sd).T)
        ys.append(np.zeros(block.shape[1]))
        ys[-1][0] = 1.0  # positive molecule is block-first
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return clf.coef_[0]


def fused_ranks(graph, mol, w):
    ptr = graph.query_ptr
    ranks = np.zeros(ptr.shape[0] - 1, dtype=np.int32)
    for i in range(graph.n_queries):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if hi <= lo:
            continue
        block = mol[:, lo:hi].astype(np.float64)
        mu = block.mean(axis=1, keepdims=True)
        sd = np.maximum(block.std(axis=1, keepdims=True), 1e-9)
        agg = (w[:, None] * ((block - mu) / sd)).sum(axis=0)
        ranks[i] = 1 + int((agg[1:] >= agg[0]).sum())
    return ranks


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

    report: dict = {"status": "GLM_ROUTER_V2", "panels": {}}
    for panel in PANELS:
        other = PANELS[1 - PANELS.index(panel)]
        d, o = data[panel], data[other]
        methods = d["methods"]
        M, N = d["rank"].shape
        correct = d["rank"] == 1
        best_i = int(np.argmax(correct.mean(axis=1)))
        best_mask = correct[best_i]
        near = d["graph"].query_has_near

        Xtr, ytr = build_features(o, o["t1m"]), (o["rank"] == 1).T
        Xte = build_features(d, d["t1m"])

        res: dict = {"_best_single": {"method": methods[best_i],
                                      "recall1": r1(best_mask)}}
        for model_name, mk in (
            ("logreg", lambda: Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, C=1.0))])),
            ("hgb", lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_depth=None,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=20261007))):
            probs = np.zeros((N, M))
            for m_i in range(M):
                clf = mk()
                clf.fit(Xtr, ytr[:, m_i])
                probs[:, m_i] = clf.predict_proba(Xte)[:, 1]
            pick = probs.argmax(axis=1)
            mask = correct[pick, np.arange(N)]
            res[f"R2_router_{model_name}"] = {
                "trained_on": other, "recall1": r1(mask),
                "near_recall1": r1(mask[near]),
                "delta_vs_best_single_pp": round(r1(mask) - r1(best_mask), 2),
                "ci95_vs_best_single": paired_bootstrap(mask, best_mask),
                "supervised": True,
                "selection_distribution": {
                    methods[m_i]: {"selected": int((pick == m_i).sum()),
                                   "accuracy": r1(mask[pick == m_i])}
                    for m_i in range(M) if (pick == m_i).any()},
            }

        # learned weighted fusion (weights from train panel)
        w = learned_fusion_weights(o["graph"], o["mol"], (o["rank"] == 1),
                                   methods)
        ranks = fused_ranks(d["graph"], d["mol"], w)
        mask = ranks == 1
        res["R2_learned_fusion"] = {
            "trained_on": other, "recall1": r1(mask),
            "near_recall1": r1(mask[near]),
            "delta_vs_best_single_pp": round(r1(mask) - r1(best_mask), 2),
            "ci95_vs_best_single": paired_bootstrap(mask, best_mask),
            "supervised": True,
            "fusion_weights": {m: round(float(x), 4)
                               for m, x in zip(methods, w)},
        }

        report["panels"][panel] = res
        print(f"\n=== {panel} (train: {other}) ===")
        for name in res:
            if name.startswith("_"):
                continue
            r = res[name]
            print(f"  {name:22s} {r['recall1']:6.2f}%  "
                  f"delta {r['delta_vs_best_single_pp']:+.2f}pp  "
                  f"CI [{r['ci95_vs_best_single'][0]:+.2f}, "
                  f"{r['ci95_vs_best_single'][1]:+.2f}]  "
                  f"near {r['near_recall1']}")

    out = ROOT / "deliverables/GLM_gnps_article_ladder/run_local/router_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
