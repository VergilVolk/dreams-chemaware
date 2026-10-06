"""Truth-blind ensemble analysis on the frozen 14-method GNPS benchmark.

RETRACTION CONTEXT (2026-10-07):
  The earlier "S1 margin-selection = oracle (+4.93/+5.44pp)" result used
  `positive_vs_best_negative_margin` as the per-method confidence signal.
  That quantity is computed from the TRUE positive's score, so it leaks the
  answer key: any method that ranked the positive first tends to show a
  large positive margin, and argmax over methods then reproduces the oracle.
  S1-as-reported was therefore an oracle bound, not a deployable algorithm.

This script quantifies what is achievable WITHOUT truth leakage:
  Unsupervised (no labels, deployment-computable):
    U1 select method by raw top1-top2 gap            (blind S1)
    U2 select method by batch-z-scored gap           (transductive)
    U2x z-constants transferred from the other panel (inductive)
    U3 select method by batch-percentile of gap
    U4 within-query z-score fusion of molecule scores (sum)
    U5 Borda rank fusion at molecule level
    U6 reciprocal-rank fusion (RRF, k=60)
  Supervised (labels used on the TRAIN panel only):
    R_cross_panel_router: per-method logistic router, trained on one panel,
      evaluated on the other (both directions).
  References:
    best single method, oracle-any-method (upper bound, NOT deployable).

All R@1 are molecule-level Top-1 identification, identical to the frozen
evaluator semantics (max over candidate pairs per molecule, rank of the
labeled molecule within the query's molecule block).
Verification: per-method R@1 must reproduce the frozen per-query tables.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel  # noqa: E402

BENCH = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"
RUN = Path(os.environ.get("GLM_RUN_DIR",
                          "data/validation/GLM_gnps_article_benchmark_"
                          "challengers/run_local"))
OUT = Path(os.environ.get(
    "GLM_OUT_DIR", "deliverables/GLM_gnps_article_ladder/run_local"))
PANELS = ("identity_disjoint", "formula_disjoint")
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def load_panel(name: str):
    graph = graph_from_panel(BENCH / f"panel_{name}.npz")
    with np.load(RUN / "bundle/method_scores.npz", allow_pickle=False) as b:
        methods = [str(m) for m in b["method_names"]]
        scores = np.asarray(b[f"scores_{name}"], dtype=np.float32)
    assert scores.shape[1] == len(graph.pair_candidate_row)
    mol = np.maximum.reduceat(scores, graph.molecule_ptr[:-1], axis=1)
    return graph, methods, mol


def per_query_stats(graph, mol):
    """Per (method, query): rank of the positive molecule, top1-top2 gap, top1 score."""
    M, _ = mol.shape
    q = graph.n_queries
    ptr, mptr = graph.query_ptr, graph.molecule_ptr
    label = graph.molecule_label
    rank = np.zeros((M, q), dtype=np.int32)
    gap = np.zeros((M, q), dtype=np.float32)
    top1 = np.zeros((M, q), dtype=np.float32)
    for i in range(q):
        lo, hi = ptr[i], ptr[i + 1]
        if hi <= lo:
            continue
        block = mol[:, lo:hi].astype(np.float64)  # mol columns are molecules
        # frozen semantics: the positive molecule is always the FIRST row of
        # the query block; strict tie policy counts ties AGAINST the positive.
        rank[:, i] = 1 + (block[:, 1:] >= block[:, 0:1]).sum(axis=1)
        order = np.argsort(-block, axis=1, kind="stable")
        srt = np.take_along_axis(block, order, axis=1)
        top1[:, i] = srt[:, 0]
        gap[:, i] = (srt[:, 0] - srt[:, 1] if block.shape[1] > 1
                     else srt[:, 0])
    return rank, gap, top1


def paired_bootstrap(a_mask: np.ndarray, b_mask: np.ndarray) -> list[float]:
    n = len(a_mask)
    a = a_mask.astype(np.float64)
    b = b_mask.astype(np.float64)
    idx = RNG.integers(0, n, size=(BOOT_N, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo * 100), 2), round(float(hi * 100), 2)]


def r1(mask: np.ndarray) -> float:
    return round(float(mask.mean() * 100), 2)


def make_features(dd, n_queries):
    gp = np.argsort(np.argsort(dd["gap"], axis=1), axis=1) / (n_queries - 1)
    tp = np.argsort(np.argsort(dd["top1"], axis=1), axis=1) / (n_queries - 1)
    return np.column_stack([gp.T.astype(np.float64),
                            tp.T.astype(np.float64),
                            dd["n_mol"].astype(np.float64)[:, None],
                            dd["mz"][:, None]])


def main():
    data = {}
    for panel in PANELS:
        graph, methods, mol = load_panel(panel)
        rank, gap, top1 = per_query_stats(graph, mol)
        for m_i, m in enumerate(methods):
            path = RUN / "evaluation" / f"queries_{panel}_{m}.csv.gz"
            if not path.is_file():
                continue
            tbl = pd.read_csv(path, low_memory=False).sort_values(
                "query_index", kind="stable")
            assert (tbl["query_index"].to_numpy()
                    == np.arange(graph.n_queries)).all()
            got = float((rank[m_i] == 1).mean() * 100)
            want = float((tbl["rank"].to_numpy() == 1).mean() * 100)
            assert abs(got - want) < 0.02, (panel, m, got, want)
        data[panel] = dict(graph=graph, methods=methods, rank=rank, gap=gap,
                           top1=top1, mol=mol,
                           n_mol=np.diff(graph.query_ptr).astype(np.float64),
                           mz=np.load(BENCH / f"panel_{panel}.npz")[
                               "query_precursor_mz"].astype(np.float64))
    print("dual-source verification passed: per-method R@1 reproduces the "
          "frozen per-query tables on both panels")

    report: dict = {"status": "GLM_TRUTHBLIND_ENSEMBLE_ANALYSIS",
                    "retraction": ("Earlier S1 margin-selection used "
                                   "positive_vs_best_negative_margin which "
                                   "leaks the answer key; it is an oracle "
                                   "reference, not a deployable algorithm."),
                    "panels": {}}

    for panel in PANELS:
        d = data[panel]
        other = PANELS[1 - PANELS.index(panel)]
        o = data[other]
        methods, rank, gap, top1 = d["methods"], d["rank"], d["gap"], d["top1"]
        graph = d["graph"]
        M, N = rank.shape
        correct = rank == 1
        best_i = int(np.argmax(correct.mean(axis=1)))
        best_m = methods[best_i]
        oracle = correct.any(axis=0)
        near = graph.query_has_near
        ptr, mptr = graph.query_ptr, graph.molecule_ptr

        sel: dict[str, np.ndarray] = {}

        def selection_outcome(signal):
            pick = np.argmax(signal, axis=0)
            return correct[pick, np.arange(N)]

        sel["U1_select_raw_gap"] = selection_outcome(gap)

        mu, sd = gap.mean(axis=1, keepdims=True), gap.std(axis=1, keepdims=True)
        sel["U2_select_batch_z_gap"] = selection_outcome(
            (gap - mu) / np.maximum(sd, 1e-9))

        # inductive: z-constants from the OTHER panel
        mu_o = o["gap"].mean(axis=1, keepdims=True)
        sd_o = np.maximum(o["gap"].std(axis=1, keepdims=True), 1e-9)
        sel["U2x_select_transfer_z_gap"] = selection_outcome(
            (gap - mu_o) / sd_o)

        pct = np.argsort(np.argsort(gap, axis=1), axis=1) / (N - 1)
        sel["U3_select_batch_pct_gap"] = selection_outcome(pct)

        fused_rank = np.zeros(N, dtype=np.int32)
        bordo_rank = np.zeros(N, dtype=np.int32)
        rrf_rank = np.zeros(N, dtype=np.int32)
        mol = d["mol"].astype(np.float64)
        for i in range(N):
            lo, hi = ptr[i], ptr[i + 1]
            if hi <= lo:
                continue
            block = mol[:, lo:hi]
            mu_q = block.mean(axis=1, keepdims=True)
            sd_q = np.maximum(block.std(axis=1, keepdims=True), 1e-9)
            agg = ((block - mu_q) / sd_q).sum(axis=0)
            fused_rank[i] = 1 + int((agg[1:] >= agg[0]).sum())
            r = np.argsort(np.argsort(-block, axis=1), axis=1) + 1
            b = r.mean(axis=0)
            bordo_rank[i] = 1 + int((b[1:] <= b[0]).sum())
            rr = (1.0 / (60 + r)).sum(axis=0)
            rrf_rank[i] = 1 + int((rr[1:] >= rr[0]).sum())
        sel["U4_query_z_fusion"] = fused_rank == 1
        sel["U5_borda_fusion"] = bordo_rank == 1
        sel["U6_rrf_fusion"] = rrf_rank == 1

        res: dict = {
            "_best_single": {"method": best_m, "recall1": r1(correct[best_i])},
            "_oracle_any": {"recall1": r1(oracle)},
        }
        for name, mask in sel.items():
            res[name] = {
                "recall1": r1(mask),
                "near_recall1": r1(mask[near]) if near.any() else None,
                "delta_vs_best_single_pp": round(
                    r1(mask) - r1(correct[best_i]), 2),
                "ci95_vs_best_single": paired_bootstrap(mask, correct[best_i]),
                "supervised": False,
            }

        # supervised cross-panel router
        Xtr, ytr = make_features(o, N), (o["rank"] == 1).T
        Xte, yte = make_features(d, N), (d["rank"] == 1).T
        probs = np.zeros((Xte.shape[0], M))
        for m_i in range(M):
            clf = Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=2000, C=1.0))])
            clf.fit(Xtr, ytr[:, m_i])
            probs[:, m_i] = clf.predict_proba(Xte)[:, 1]
        pick = probs.argmax(axis=1)
        router_mask = yte[np.arange(N), pick]
        res["R_cross_panel_router"] = {
            "trained_on": other,
            "recall1": r1(router_mask),
            "near_recall1": r1(router_mask[near]) if near.any() else None,
            "delta_vs_best_single_pp": round(
                r1(router_mask) - r1(correct[best_i]), 2),
            "ci95_vs_best_single": paired_bootstrap(router_mask,
                                                    correct[best_i]),
            "supervised": True,
            "selection_distribution": {
                methods[m_i]: {"selected": int((pick == m_i).sum()),
                               "accuracy": r1(router_mask[pick == m_i])}
                for m_i in range(M) if (pick == m_i).any()},
        }

        report["panels"][panel] = res
        print(f"\n=== {panel} (n={N:,}) ===")
        print(f"  best single: {best_m} = {r1(correct[best_i]):.2f}%   "
              f"oracle-any = {r1(oracle):.2f}%")
        for name in res:
            if name.startswith("_"):
                continue
            r = res[name]
            print(f"  {name:28s} {r['recall1']:6.2f}%  "
                  f"delta {r['delta_vs_best_single_pp']:+.2f}pp  "
                  f"CI [{r['ci95_vs_best_single'][0]:+.2f}, "
                  f"{r['ci95_vs_best_single'][1]:+.2f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truthblind_ensemble.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(f"\nwritten: {OUT / 'truthblind_ensemble.json'}")


if __name__ == "__main__":
    main()
