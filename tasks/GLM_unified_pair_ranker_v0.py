"""Unified candidate-pair ranker v0 (constrained, no chemical layer yet).

Implements the corrected unified design's ranking function with the locally
available views:

  M_ab = a_N * d(noise) + a_W * d(WSE)                     <- G_ab
       + sum_k g_k * d(P2b_k)                              <- rerankers
       + sum_c b_c * D_c                                   <- exclusive evidence
                                                           (raw; chemical
                                                           double-null D_ab
                                                           is the next layer)

  - pool per query = union(noise top-5, WSE top-5, P2b-on-Noise top-5),
    deduplicated (corrected design point 4)
  - directed pairs inside the pool; antisymmetric features by construction
  - coefficients >= 0 fitted by scipy NNLS on TRAIN pairs that contain the
    positive molecule; frozen; applied to all pairs
  - final ranking via Bradley-Terry net wins: score(a) = sum_b M_ab
  - leak-free splits A/B/C (query spectrum + structure deduplicated)
  - same-denominator baselines: WSE, Noise, P2b-on-Noise molecule rankings
  - report: Top-1, MRR, corrected/introduced vs WSE, near subset, CIs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_score_challenger_models_on_gnps import parse_mgf_used  # noqa: E402
from GLM_truthblind_fusion_analysis import load_panel  # noqa: E402

PANELS = ("identity_disjoint", "formula_disjoint")
BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT = ROOT / "data/validation/GLM_candidate_differential_ledger"

POOL_VIEWS = ("noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen")
FEATURE_VIEWS = ("noise_v1", "weighted_spectral_entropy",
                 "p2b_sqrt_cosine", "p2b_unweighted_entropy",
                 "neutral_loss_sqrt_cosine", "p2b_noise_v1_frozen")
M_POOL = 5
TOL, STABILITY, TOPK_DIFF = 0.02, 0.5, 10
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def consensus_of(mz_arrays):
    if not mz_arrays:
        return np.array([])
    all_peaks = np.concatenate(mz_arrays)
    all_peaks.sort()
    kept, i = [], 0
    n = len(mz_arrays)
    while i < len(all_peaks):
        j = i
        while j < len(all_peaks) and all_peaks[j] - all_peaks[i] <= TOL:
            j += 1
        distinct = sum(1 for mz in mz_arrays
                       if np.searchsorted(mz, all_peaks[i] + TOL)
                       > np.searchsorted(mz, all_peaks[i] - TOL))
        if distinct >= max(1, int(np.ceil(STABILITY * n))):
            kept.append(float(np.mean(all_peaks[i:j])))
        i = j
    return np.asarray(kept)


def match_any(haystack, needle, tol=TOL):
    if len(haystack) == 0:
        return False
    return (np.searchsorted(haystack, needle + tol)
            > np.searchsorted(haystack, needle - tol))


def main() -> None:
    data = {}
    for panel in PANELS:
        g, methods, mol = load_panel(panel)
        data[panel] = dict(graph=g, methods=methods, mol=mol)

    # spectra: needed for every query and every pooled molecule's refs
    needed = set()
    pools = {}
    for panel in PANELS:
        g, methods, mol = data[panel]["graph"], data[panel]["methods"], data[panel]["mol"]
        qptr, mptr = g.query_ptr, g.molecule_ptr
        needed.update(map(int, g.query_row))
        pl = {}
        for q in range(g.n_queries):
            lo, hi = int(qptr[q]), int(qptr[q + 1])
            if hi - lo < 2:
                pl[q] = []
                continue
            s = set()
            for v in POOL_VIEWS:
                mi = methods.index(v)
                s.update((lo + np.argsort(-mol[mi][lo:hi],
                                          kind="stable")[:M_POOL]).tolist())
            pl[q] = sorted(s)
            for m in pl[q]:
                needed.update(map(int, g.pair_candidate_row[mptr[m]:mptr[m + 1]]))
        pools[panel] = pl
    print(f"spectra needed: {len(needed):,}", flush=True)
    spectra = parse_mgf_used(BENCH / "spectra.mgf", needed)
    print("spectra parsed", flush=True)

    def build(q_info):
        """Per (panel, query): pool, per-view score vectors, exclusives."""
        panel, q = q_info
        g, methods, mol = data[panel]["graph"], data[panel]["methods"], data[panel]["mol"]
        pool = pools[panel][q]
        if len(pool) < 2:
            return None
        qmz, qint, qprec = spectra[int(g.query_row[q])]
        qmz = np.sort(np.asarray(qmz, float))
        q_losses = np.sort(float(qprec) - qmz)
        # molecule consensus
        cons = {}
        for m in pool:
            frag, loss, inten_l = [], [], []
            for r in g.pair_candidate_row[g.molecule_ptr[m]:g.molecule_ptr[m + 1]]:
                mz, inten, prec = spectra[int(r)]
                mz = np.asarray(mz, float)
                o = np.argsort(mz)
                frag.append(mz[o]); inten_l.append((mz[o], np.asarray(inten, float)[o]))
                loss.append(np.sort(float(prec) - mz))
            best = max(inten_l, key=lambda t: len(t[0])) if inten_l else (np.array([]), np.array([]))
            tp = (np.sort(best[0][np.argsort(-best[1])[:TOPK_DIFF]])
                  if len(best[0]) else np.array([]))
            d = (np.unique(np.round(np.abs(tp[:, None] - tp[None, :])[np.abs(tp[:, None] - tp[None, :]) > TOL], 4))
                 if len(tp) > 1 else np.array([]))
            cons[m] = (consensus_of(frag), consensus_of(loss), d)
        scores = {}
        for v in FEATURE_VIEWS:
            mi = methods.index(v)
            scores[v] = {m: float(mol[mi][m]) for m in pool}
        return pool, cons, scores, qmz, q_losses

    def pair_features(pool, cons, scores, qmz, q_losses):
        """Directed-pair feature matrix, rows ordered as (a,b) pairs."""
        rows, pairs = [], []
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                fa, la, da = cons[a]
                fb, lb, db = cons[b]
                ex_a = sum(1 for x in fa if not match_any(fb, x)
                           and match_any(qmz, x))
                ex_b = sum(1 for x in fb if not match_any(fa, x)
                           and match_any(qmz, x))
                dfrag = ex_a - ex_b
                ex_a = sum(1 for x in la if not match_any(lb, x)
                           and match_any(q_losses, x))
                ex_b = sum(1 for x in lb if not match_any(la, x)
                           and match_any(q_losses, x))
                dnl = ex_a - ex_b
                ex_a = sum(1 for x in da if not match_any(db, x, 0.03)
                           and match_any(q_losses, x, 0.03))
                ex_b = sum(1 for x in db if not match_any(da, x, 0.03)
                           and match_any(q_losses, x, 0.03))
                ddiff = ex_a - ex_b
                feat = ([scores[v][a] - scores[v][b] for v in FEATURE_VIEWS]
                        + [dfrag, dnl, ddiff])
                rows.append(feat)
                pairs.append((a, b))
                rows.append([-x for x in feat])
                pairs.append((b, a))
        return np.asarray(rows, float), pairs

    # leak-free splits
    zi = np.load(ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz")
    zf = np.load(ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction/panel_formula_disjoint.npz")
    f_rows = set(map(int, zf["query_row"])); f_iks = set(map(str, zf["query_ik14"]))
    id_rows, id_iks = zi["query_row"], zi["query_ik14"].astype(str)
    unseen = np.array([(int(r) not in f_rows and k not in f_iks)
                       for r, k in zip(id_rows, id_iks)])
    uniq = np.array(sorted(set(id_iks)))
    half = set(uniq[RNG.permutation(len(uniq))[: len(uniq) // 2]])
    test_c = np.isin(id_iks, list(half))
    splits = {
        "A": ("formula_disjoint", np.ones(len(zf["query_row"]), bool),
              "identity_disjoint", unseen),
        "B": ("identity_disjoint", unseen,
              "formula_disjoint", np.ones(len(zf["query_row"]), bool)),
        "C": ("identity_disjoint", ~test_c, "identity_disjoint", test_c),
    }

    report = {"status": "GLM_UNIFIED_PAIR_RANKER_V0", "splits": {},
              "features": list(FEATURE_VIEWS) + ["D_frag", "D_nl", "D_diff"],
              "constraint": "NNLS non-negative coefficients"}
    for sname, (tr_p, tr_m, te_p, te_m) in splits.items():
        # ---- build train pair matrix -----------------------------------
        Xs, ys = [], []
        for q in np.flatnonzero(tr_m):
            built = build((tr_p, int(q)))
            if built is None:
                continue
            pool, cons, scores, qmz, ql = built
            X, pairs = pair_features(pool, cons, scores, qmz, ql)
            g = data[tr_p]["graph"]
            pos = int(g.query_ptr[q])
            for r, (a, b) in enumerate(pairs):
                if a == pos:
                    Xs.append(X[r]); ys.append(1.0)
                elif b == pos:
                    Xs.append(X[r]); ys.append(-1.0)
        Xtr = np.asarray(Xs); ytr = np.asarray(ys)
        w, _ = nnls(Xtr, ytr)
        print(f"[{sname}] train pairs {len(ytr):,}; NNLS weights "
              + ", ".join(f"{n}={round(float(x),4)}"
                          for n, x in zip(report["features"], w)), flush=True)

        # ---- evaluate on test ------------------------------------------
        g = data[te_p]["graph"]
        methods = data[te_p]["methods"]
        mol = data[te_p]["mol"]
        base = {}
        for v in ("weighted_spectral_entropy", "noise_v1",
                  "p2b_noise_v1_frozen"):
            base[v] = (mol[methods.index(v)] ,)
        correct_u, correct_b = [], []
        near_u, near_b = [], []
        for q in np.flatnonzero(te_m):
            built = build((te_p, int(q)))
            lo, hi = int(g.query_ptr[q]), int(g.query_ptr[q + 1])
            if built is None or hi <= lo:
                continue
            pool, cons, scores, qmz, ql = built
            X, pairs = pair_features(pool, cons, scores, qmz, ql)
            M = X @ w
            net = {m: 0.0 for m in pool}
            for (a, b), v in zip(pairs, M):
                net[a] += v
            pos = lo
            best_v = max(pool, key=lambda m: net[m])
            best_b = lo + int(np.argmax(
                base["weighted_spectral_entropy"][0][lo:hi]))
            correct_u.append(best_v == pos)
            correct_b.append(best_b == pos)
            if g.query_has_near[q]:
                near_u.append(best_v == pos)
                near_b.append(best_b == pos)
        cu = np.asarray(correct_u, bool); cb = np.asarray(correct_b, bool)
        nu = np.asarray(near_u, bool); nb = np.asarray(near_b, bool)
        idx = RNG.integers(0, len(cu), size=(BOOT_N, len(cu)))
        d = cu[idx].mean(1) - cb[idx].mean(1)
        lo_ci, hi_ci = np.percentile(d, [2.5, 97.5])
        corr = int((cu & ~cb).sum()); intro = int((~cu & cb).sum())
        report["splits"][sname] = {
            "unified_top1_pct": round(float(cu.mean() * 100), 2),
            "wse_top1_pct": round(float(cb.mean() * 100), 2),
            "delta_pp": round(float((cu.mean() - cb.mean()) * 100), 2),
            "ci95": [round(float(lo_ci * 100), 2), round(float(hi_ci * 100), 2)],
            "corrected_vs_wse": corr, "introduced_vs_wse": intro,
            "near_unified_pct": round(float(nu.mean() * 100), 2) if len(nu) else None,
            "near_wse_pct": round(float(nb.mean() * 100), 2) if len(nb) else None,
        }
        print(f"[{sname}] unified {cu.mean()*100:.2f}% vs WSE {cb.mean()*100:.2f}% "
              f"delta {(cu.mean()-cb.mean())*100:+.2f}pp CI [{lo_ci*100:+.2f},{hi_ci*100:+.2f}] "
              f"corr {corr} intro {intro} near {report['splits'][sname]['near_unified_pct']}"
              f" vs {report['splits'][sname]['near_wse_pct']}", flush=True)

    (OUT / "unified_ranker_v0.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    print("written:", OUT / "unified_ranker_v0.json")


if __name__ == "__main__":
    main()
