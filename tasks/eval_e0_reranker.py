"""
E0 Re-ranker: minimal multi-signal re-ranking over frozen DreaMS embeddings.

Reads the pair-level feature table already dumped by eval_e0_baseline.py
(e0_pair_arrays.npz + e0_manifest.json) and fits a lightweight logistic
regression that fuses DreaMS cosine with cheap orthogonal metadata:
  - precursor mass error (ppm)
  - collision-energy difference (eV)
  - instrument match (Orbitrap / QTOF / nan)

Goal: test at near-zero cost whether the residual Top-1 errors can be
recovered by signals DreaMS does not observe, without touching the embedding.

Protocol (matches E0 query-level evaluation):
  - Query-clustered 5-fold CV: pairs of one query are never split across folds,
    so the same molecule/spectrum cannot leak train -> test.
  - Molecule-level aggregation by 14-char IK (max score per IK), then rank IKs.
  - Baseline = DreaMS cosine alone; Re-ranker = logistic regression on all
    features. Both are evaluated identically on the held-out test queries.

Usage:
  python tasks/eval_e0_reranker.py --dry-run   # 20k pairs / 2k queries smoke
  python tasks/eval_e0_reranker.py             # full (5-fold CV)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
E0_DIR = REPO_ROOT / 'data' / 'validation' / 'e0_baseline'
OUT_DIR = REPO_ROOT / 'data' / 'validation' / 'e0_reranker'

FEATURE_NAMES = ['cos', 'ppm_err', 'ce_diff', 'ce_missing', 'inst_match']


def parse_args():
    p = argparse.ArgumentParser(description='E0 minimal re-ranker over frozen embeddings')
    p.add_argument('--dry-run', action='store_true',
                   help='Subsample to ~20k pairs / 2k queries for a quick check')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--output-dir', type=str, default=str(OUT_DIR))
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_artifacts():
    with open(E0_DIR / 'e0_manifest.json') as f:
        manifest = json.load(f)
    pmz = np.array([float(m['precursor_mz']) for m in manifest], dtype=np.float64)
    iks = np.array([m['inchikey_14'] for m in manifest])
    ce = np.array([float(m.get('ce', 0.0)) for m in manifest], dtype=np.float64)
    inst = np.array([str(m.get('instrument', '')) for m in manifest])

    z = np.load(E0_DIR / 'e0_pair_arrays.npz')
    pi = z['primary__pair_i'].astype(np.int64)
    pj = z['primary__pair_j'].astype(np.int64)
    labels = z['primary__labels'].astype(np.int8)
    cos = z['primary__scores'].astype(np.float64)
    qid = z['primary__query_ids'].astype(np.int64)
    return pi, pj, labels, cos, qid, pmz, iks, ce, inst


def build_features(pi, pj, cos, pmz, ce, inst):
    ppm_err = np.abs(pmz[pi] - pmz[pj]) / pmz[pi] * 1e6
    ce_missing = (np.isnan(ce[pi]) | np.isnan(ce[pj])).astype(np.float64)
    ce_i = np.nan_to_num(ce[pi], nan=0.0)
    ce_j = np.nan_to_num(ce[pj], nan=0.0)
    ce_diff = np.abs(ce_i - ce_j)
    inst_match = ((inst[pi] == inst[pj]) & (inst[pi] != 'nan')).astype(np.float64)
    return np.stack([cos, ppm_err, ce_diff, ce_missing, inst_match], axis=1)


def query_metrics(pair_i, pair_j, query_ids, iks, scores):
    """Per-query R@1 / MRR / R@5 after molecule (IK) aggregation.

    Returns (unique_query_ids, r1, mrr, r5) with the metric arrays aligned to
    the sorted unique query ids. Uses the full `iks` array; pair_i/pair_j are
    global embedding indices, so they index `iks` regardless of any pair mask.
    """
    groups = defaultdict(list)
    for k, q in enumerate(query_ids):
        groups[q].append(k)
    uq = np.array(sorted(groups.keys()), dtype=np.int64)
    r1 = np.empty(len(uq), dtype=np.float64)
    mrr = np.empty(len(uq), dtype=np.float64)
    r5 = np.empty(len(uq), dtype=np.float64)
    for qi, q in enumerate(uq):
        idxs = groups[int(q)]
        q_ik = iks[pair_i[idxs[0]]]
        c_iks = iks[pair_j[idxs]]
        c_scores = scores[idxs]
        ik2max = {}
        for ik, s in zip(c_iks, c_scores):
            cur = ik2max.get(ik)
            if cur is None or s > cur:
                ik2max[ik] = s
        ranked = sorted(ik2max.items(), key=lambda kv: -kv[1])
        rank = len(ranked) + 1
        for rpos, (ik, _) in enumerate(ranked, 1):
            if ik == q_ik:
                rank = rpos
                break
        r1[qi] = float(rank <= 1)
        mrr[qi] = 1.0 / rank
        r5[qi] = float(rank <= 5)
    return uq, r1, mrr, r5


def main():
    args = parse_args()
    pi, pj, labels, cos, qid, pmz, iks, ce, inst = load_artifacts()
    X = build_features(pi, pj, cos, pmz, ce, inst)

    if args.dry_run:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(labels), 20000, replace=False)
        pi, pj, labels, cos, qid, X = (a[keep] for a in (pi, pj, labels, cos, qid, X))
        uq_sub = np.unique(qid)[:2000]
        m = np.isin(qid, uq_sub)
        pi, pj, labels, cos, qid, X = (a[m] for a in (pi, pj, labels, cos, qid, X))
        print(f'DRY RUN: {len(labels):,} pairs, {len(np.unique(qid)):,} queries')

    n_queries = len(np.unique(qid))
    print(f'Pairs: {len(labels):,}  |  Queries: {n_queries:,}')
    print(f'Positive rate: {float(labels.mean()):.4f}')

    # ── Baseline per-query metrics (cosine alone), computed once ──
    buq, br1, bmrr, br5 = query_metrics(pi, pj, qid, iks, cos)
    base_map = {int(q): (r1, mrr, r5)
                for q, r1, mrr, r5 in zip(buq, br1, bmrr, br5)}
    print(f'\nBaseline (cos only):  R@1={br1.mean():.4f}  MRR={bmrr.mean():.4f}  R@5={br5.mean():.4f}')

    # ── Feature diagnostics ──
    from sklearn import metrics
    feat_auc = {}
    print('\nFeature diagnostics (pair-level):')
    for i, name in enumerate(FEATURE_NAMES):
        auc = float(metrics.roc_auc_score(labels, X[:, i])) if len(np.unique(labels)) == 2 else 0.5
        feat_auc[name] = auc
        pos_mean = X[labels == 1, i].mean()
        neg_mean = X[labels == 0, i].mean()
        print(f'  {name:12s} AUC={auc:.4f}  pos_mean={pos_mean:.4f}  neg_mean={neg_mean:.4f}')

    # ── Query-clustered CV ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    uq_all = np.unique(qid)

    folds = []
    pooled_r1, pooled_mrr, pooled_r5 = [], [], []
    pooled_b_r1, pooled_b_mrr, pooled_b_r5 = [], [], []
    coef_sum = np.zeros(X.shape[1])

    for fold, (tr_idx, te_idx) in enumerate(kf.split(uq_all)):
        tr_queries = uq_all[tr_idx]
        te_queries = uq_all[te_idx]
        tr_mask = np.isin(qid, tr_queries)
        te_mask = np.isin(qid, te_queries)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr_mask], labels[tr_mask])
        te_score = clf.predict_proba(X[te_mask])[:, 1]
        coef_sum += clf.coef_[0]

        b_r1 = np.mean([base_map[int(q)][0] for q in te_queries])
        b_mrr = np.mean([base_map[int(q)][1] for q in te_queries])
        b_r5 = np.mean([base_map[int(q)][2] for q in te_queries])

        te_uq, rr1, rmrr, rr5 = query_metrics(pi[te_mask], pj[te_mask], qid[te_mask], iks, te_score)
        r_r1 = rr1.mean()
        r_mrr = rmrr.mean()
        r_r5 = rr5.mean()
        base_r1_arr = np.array([base_map[int(q)][0] for q in te_uq])
        n_fix = int(((base_r1_arr == 0.0) & (rr1 == 1.0)).sum())
        n_reg = int(((base_r1_arr == 1.0) & (rr1 == 0.0)).sum())

        pooled_b_r1.append(b_r1); pooled_b_mrr.append(b_mrr); pooled_b_r5.append(b_r5)
        pooled_r1.append(r_r1); pooled_mrr.append(r_mrr); pooled_r5.append(r_r5)

        folds.append({
            'fold': fold,
            'n_train_queries': int(len(tr_queries)),
            'n_test_queries': int(len(te_queries)),
            'baseline': {'r1': b_r1, 'mrr': b_mrr, 'r5': b_r5},
            'reranker': {'r1': r_r1, 'mrr': r_mrr, 'r5': r_r5},
            'delta': {'r1': r_r1 - b_r1, 'mrr': r_mrr - b_mrr, 'r5': r_r5 - b_r5},
            'n_fixed': n_fix,
            'n_regressed': n_reg,
        })
        print(f'  fold {fold}: base R@1={b_r1:.4f} -> rerank R@1={r_r1:.4f} '
              f'(Δ={r_r1 - b_r1:+.4f})  MRR Δ={r_mrr - b_mrr:+.4f}  '
              f'fix={n_fix} reg={n_reg}')

    mean_delta = {
        'r1': float(np.mean([f['delta']['r1'] for f in folds])),
        'mrr': float(np.mean([f['delta']['mrr'] for f in folds])),
        'r5': float(np.mean([f['delta']['r5'] for f in folds])),
    }
    total_fix = int(sum(f['n_fixed'] for f in folds))
    total_reg = int(sum(f['n_regressed'] for f in folds))
    report = {
        'version': '0.1',
        'features': FEATURE_NAMES,
        'model': 'LogisticRegression(C=1.0, max_iter=1000)',
        'n_pairs': int(len(labels)),
        'n_queries': int(n_queries),
        'positive_rate': float(labels.mean()),
        'baseline_overall': {'r1': float(br1.mean()), 'mrr': float(bmrr.mean()),
                             'r5': float(br5.mean())},
        'feature_auc': feat_auc,
        'mean_coefficients': {name: float(c) for name, c in zip(FEATURE_NAMES, coef_sum / args.folds)},
        'mean_delta_vs_baseline': mean_delta,
        'total_fixed': total_fix,
        'total_regressed': total_reg,
        'folds': folds,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir and Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / 'reranker_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 70)
    print(f'Mean delta vs cosine-only baseline (over {args.folds} folds):')
    print(f'  R@1 : {mean_delta["r1"]:+.4f}')
    print(f'  MRR : {mean_delta["mrr"]:+.4f}')
    print(f'  R@5 : {mean_delta["r5"]:+.4f}')
    print(f'  Top-1 fixed / regressed: {total_fix} / {total_reg}')
    print(f'\nMean LR coefficients:')
    for name, c in zip(FEATURE_NAMES, coef_sum / args.folds):
        print(f'  {name:12s} {c:+.4f}')
    print(f'\nReport: {report_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
