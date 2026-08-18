"""
Analyze WHY the peak correction helps and hurts (B step: understand before A).

Two questions, answered on the SAME query-clustered 5-fold CV and the SAME
peak-feature pipeline as eval_e0_peak_correction.py (numbers must match):

  Q1. Why does cos get a NEGATIVE coefficient in the combined LR?
      -> Ablation: cos-only vs top_overlap-only vs peak-only vs combined, plus
         the feature correlation matrix. If peak-only already ~= combined, cos is
         being "de-biased" out because raw peak overlap is the stronger signal.

  Q2. What are the 805 regressions (cos right -> peak wrong)? Is there a
      conditional rule that avoids them (e.g. "high cos + low top_overlap")?
      -> For every test query, record base rank (cos) and peak rank (LR), plus
         the true-IK's and the displacing wrong-IK's peak features. Classify the
         regression by the SIGN of the feature gap, never by a hand-picked
         threshold, then report the distribution.

Reuses load_artifacts / load_raw_spectra / precompute_peaks / pair_peak_features
from eval_e0_peak_correction.py so there is no divergent reimplementation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_e0_peak_correction import (
    DATA_PATH,
    FEATURE_NAMES,
    load_artifacts,
    load_raw_spectra,
    pair_peak_features,
    precompute_peaks,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "validation" / "e0_peak_correction"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fragment-tolerance", type=float, default=0.02)
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_feature_matrix(pi, pj, cos, mzs, ints, top_sets, n_pairs, tol):
    """Same X as eval_e0_peak_correction.py: [cos, n_matched, matched_frac,
    shared_intensity, top_overlap]."""
    X = np.empty((n_pairs, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, 0] = cos
    for k in range(n_pairs):
        nm, mf, si, tov, _ = pair_peak_features(mzs, ints, top_sets, int(pi[k]), int(pj[k]), tol)
        X[k, 1] = nm
        X[k, 2] = mf
        X[k, 3] = si
        X[k, 4] = tov
    return X


def rank_of(query_ik, ik_to_score):
    ranked = sorted(ik_to_score.items(), key=lambda kv: -kv[1])
    for pos, (ik, _) in enumerate(ranked, 1):
        if ik == query_ik:
            return pos
    return len(ranked) + 1


def query_ranks_and_details(pi, pj, qid, iks, cos, X, lr_scores, q):
    """Per-query: base rank (cos), peak rank (LR), and true/wrong winner features.

    lr_scores is per-PAIR (aligned to X / cos / qid). IK aggregation = max score
    per IK (max cos for base, max lr for peak), matching query_metrics.
    """
    idxs = np.where(qid == q)[0]
    c_iks = iks[pj[idxs]]
    c_cos = cos[idxs]
    c_lr = lr_scores[idxs]
    c_feat = X[idxs]
    q_ik = iks[pi[idxs[0]]]

    ik_to_cos = {}
    ik_to_lr = {}
    ik_to_feat = {}  # 5-feature row of the pair achieving max lr for that IK
    for k in range(len(idxs)):
        ik = c_iks[k]
        if ik not in ik_to_lr or c_lr[k] > ik_to_lr[ik]:
            ik_to_lr[ik] = c_lr[k]
            ik_to_feat[ik] = c_feat[k]
        if ik not in ik_to_cos or c_cos[k] > ik_to_cos[ik]:
            ik_to_cos[ik] = c_cos[k]

    base_rank = rank_of(q_ik, ik_to_cos)
    peak_rank = rank_of(q_ik, ik_to_lr)

    true_feat = ik_to_feat.get(q_ik)
    wrong_ik = None
    wrong_feat = None
    wrong_lr = -np.inf
    for ik, lr in ik_to_lr.items():
        if ik != q_ik and lr > wrong_lr:
            wrong_lr = lr
            wrong_ik = ik
            wrong_feat = ik_to_feat[ik]

    return {
        "q": int(q),
        "ik_q": q_ik,
        "base_rank": base_rank,
        "peak_rank": peak_rank,
        "true": true_feat,
        "wrong_ik": wrong_ik,
        "wrong": wrong_feat,
    }


def outcome(detail):
    base_ok = detail["base_rank"] == 1
    peak_ok = detail["peak_rank"] == 1
    if base_ok and not peak_ok:
        return "regress"
    if not base_ok and peak_ok:
        return "fix"
    if base_ok and peak_ok:
        return "both_ok"
    return "both_wrong"


def mechanism(detail):
    """Classify a regress/fix by the SIGN of the top_overlap gap, no thresholds."""
    t = detail["true"]
    w = detail["wrong"]
    if t is None or w is None:
        return "no_wrong_winner"
    t_topov = float(t[4])
    w_topov = float(w[4])
    t_shared = float(t[3])
    w_shared = float(w[3])
    if w_topov > t_topov:
        return "wrong_has_more_top_overlap"  # peak signal prefers a lookalike
    if t_topov >= w_topov and t_shared < w_shared:
        return "wrong_has_more_shared_intensity"
    return "marginal_flip"  # true overlap not worse -> cos barely won, LR tipped it


def main():
    args = parse_args()
    spectrum_ids, iks, pi, pj, labels, cos, qid = load_artifacts()
    n_pairs = len(labels)

    print("Loading raw spectra + precomputing peaks ...", flush=True)
    spec = load_raw_spectra(spectrum_ids, DATA_PATH)
    mzs, ints, top_sets, _ = precompute_peaks(spec)

    print("Building peak-feature matrix ...", flush=True)
    X = build_feature_matrix(pi, pj, cos, mzs, ints, top_sets, n_pairs, args.fragment_tolerance)

    # ── Q1a: feature correlation matrix (explains the negative cos coefficient) ──
    corr = np.corrcoef(X, rowvar=False)
    print("\nFeature correlation matrix:")
    print(f"  {'':16s}" + "".join(f"{n:>14s}" for n in FEATURE_NAMES))
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:16s}" + "".join(f"{corr[i, j]:14.3f}" for j in range(len(FEATURE_NAMES))))

    # ── Query-clustered CV: 4 scorers, same folds ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    uq_all = np.unique(qid)

    # scorer scores are per-pair, aligned to X / cos / qid
    scorers = {
        "cos_only": cos,                              # baseline (monotone)
        "topov_only": X[:, 4],                        # rank by top_overlap
    }
    pooled = {name: {"r1": [], "mrr": [], "r5": []} for name in
              ["cos_only", "topov_only", "peak_only", "combined"]}

    # peak-only / combined need fitting, so accumulate their per-fold test scores
    details = []  # per-query records for the COMBINED model (regression analysis)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(uq_all)):
        tr_queries = uq_all[tr_idx]
        te_queries = uq_all[te_idx]
        tr_mask = np.isin(qid, tr_queries)
        te_mask = np.isin(qid, te_queries)

        clf_peak = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr_mask][:, 1:5], labels[tr_mask])
        clf_combined = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr_mask], labels[tr_mask])
        peak_score = clf_peak.predict_proba(X[te_mask][:, 1:5])[:, 1]
        combined_score = clf_combined.predict_proba(X[te_mask])[:, 1]

        scorers["peak_only"] = np.zeros(n_pairs)
        scorers["peak_only"][te_mask] = peak_score
        scorers["combined"] = np.zeros(n_pairs)
        scorers["combined"][te_mask] = combined_score

        for name in ["cos_only", "topov_only", "peak_only", "combined"]:
            s = scorers[name]
            _, r1, mrr, r5 = _query_metrics(pi[te_mask], pj[te_mask], qid[te_mask], iks, s[te_mask])
            pooled[name]["r1"].append(float(r1.mean()))
            pooled[name]["mrr"].append(float(mrr.mean()))
            pooled[name]["r5"].append(float(r5.mean()))

        # per-query details for the combined model (test queries only)
        for q in te_queries:
            d = query_ranks_and_details(pi, pj, qid, iks, cos, X, scorers["combined"], q)
            d["fold"] = fold
            details.append(d)

    print("\nAblation (mean over folds):")
    for name in ["cos_only", "topov_only", "peak_only", "combined"]:
        r1 = np.mean(pooled[name]["r1"])
        mrr = np.mean(pooled[name]["mrr"])
        r5 = np.mean(pooled[name]["r5"])
        print(f"  {name:12s} R@1={r1:.4f}  MRR={mrr:.4f}  R@5={r5:.4f}")

    # ── Q2: classify every query outcome, summarize regressions vs fixes ──
    counts = defaultdict(int)
    mech_counts = defaultdict(int)
    mech_feature_gaps = defaultdict(list)  # true - wrong per mechanism
    regress_records = []
    fix_records = []

    for d in details:
        oc = outcome(d)
        counts[oc] += 1
        if oc in ("regress", "fix"):
            mech = mechanism(d)
            mech_counts[mech] += 1
            t = d["true"]
            w = d["wrong"]
            if t is not None and w is not None:
                gap = [float(t[i]) - float(w[i]) for i in range(len(FEATURE_NAMES))]
                mech_feature_gaps[mech].append(gap)
            rec = {
                "q": d["q"], "ik_q": d["ik_q"], "fold": d["fold"],
                "base_rank": d["base_rank"], "peak_rank": d["peak_rank"],
                "mechanism": mech,
                "true": [float(x) for x in t] if t is not None else None,
                "wrong_ik": d["wrong_ik"],
                "wrong": [float(x) for x in w] if w is not None else None,
            }
            (regress_records if oc == "regress" else fix_records).append(rec)

    print("\nOutcome counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k:12s} {v:,}")

    print("\nMechanism counts (regress + fix combined):")
    for k, v in sorted(mech_counts.items()):
        print(f"  {k:36s} {v:,}")

    print("\nMean feature gap (true - wrong) per mechanism:")
    for mech, gaps in mech_feature_gaps.items():
        g = np.array(gaps)
        gm = g.mean(axis=0)
        line = "  " + f"{mech:36s}" + "  ".join(
            f"{name}={gm[i]:+.3f}" for i, name in enumerate(FEATURE_NAMES))
        print(line)

    report = {
        "version": "0.1",
        "features": FEATURE_NAMES,
        "correlation": corr.tolist(),
        "ablation": {name: {k: float(np.mean(v)) for k, v in pooled[name].items()}
                     for name in pooled},
        "outcome_counts": {k: v for k, v in counts.items()},
        "mechanism_counts": {k: v for k, v in mech_counts.items()},
        "mean_feature_gap_by_mechanism": {
            mech: {name: float(np.array(gaps)[:, i].mean())
                   for i, name in enumerate(FEATURE_NAMES)}
            for mech, gaps in mech_feature_gaps.items()},
        "n_regressions": len(regress_records),
        "n_fixes": len(fix_records),
        "regressions": regress_records,
        "fixes": fix_records,
    }

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "peak_regression_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")


def _query_metrics(pair_i, pair_j, query_ids, iks, scores):
    """Same as query_metrics in eval_e0_peak_correction.py (kept local to avoid a
    cyclic import); returns (uq, r1, mrr, r5)."""
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


if __name__ == "__main__":
    main()
