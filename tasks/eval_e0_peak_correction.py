"""
E0 peak-level correction: test whether the DIRECT peak-overlap signal fixes
DreaMS retrieval errors, reusing the exact peak interfaces already validated in
train_causal_chemmask_head.py (raw_peaks / greedy_peak_matches / shared_major_score).

Why this is more systematic than the metadata re-ranker (eval_e0_reranker.py):

  1. It reuses the codebase's OWN peak-matching interfaces instead of inventing
     new logic. `shared_major_score` is the exact score whose peak-deletion
     experiments produced the error atlas (shared-fragment -> FP, instrument -> FN),
     so the correction signal is the DIRECT causal one, not an indirect proxy.

  2. It maps every candidate pair back to its RAW spectrum via
     manifest['spectrum_id'] -> HDF5 IDENTIFIER -> row index, so peak features are
     computed on the real spectra (the same (2,128) [m/z, intensity] arrays the
     error atlas used), not on coarse metadata.

  3. It reports a transparent diagnostic BEFORE fitting anything: per-label feature
     means, feature AUC, and the FP/FN signatures implied by the error atlas. The
     correction is therefore readable, not a black-box sklearn number.

  4. It uses the SAME query-clustered 5-fold CV + molecule (IK14) aggregation as
     the E0 baseline, so the comparison is apples-to-apples and leakage-free.

What this does NOT claim: any guaranteed improvement. It MEASURES whether the peak
signal corrects more errors than it introduces; the printed numbers decide.

Peak features per candidate pair (matching tolerance = --fragment-tolerance Da):
  - n_matched        : number of m/z-matched peaks (two-pointer greedy match)
  - matched_frac     : n_matched / min(n_peaks_a, n_peaks_b)
  - shared_intensity : min(fraction of total intensity in matched peaks, over both)
  - top_overlap      : fraction of the top-10 peaks (by intensity) that match
  - shared_major     : shared_intensity + top_overlap  (== shared_major_score return)

Why the decomposition matters (the error atlas made quantitative):
  - FP (shared-fragment) error: label=0, HIGH shared_intensity but LOW top_overlap.
    Two different molecules share one big fragment -> cos is high but top peaks differ.
  - FN (instrument/CE) error: label=1, LOW shared_intensity / matched_frac. Same
    molecule measured on a different instrument -> few peaks line up -> cos is low.
  The metadata re-ranker could not express either of these.

Usage:
  python tasks/eval_e0_peak_correction.py --dry-run       # ~20k pairs / 2k queries smoke
  python tasks/eval_e0_peak_correction.py                  # full (all pairs, 5-fold CV)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

# Reuse the exact, already-validated peak interfaces (this is the point).
from train_causal_chemmask_head import (
    greedy_peak_matches,
    raw_peaks,
    shared_major_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
E0_DIR = REPO_ROOT / "data" / "validation" / "e0_baseline"
OUT_DIR = REPO_ROOT / "data" / "validation" / "e0_peak_correction"
DATA_PATH = REPO_ROOT / "data" / "models" / "MassSpecGym_MurckoHist_split.hdf5"

# cos + the four decomposed peak features fed to the logistic re-ranker.
FEATURE_NAMES = ["cos", "n_matched", "matched_frac", "shared_intensity", "top_overlap"]


def decode_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Subsample to ~20k pairs / 2k queries for a quick smoke check")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fragment-tolerance", type=float, default=0.02,
                   help="m/z matching tolerance in Da (same default as the causal head)")
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-self-check", action="store_true",
                   help="Skip the shared_major_score equivalence self-check")
    return p.parse_args()


def load_artifacts():
    """Read the E0 pair feature table already dumped by eval_e0_baseline.py."""
    with open(E0_DIR / "e0_manifest.json") as f:
        manifest = json.load(f)
    spectrum_ids = [m["spectrum_id"] for m in manifest]
    iks = np.array([m["inchikey_14"] for m in manifest])

    z = np.load(E0_DIR / "e0_pair_arrays.npz")
    pi = z["primary__pair_i"].astype(np.int64)
    pj = z["primary__pair_j"].astype(np.int64)
    labels = z["primary__labels"].astype(np.int8)
    cos = z["primary__scores"].astype(np.float64)
    qid = z["primary__query_ids"].astype(np.int64)
    return spectrum_ids, iks, pi, pj, labels, cos, qid


def load_raw_spectra(spectrum_ids, data_path):
    """Map manifest spectrum_id -> HDF5 IDENTIFIER row and load raw (2,128) spectra."""
    with h5py.File(data_path, "r") as f:
        ids = [decode_bytes(x) for x in f["IDENTIFIER"][:]]
        id_to_row = {sid: row for row, sid in enumerate(ids)}
        rows = np.array([id_to_row[s] for s in spectrum_ids], dtype=np.int64)
        spec = f["spectrum"][rows].astype(np.float64)  # (N, 2, 128): [m/z, intensity]
    return spec


def precompute_peaks(spec):
    """Run raw_peaks() once per spectrum; cache sorted m/z, intensity, top-10 sets."""
    mzs, ints, top_sets, n_peaks = [], [], [], []
    for k in range(spec.shape[0]):
        mz, intensity, _ = raw_peaks(spec[k])
        mzs.append(mz)
        ints.append(intensity)
        n_peaks.append(len(mz))
        if len(intensity):
            top_sets.append(set(np.argsort(intensity)[-min(10, len(intensity)):].tolist()))
        else:
            top_sets.append(set())
    return mzs, ints, top_sets, n_peaks


def pair_peak_features(mzs, ints, top_sets, i, j, tol):
    """Return (n_matched, matched_frac, shared_intensity, top_overlap, shared_major).

    shared_major here == shared_major_score(spec[i], spec[j], tol): both are
    min(shared_a, shared_b) + top_fraction computed from the SAME greedy matches.
    We recompute it from precomputed raw_peaks so 1.7M pairs do not re-parse the
    (2,128) array each time; the self-check proves the equivalence.
    """
    mz_a, mz_b = mzs[i], mzs[j]
    int_a, int_b = ints[i], ints[j]
    n_a, n_b = len(mz_a), len(mz_b)
    if n_a == 0 or n_b == 0:
        return 0, 0.0, 0.0, 0.0, -1.0
    matches = greedy_peak_matches(mz_a, mz_b, tol)
    n_matched = len(matches)
    matched_frac = n_matched / max(min(n_a, n_b), 1)
    if not matches:
        return 0, 0.0, 0.0, 0.0, 0.0
    matched_a = np.asarray([a for a, _ in matches], dtype=int)
    matched_b = np.asarray([b for _, b in matches], dtype=int)
    shared_a = float(int_a[matched_a].sum() / max(int_a.sum(), 1e-12))
    shared_b = float(int_b[matched_b].sum() / max(int_b.sum(), 1e-12))
    shared_intensity = min(shared_a, shared_b)
    top_a, top_b = top_sets[i], top_sets[j]
    top_matches = sum(a in top_a and b in top_b for a, b in matches)
    top_overlap = top_matches / max(min(10, n_a, n_b), 1)
    shared_major = shared_intensity + top_overlap
    return n_matched, matched_frac, shared_intensity, top_overlap, shared_major


def self_check(spec, mzs, ints, top_sets, rows, tol):
    """Prove pair_peak_features.shared_major == shared_major_score on a sample."""
    max_err = 0.0
    for i, j in rows:
        _, _, _, _, mine = pair_peak_features(mzs, ints, top_sets, i, j, tol)
        ref = shared_major_score(spec[i], spec[j], tol)
        max_err = max(max_err, abs(mine - ref))
    return max_err


def query_metrics(pair_i, pair_j, query_ids, iks, scores):
    """Per-query R@1 / MRR / R@5 after molecule (IK14) aggregation.

    Identical protocol to eval_e0_baseline.py / eval_e0_reranker.py: aggregate
    candidate scores by max over 14-char IK, then rank IKs; pair_i/pair_j are
    global embedding indices and index `iks` regardless of any pair mask.
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
    spectrum_ids, iks, pi, pj, labels, cos, qid = load_artifacts()

    if args.dry_run:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(labels), 20000, replace=False)
        pi, pj, labels, cos, qid = (a[keep] for a in (pi, pj, labels, cos, qid))
        uq_sub = np.unique(qid)[:2000]
        m = np.isin(qid, uq_sub)
        pi, pj, labels, cos, qid = (a[m] for a in (pi, pj, labels, cos, qid))
        print(f"DRY RUN: {len(labels):,} pairs, {len(np.unique(qid)):,} queries")

    n_pairs = len(labels)
    n_queries = len(np.unique(qid))
    print(f"Pairs: {n_pairs:,}  |  Queries: {n_queries:,}  |  tol={args.fragment_tolerance:.3f} Da")
    print(f"Positive rate: {float(labels.mean()):.4f}")

    # ── Raw spectra + peak feature table ──
    print("Loading raw spectra + precomputing peaks ...", flush=True)
    spec = load_raw_spectra(spectrum_ids, DATA_PATH)
    mzs, ints, top_sets, _ = precompute_peaks(spec)

    # ── Self-check: our shared_major reproduces shared_major_score exactly ──
    if not args.skip_self_check:
        rng = np.random.RandomState(args.seed)
        check_rows = [(int(pi[k]), int(pj[k])) for k in rng.choice(n_pairs, 200, replace=False)]
        max_err = self_check(spec, mzs, ints, top_sets, check_rows, args.fragment_tolerance)
        print(f"Self-check: max |recomputed shared_major - shared_major_score| = {max_err:.3e}")
        if max_err > 1e-9:
            raise RuntimeError("Peak-feature recomputation diverged from shared_major_score — aborting.")

    # ── Feature matrix (loop; greedy match dominates runtime) ──
    print("Computing peak features for all pairs ...", flush=True)
    X = np.empty((n_pairs, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, 0] = cos
    for k in range(n_pairs):
        nm, mf, si, tov, _ = pair_peak_features(mzs, ints, top_sets, int(pi[k]), int(pj[k]),
                                                args.fragment_tolerance)
        X[k, 1] = nm
        X[k, 2] = mf
        X[k, 3] = si
        X[k, 4] = tov
        if (k + 1) % 500000 == 0:
            print(f"  {k + 1:,}/{n_pairs:,} pairs", flush=True)

    # ── Transparent diagnostic BEFORE fitting (read the signal, don't hide it) ──
    from sklearn import metrics

    pos = labels == 1
    neg = labels == 0
    print("\nPeak-feature diagnostic (pair-level):")
    print(f"  {'feature':16s} {'AUC':>7s} {'pos_mean':>9s} {'neg_mean':>9s}")
    diag = {}
    for c, name in enumerate(FEATURE_NAMES):
        col = X[:, c]
        if len(np.unique(labels)) == 2:
            auc = float(metrics.roc_auc_score(labels, col))
        else:
            auc = float("nan")
        pm = float(col[pos].mean()) if pos.any() else float("nan")
        nm_ = float(col[neg].mean()) if neg.any() else float("nan")
        diag[name] = {"auc": auc, "pos_mean": pm, "neg_mean": nm_}
        print(f"  {name:16s} {auc:7.4f} {pm:9.4f} {nm_:9.4f}")

    # FP signature: among WRONG candidates, do high-cos ones share more peak mass?
    # FN signature: among CORRECT candidates, do low-cos ones share fewer peaks?
    print("\nError-atlas signatures:")
    cos_col = X[:, 0]
    si_col = X[:, 3]
    tov_col = X[:, 4]
    mf_col = X[:, 2]
    if neg.any() and neg.sum() > 6:
        neg_cos = cos_col[neg]
        lo = np.percentile(neg_cos, 33); hi = np.percentile(neg_cos, 67)
        lo_m = neg & (cos_col <= lo); hi_m = neg & (cos_col >= hi)
        print(f"  FP (label=0): low-cos wrong {si_col[lo_m].mean():.3f} shared_int / "
              f"{tov_col[lo_m].mean():.3f} top_ov  vs  high-cos wrong "
              f"{si_col[hi_m].mean():.3f} shared_int / {tov_col[hi_m].mean():.3f} top_ov")
    if pos.any() and pos.sum() > 6:
        pos_cos = cos_col[pos]
        lo = np.percentile(pos_cos, 33); hi = np.percentile(pos_cos, 67)
        lo_m = pos & (cos_col <= lo); hi_m = pos & (cos_col >= hi)
        print(f"  FN (label=1): low-cos correct {mf_col[lo_m].mean():.3f} matched_frac / "
              f"{si_col[lo_m].mean():.3f} shared_int  vs  high-cos correct "
              f"{mf_col[hi_m].mean():.3f} matched_frac / {si_col[hi_m].mean():.3f} shared_int")

    # ── Baseline per-query metrics (cosine alone) ──
    buq, br1, bmrr, br5 = query_metrics(pi, pj, qid, iks, cos)
    base_map = {int(q): (r1, mrr, r5) for q, r1, mrr, r5 in zip(buq, br1, bmrr, br5)}
    print(f"\nBaseline (cos only):  R@1={br1.mean():.4f}  MRR={bmrr.mean():.4f}  R@5={br5.mean():.4f}")

    # ── Query-clustered CV: cos vs cos+peak-features ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    uq_all = np.unique(qid)

    pooled_r1, pooled_mrr, pooled_r5 = [], [], []
    pooled_b_r1, pooled_b_mrr, pooled_b_r5 = [], [], []
    coef_sum = np.zeros(X.shape[1])
    folds = []

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
        r_r1 = rr1.mean(); r_mrr = rmrr.mean(); r_r5 = rr5.mean()
        base_r1_arr = np.array([base_map[int(q)][0] for q in te_uq])
        n_fix = int(((base_r1_arr == 0.0) & (rr1 == 1.0)).sum())
        n_reg = int(((base_r1_arr == 1.0) & (rr1 == 0.0)).sum())

        pooled_b_r1.append(b_r1); pooled_b_mrr.append(b_mrr); pooled_b_r5.append(b_r5)
        pooled_r1.append(r_r1); pooled_mrr.append(r_mrr); pooled_r5.append(r_r5)
        folds.append({
            "fold": fold,
            "n_train_queries": int(len(tr_queries)),
            "n_test_queries": int(len(te_queries)),
            "baseline": {"r1": b_r1, "mrr": b_mrr, "r5": b_r5},
            "reranker": {"r1": r_r1, "mrr": r_mrr, "r5": r_r5},
            "delta": {"r1": r_r1 - b_r1, "mrr": r_mrr - b_mrr, "r5": r_r5 - b_r5},
            "n_fixed": n_fix, "n_regressed": n_reg,
        })
        print(f"  fold {fold}: base R@1={b_r1:.4f} -> peak R@1={r_r1:.4f} "
              f"(Δ={r_r1 - b_r1:+.4f})  MRR Δ={r_mrr - b_mrr:+.4f}  fix={n_fix} reg={n_reg}")

    mean_delta = {
        "r1": float(np.mean([f["delta"]["r1"] for f in folds])),
        "mrr": float(np.mean([f["delta"]["mrr"] for f in folds])),
        "r5": float(np.mean([f["delta"]["r5"] for f in folds])),
    }
    total_fix = int(sum(f["n_fixed"] for f in folds))
    total_reg = int(sum(f["n_regressed"] for f in folds))

    report = {
        "version": "0.1",
        "features": FEATURE_NAMES,
        "fragment_tolerance": args.fragment_tolerance,
        "model": "LogisticRegression(C=1.0, max_iter=1000) on cos + 4 peak features",
        "n_pairs": int(n_pairs),
        "n_queries": int(n_queries),
        "positive_rate": float(labels.mean()),
        "baseline_overall": {"r1": float(br1.mean()), "mrr": float(bmrr.mean()),
                             "r5": float(br5.mean())},
        "feature_diagnostic": diag,
        "mean_coefficients": {name: float(c) for name, c in zip(FEATURE_NAMES, coef_sum / args.folds)},
        "mean_delta_vs_baseline": mean_delta,
        "total_fixed": total_fix,
        "total_regressed": total_reg,
        "folds": folds,
    }

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "peak_correction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Mean delta vs cosine-only baseline (over {args.folds} folds):")
    print(f"  R@1 : {mean_delta['r1']:+.4f}")
    print(f"  MRR : {mean_delta['mrr']:+.4f}")
    print(f"  R@5 : {mean_delta['r5']:+.4f}")
    print(f"  Top-1 fixed / regressed: {total_fix} / {total_reg}")
    print(f"\nMean LR coefficients:")
    for name, c in zip(FEATURE_NAMES, coef_sum / args.folds):
        print(f"  {name:16s} {c:+.4f}")
    print(f"\nReport: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
