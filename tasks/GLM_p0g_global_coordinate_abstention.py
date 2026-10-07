#!/usr/bin/env python
"""P0-G: Global coordinate dictionary with no-match abstention.

Uses the same frozen panels and score bundle as P0-L.
Splits identities into dictionary (70%) and held-out (30%).
Match queries: correct molecule IS in the panel's candidate set.
No-match queries: correct molecule is NOT → should ABSTAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
COVERAGES = (0.2, 0.4, 0.6, 0.8, 1.0)


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        while b := h.read(8 << 20):
            d.update(b)
    return d.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True,
                        help="Panel NPZ (panel_identity_disjoint.npz or panel_formula_disjoint.npz)")
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--score-key", type=str, required=True,
                        help="Key in score bundle (e.g. scores_identity_disjoint)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dict-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--methods", nargs="+",
                        default=["official_dreams", "noise_v1", "weighted_spectral_entropy"])
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    panel = load_npz(args.panel)
    bundle = load_npz(args.score_bundle)
    methods_list = [str(m) for m in bundle["method_names"]]
    scores = np.asarray(bundle[args.score_key], dtype=np.float64)  # (15, n_pairs)

    n_queries = len(panel["query_row"])
    query_ptr = panel["query_ptr"]
    molecule_ptr = panel["molecule_ptr"]
    molecule_label = panel["molecule_label"].astype(bool)
    query_ik = panel["query_ik14"].astype(str)
    print(f"panel: {n_queries} queries, {len(molecule_label)} molecules, "
          f"{scores.shape[1]} pairs", flush=True)

    # Build identity split (deterministic)
    unique_identities = sorted(set(query_ik))
    rng = np.random.default_rng(args.seed)
    n_dict = int(len(unique_identities) * args.dict_fraction)
    shuffled = rng.permutation(len(unique_identities))
    dict_ids = set(unique_identities[i] for i in shuffled[:n_dict])
    print(f"identities: {len(unique_identities)} total → {len(dict_ids)} dict, "
          f"{len(unique_identities) - len(dict_ids)} held-out", flush=True)

    # Classify queries
    is_match = np.array([q in dict_ids for q in query_ik])
    n_match = int(is_match.sum())
    n_nomatch = int((~is_match).sum())
    print(f"queries: {n_match} match + {n_nomatch} no-match", flush=True)

    results = {}
    for method in args.methods:
        if method not in methods_list:
            print(f"  skipping {method} (not in bundle)", flush=True)
            continue
        midx = methods_list.index(method)
        pair_scores = scores[midx]  # (n_pairs,)

        # Aggregate to molecule level: max score per molecule
        molecule_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

        # For each query: get its molecule candidates and their scores
        match_data = {"ranks": [], "max_scores": [], "positive_scores": []}
        nomatch_data = {"max_scores": []}

        for q in range(n_queries):
            lo, hi = query_ptr[q], query_ptr[q + 1]
            mol_scores = molecule_scores[lo:hi]
            mol_labels = molecule_label[lo:hi]

            if is_match[q]:
                pos = np.flatnonzero(mol_labels)
                if len(pos) == 1:
                    true_score = mol_scores[pos[0]]
                    neg = mol_scores[~mol_labels]
                    rank = 1 + int(np.sum(neg >= true_score))
                    match_data["ranks"].append(rank)
                    match_data["max_scores"].append(float(np.max(mol_scores)))
                    match_data["positive_scores"].append(float(true_score))
            else:
                nomatch_data["max_scores"].append(float(np.max(mol_scores)))

        ranks = np.array(match_data["ranks"])
        match_max = np.array(match_data["max_scores"])
        nomatch_max = np.array(nomatch_data["max_scores"])

        # Core metrics
        top1 = float(np.mean(ranks <= 1)) if len(ranks) > 0 else 0
        top5 = float(np.mean(ranks <= 5)) if len(ranks) > 0 else 0

        # Score separation (match max_score vs no-match max_score)
        sep_auc = None
        if len(match_max) >= 10 and len(nomatch_max) >= 10:
            u, _ = sps.mannwhitneyu(match_max, nomatch_max, alternative="greater")
            sep_auc = round(float(u / (len(match_max) * len(nomatch_max))), 4)

        # Coverage-error curve
        thresholds = np.linspace(0, 1, 201)
        cov, err = [], []
        for t in thresholds:
            c = np.mean(match_max > t) if len(match_max) > 0 else 0
            e = np.mean(nomatch_max > t) if len(nomatch_max) > 0 else 0
            cov.append(c)
            err.append(e)

        def best_cov_at_err(target):
            best = None
            for i, e in enumerate(err):
                if e <= target and cov[i] > 0:
                    if best is None or cov[i] > cov[best]:
                        best = i
            return (round(cov[best], 4), round(float(thresholds[best]), 4)) if best is not None else (None, None)

        cov_5, thr_5 = best_cov_at_err(0.05)
        cov_2, thr_2 = best_cov_at_err(0.02)

        result = {
            "match_queries": len(ranks),
            "no_match_queries": len(nomatch_max),
            "match_top1": round(top1, 4),
            "match_top5": round(top5, 4),
            "match_median_max_score": round(float(np.median(match_max)), 4) if len(match_max) > 0 else None,
            "no_match_median_max_score": round(float(np.median(nomatch_max)), 4) if len(nomatch_max) > 0 else None,
            "no_match_q95_max_score": round(float(np.quantile(nomatch_max, 0.95)), 4) if len(nomatch_max) > 0 else None,
            "score_separation_auc": sep_auc,
            "coverage_at_5pct_error": cov_5,
            "threshold_at_5pct_error": thr_5,
            "coverage_at_2pct_error": cov_2,
            "threshold_at_2pct_error": thr_2,
        }
        results[method] = result
        print(f"  {method}: R@1={top1:.4f} sep={sep_auc} "
              f"cov@2%err={cov_2} cov@5%err={cov_5}", flush=True)

    # Gates
    gates = {}
    for method, r in results.items():
        gates[f"{method}_top1_gt_50"] = r["match_top1"] > 0.50
        gates[f"{method}_separation_gt_65"] = (r["score_separation_auc"] or 0) > 0.65
        if r["coverage_at_2pct_error"] is not None:
            gates[f"{method}_cov_at_2pct_gt_30"] = r["coverage_at_2pct_error"] > 0.30

    all_pass = all(v for v in gates.values() if isinstance(v, bool))
    report = {
        "status": "P0_G_" + ("PASS" if all_pass else "STOP"),
        "dict_fraction": args.dict_fraction,
        "dict_identities": len(dict_ids),
        "heldout_identities": len(unique_identities) - len(dict_ids),
        "results": results,
        "gates": gates,
        "claim_limit": "P0-G: open-set coordinate mapping. Tests find-correct AND abstain-on-unknown.",
    }
    with open(args.output / "p0g_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nP0-G: {report['status']}", flush=True)
    print(json.dumps(results, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
