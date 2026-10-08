#!/usr/bin/env python
"""Unified entity recovery evaluation (GPT-recommended design).

Core question: at the same error risk, how many MORE correct entities can
our unified system recover compared to each individual method?

This is NOT just retrieval accuracy — it measures the practical value of
coordinate mapping: entity recovery at calibrated confidence.

Design:
  1. Frozen identity split (70% dictionary, 30% held-out as no-match)
  2. For each method: compute molecule-level scores from frozen bundle
  3. For match queries: rank + correctness
  4. For no-match queries: REMOVE positive, then max score = false assign risk
  5. Sweep margin thresholds → coverage vs error tradeoff
  6. Primary metric: coverage at matched FDR level

Uses CORRECTED entropy similarity and proper no-match handling.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--score-key", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dict-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--fdr-targets", nargs="+", type=float,
                        default=[0.02, 0.05, 0.10],
                        help="Target FDR levels for coverage comparison")
    parser.add_argument("--methods", nargs="+",
                        default=["official_dreams", "noise_v1",
                                 "weighted_spectral_entropy",
                                 "p2b_noise_v1_frozen",
                                 "spec2vec_gnps_public"])
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    panel = load_npz(args.panel)
    bundle = load_npz(args.score_bundle)
    methods_list = [str(m) for m in bundle["method_names"]]
    scores = np.asarray(bundle[args.score_key], dtype=np.float64)

    n_queries = len(panel["query_row"])
    query_ptr = panel["query_ptr"]
    molecule_ptr = panel["molecule_ptr"]
    molecule_label = panel["molecule_label"].astype(bool)
    query_ik = panel["query_ik14"].astype(str)
    near = panel["near_query"].astype(bool)

    # Identity split
    identities = sorted(set(query_ik))
    rng = np.random.default_rng(args.seed)
    n_dict = int(len(identities) * args.dict_fraction)
    perm = rng.permutation(len(identities))
    dict_ids = set(identities[i] for i in perm[:n_dict])
    is_match = np.array([q in dict_ids for q in query_ik])
    print(f"{int(is_match.sum())} match + {int((~is_match).sum())} no-match", flush=True)

    results = {}
    for method in args.methods:
        if method not in methods_list:
            continue
        midx = methods_list.index(method)
        pair_scores = scores[midx]
        mol_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

        # For each query: top1, top2, margin, correctness
        # Match: normal ranking (positive present)
        # No-match: REMOVE positive, then top1/top2 among wrong candidates
        data = {"match": [], "nomatch": []}
        for q in range(n_queries):
            lo, hi = query_ptr[q], query_ptr[q+1]
            ms = mol_scores[lo:hi].copy()
            ml = molecule_label[lo:hi]

            if is_match[q]:
                pos = np.flatnonzero(ml)
                if len(pos) != 1:
                    continue
                true_score = ms[pos[0]]
                neg_scores = ms[~ml]
                rank = 1 + int(np.sum(neg_scores >= true_score))
                sorted_all = np.sort(ms)
                margin = float(sorted_all[-1] - sorted_all[-2]) if len(ms) >= 2 else 1.0
                data["match"].append({
                    "rank": rank, "correct": rank == 1,
                    "margin": margin,
                    "top1": float(np.max(ms)),
                })
            else:
                # PROPER no-match: remove positive from scores
                ms_nomatch = ms.copy()
                ms_nomatch[ml] = -np.inf
                valid = ms_nomatch[~ml]
                if len(valid) < 2:
                    margin = 0.0
                    top1 = float(np.max(valid)) if len(valid) > 0 else 0.0
                else:
                    sorted_valid = np.sort(valid)
                    top1 = float(sorted_valid[-1])
                    margin = float(sorted_valid[-1] - sorted_valid[-2])
                data["nomatch"].append({"margin": margin, "top1": top1})

        match_data = data["match"]
        nomatch_data = data["nomatch"]
        match_margins = np.array([d["margin"] for d in match_data])
        match_correct = np.array([d["correct"] for d in match_data])
        nomatch_margins = np.array([d["margin"] for d in nomatch_data])

        # Score separation
        sep_auc = None
        if len(match_margins) >= 10 and len(nomatch_margins) >= 10:
            u, p = sps.mannwhitneyu(match_margins, nomatch_margins, alternative="greater")
            sep_auc = round(float(u / (len(match_margins) * len(nomatch_margins))), 4)

        # Coverage-error curves (margin-based, proper mixed population)
        thresholds = np.linspace(0, 1, 201)
        curves = []
        for t in thresholds:
            # Match queries assigned
            m_assigned = match_margins > t
            n_m = int(m_assigned.sum())
            # Of assigned, how many correct?
            precision = float(np.mean(match_correct[m_assigned])) if n_m > 0 else 0
            # No-match queries falsely assigned
            n_f = int(np.sum(nomatch_margins > t)) if len(nomatch_margins) > 0 else 0
            # Total assigned = correct + incorrect + false
            n_total = n_m + n_f
            # FDR = (false + incorrect) / total
            n_incorrect = n_m - int(np.sum(match_correct[m_assigned]))
            fdr = (n_f + n_incorrect) / max(n_total, 1) if n_total > 0 else 0
            # Coverage = fraction of ALL match queries correctly assigned
            coverage = int(np.sum(match_correct[m_assigned])) / max(len(match_data), 1)
            curves.append({
                "threshold": round(float(t), 4),
                "coverage": round(coverage, 4),
                "fdr": round(fdr, 4),
                "n_assigned": n_total,
                "n_correct": int(np.sum(match_correct[m_assigned])),
                "n_false_nomatch": n_f,
                "n_incorrect_match": n_incorrect,
            })

        # Coverage at each FDR target
        cov_at_fdr = {}
        for target in args.fdr_targets:
            best = None
            for c in curves:
                if c["fdr"] <= target and c["coverage"] > 0:
                    if best is None or c["coverage"] > best["coverage"]:
                        best = c
            if best:
                cov_at_fdr[str(target)] = {
                    "coverage": best["coverage"],
                    "threshold": best["threshold"],
                    "n_correct": best["n_correct"],
                    "fdr": best["fdr"],
                }
            else:
                cov_at_fdr[str(target)] = {"coverage": 0}

        # Near-structure subset
        near_match_margins = []
        near_match_correct = []
        for q in range(n_queries):
            if not near[q] or not is_match[q]:
                continue
            lo, hi = query_ptr[q], query_ptr[q+1]
            ms = mol_scores[lo:hi]
            ml = molecule_label[lo:hi]
            pos = np.flatnonzero(ml)
            if len(pos) != 1:
                continue
            neg = ms[~ml]
            rank = 1 + int(np.sum(neg >= ms[pos[0]]))
            s = np.sort(ms)
            margin = float(s[-1] - s[-2]) if len(ms) >= 2 else 1.0
            near_match_margins.append(margin)
            near_match_correct.append(rank == 1)
        near_top1 = float(np.mean(near_match_correct)) if near_match_correct else 0

        result = {
            "n_match": len(match_data),
            "n_nomatch": len(nomatch_data),
            "top1_accuracy": round(float(np.mean(match_correct)), 4),
            "near_top1": round(near_top1, 4),
            "margin_separation_auc": sep_auc,
            "coverage_at_fdr": cov_at_fdr,
        }
        results[method] = result
        print(f"  {method}: R@1={result['top1_accuracy']:.4f} sep={sep_auc} "
              f"cov@5%={cov_at_fdr.get('0.05', {}).get('coverage', 'N/A')}", flush=True)

    # Compare: unified (best margin method) vs each baseline
    # Primary: coverage at 5% FDR
    best_method = max(results, key=lambda m: results[m].get("coverage_at_fdr", {})
                     .get("0.05", {}).get("coverage", 0))
    baseline_cov = {m: results[m].get("coverage_at_fdr", {}).get("0.05", {})
                   .get("coverage", 0) for m in results}

    report = {
        "status": "ENTITY_RECOVERY_EVALUATION",
        "design": (
            f"70/30 identity split; no-match positives removed; "
            f"margin-based coverage at FDR targets {args.fdr_targets}; "
            f"mixed population (match + no-match)"
        ),
        "results": results,
        "best_method_at_5pct_fdr": best_method,
        "coverage_comparison_at_5pct": baseline_cov,
        "increment_over_official": round(
            baseline_cov.get(best_method, 0) - baseline_cov.get("official_dreams", 0), 4),
        "claim_limit": (
            "Entity recovery evaluation. Coverage = fraction of match queries "
            "correctly assigned at calibrated FDR. Not a biological result."
        ),
    }
    with open(args.output / "entity_recovery_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nBest at 5% FDR: {best_method} "
          f"(cov={baseline_cov.get(best_method, 0):.3f})", flush=True)
    print(f"Increment over official: {report['increment_over_official']}", flush=True)


if __name__ == "__main__":
    main()
