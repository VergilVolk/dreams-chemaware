#!/usr/bin/env python
"""P0-M: Margin-based abstention for spectral coordinate mapping.

P0-G proved absolute scores CANNOT distinguish match from no-match (AUC 0.507).
P0-L proved margin (top1-top2 gap) IS effective for selective assignment.

P0-M tests: can margin distinguish "correct coordinate found" from
"no correct coordinate exists"?  If yes → margin is the deployment-grade
abstention signal for spectral-coordinate metabolomics.

Same frozen panels and score bundle as P0-L/P0-G. No recomputation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--score-key", type=str, required=True)
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
    scores = np.asarray(bundle[args.score_key], dtype=np.float64)

    n_queries = len(panel["query_row"])
    query_ptr = panel["query_ptr"]
    molecule_ptr = panel["molecule_ptr"]
    molecule_label = panel["molecule_label"].astype(bool)
    query_ik = panel["query_ik14"].astype(str)
    near_query = panel["near_query"].astype(bool)

    # Identity split (same as P0-G)
    unique_identities = sorted(set(query_ik))
    rng = np.random.default_rng(args.seed)
    n_dict = int(len(unique_identities) * args.dict_fraction)
    shuffled = rng.permutation(len(unique_identities))
    dict_ids = set(unique_identities[i] for i in shuffled[:n_dict])
    is_match = np.array([q in dict_ids for q in query_ik])
    print(f"queries: {int(is_match.sum())} match + {int((~is_match).sum())} no-match", flush=True)

    results = {}
    for method in args.methods:
        if method not in methods_list:
            continue
        midx = methods_list.index(method)
        pair_scores = scores[midx]

        # Aggregate to molecule level
        molecule_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

        # For each query: compute molecule-level margins and correctness
        match_margins = []       # margin for match queries
        match_correct = []       # whether top-1 is the correct molecule
        nomatch_margins = []     # margin for no-match queries
        match_margins_near = []  # margins for near-structure match queries

        for q in range(n_queries):
            lo, hi = query_ptr[q], query_ptr[q + 1]
            mol_scores = molecule_scores[lo:hi].copy()
            mol_labels = molecule_label[lo:hi]

            if is_match[q]:
                # Match query: positive present, compute normally
                if len(mol_scores) < 2:
                    margin = 1.0
                else:
                    ordered = np.sort(mol_scores)
                    margin = float(ordered[-1] - ordered[-2])
                pos = np.flatnonzero(mol_labels)
                if len(pos) == 1:
                    true_score = mol_scores[pos[0]]
                    neg = mol_scores[~mol_labels]
                    rank = 1 + int(np.sum(neg >= true_score))
                    match_margins.append(margin)
                    match_correct.append(rank == 1)
                    if near_query[q]:
                        match_margins_near.append(margin)
            else:
                # No-match query: REMOVE the positive, then compute margin
                # among remaining (all wrong) candidates
                pos_mask = mol_labels.copy()
                if np.any(pos_mask):
                    # Remove positive from scores (set to -inf)
                    mol_scores_nomatch = mol_scores.copy()
                    mol_scores_nomatch[pos_mask] = -np.inf
                    # Compute margin among non-positive candidates only
                    valid_scores = mol_scores_nomatch[~pos_mask]
                    if len(valid_scores) < 2:
                        margin = 0.0  # only one wrong candidate → no real choice
                    else:
                        ordered = np.sort(valid_scores)
                        margin = float(ordered[-1] - ordered[-2])
                else:
                    # No positive at all (shouldn't happen in this panel)
                    if len(mol_scores) < 2:
                        margin = 0.0
                    else:
                        ordered = np.sort(mol_scores)
                        margin = float(ordered[-1] - ordered[-2])
                nomatch_margins.append(margin)

        match_margins = np.array(match_margins)
        match_correct = np.array(match_correct)
        nomatch_margins = np.array(nomatch_margins)
        match_margins_near = np.array(match_margins_near) if match_margins_near else np.array([])

        # === KEY TEST: does margin separate match from no-match? ===
        sep_auc = None
        if len(match_margins) >= 10 and len(nomatch_margins) >= 10:
            u, p = sps.mannwhitneyu(match_margins, nomatch_margins, alternative="greater")
            sep_auc = round(float(u / (len(match_margins) * len(nomatch_margins))), 4)

        # === Deployment calibration: margin threshold → (coverage, precision, FDR) ===
        thresholds = np.linspace(0, 1, 201)
        deploy = []
        for t in thresholds:
            # Match queries assigned (margin > t)
            assigned_match = match_margins > t
            n_assigned = int(assigned_match.sum())
            # Of those assigned, how many are correct?
            if n_assigned > 0:
                precision = float(np.mean(match_correct[assigned_match]))
            else:
                precision = 0.0
            # No-match queries falsely assigned
            if len(nomatch_margins) > 0:
                false_rate = float(np.mean(nomatch_margins > t))
            else:
                false_rate = 0.0
            coverage = n_assigned / max(len(match_margins), 1)
            deploy.append({
                "threshold": round(float(t), 4),
                "coverage": round(coverage, 4),
                "precision_on_match": round(precision, 4),
                "no_match_false_assign_rate": round(false_rate, 4),
            })

        # Find operating points
        def find_op(target_fdr):
            best = None
            for d in deploy:
                # FDR ≈ (no_match_false_rate * n_nomatch) / (coverage * n_match + false)
                # Simplified: use no_match_false_assign_rate as proxy
                if d["no_match_false_assign_rate"] <= target_fdr and d["coverage"] > 0:
                    if best is None or d["coverage"] > best["coverage"]:
                        best = d
            return best

        op_5 = find_op(0.05)
        op_2 = find_op(0.02)

        # Selective risk: among match queries with margin > t, what fraction are wrong?
        selective = []
        for t in thresholds:
            mask = match_margins > t
            if mask.sum() >= 5:
                risk = 1.0 - float(np.mean(match_correct[mask]))
                selective.append({"threshold": round(float(t), 4),
                                  "coverage": round(float(mask.mean()), 4),
                                  "risk": round(risk, 6)})
            else:
                selective.append({"threshold": round(float(t), 4),
                                  "coverage": 0.0, "risk": None})

        result = {
            "match_queries": len(match_margins),
            "no_match_queries": len(nomatch_margins),
            "near_match_queries": len(match_margins_near),
            "match_median_margin": round(float(np.median(match_margins)), 4),
            "nomatch_median_margin": round(float(np.median(nomatch_margins)), 4) if len(nomatch_margins) > 0 else None,
            "near_match_median_margin": round(float(np.median(match_margins_near)), 4) if len(match_margins_near) > 0 else None,
            "separation_auc": sep_auc,
            "operating_point_5pct_fdr": op_5,
            "operating_point_2pct_fdr": op_2,
            "selective_risk_60pct": next((s["risk"] for s in selective
                                           if s["coverage"] >= 0.60), None),
            "selective_risk_80pct": next((s["risk"] for s in selective
                                           if s["coverage"] >= 0.80), None),
        }
        results[method] = result
        print(f"  {method}: sep_AUC={sep_auc} "
              f"match_med={result['match_median_margin']} "
              f"nomatch_med={result['nomatch_median_margin']} "
              f"risk@60%={result['selective_risk_60pct']}", flush=True)

    gates = {}
    for method, r in results.items():
        gates[f"{method}_separation_gt_60"] = (r["separation_auc"] or 0) > 0.60
        gates[f"{method}_risk_at_60pct_lt_5pct"] = (
            r["selective_risk_60pct"] is not None and r["selective_risk_60pct"] < 0.05)

    all_pass = all(v for v in gates.values() if isinstance(v, bool))
    report = {
        "status": "P0_M_" + ("PASS" if all_pass else "STOP"),
        "dict_fraction": args.dict_fraction,
        "results": results,
        "gates": gates,
        "key_finding": (
            f"Margin (top1-top2 gap) separation AUC: "
            f"{results.get('noise_v1', {}).get('separation_auc', 'N/A')}. "
            f"Absolute score AUC was 0.507 (chance). "
            f"Margin selective risk at 60% coverage: "
            f"{results.get('noise_v1', {}).get('selective_risk_60pct', 'N/A')}."
        ),
        "claim_limit": "P0-M: margin-based abstention. Tests if top1-top2 gap can replace absolute score.",
    }
    with open(args.output / "p0m_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nP0-M: {report['status']}", flush=True)
    print(f"KEY: {report['key_finding']}", flush=True)


if __name__ == "__main__":
    main()
