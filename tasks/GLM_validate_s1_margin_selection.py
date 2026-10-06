"""Validate S1 margin-selection: bootstrap CI, mechanism check, minimal method set.

Answers three questions:
1. Is +4.93pp statistically significant? (formula-cluster bootstrap CI)
2. WHY does margin perfectly predict which method is correct?
3. What is the minimal set of methods needed to capture the complementarity?
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd

RUN = Path("data/validation/GLM_gnps_article_benchmark_challengers/run_local")
PANELS = ("identity_disjoint", "formula_disjoint")
BOOTSTRAP = 10000
SEED = 20261006

report: dict = {"status": "GLM_S1_VALIDATION", "panels": {}}

for panel in PANELS:
    tables = {}
    for method in ("noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen",
                   "entropy_raw_public", "denoising_search_public",
                   "p2b_sqrt_cosine", "spec2vec_gnps_public",
                   "neutral_loss_sqrt_cosine", "cosine_greedy"):
        path = RUN / "evaluation" / f"queries_{panel}_{method}.csv.gz"
        if path.is_file():
            tables[method] = pd.read_csv(path, low_memory=False).sort_values(
                "query_index", kind="stable")
    methods = sorted(tables)
    base = tables[methods[0]]
    n = len(base)
    near = base["near"].to_numpy(bool)
    formulas = base["query_formula"].astype(str).to_numpy()

    ranks = {m: tables[m]["rank"].to_numpy() for m in methods}
    margins = {m: tables[m]["positive_vs_best_negative_margin"].to_numpy()
               for m in methods}
    correct = {m: (ranks[m] == 1) for m in methods}

    # S1: margin selection
    margin_matrix = np.stack([margins[m] for m in methods])
    best_idx = np.argmax(margin_matrix, axis=0)
    s1_correct = np.zeros(n, dtype=bool)
    for i, m in enumerate(methods):
        mask = best_idx == i
        s1_correct[mask] = correct[m][mask]
    best_single = max(correct, key=lambda m: correct[m].mean())
    wse_correct = correct.get("weighted_spectral_entropy", correct[best_single])

    # ---- Q1: bootstrap CI for S1 vs WSE ----
    rng = np.random.default_rng(SEED)
    unique_formulas = np.unique(formulas)
    formula_to_idx = {f: i for i, f in enumerate(unique_formulas)}
    formula_idx = np.array([formula_to_idx[f] for f in formulas])

    def cluster_bootstrap_delta(a, b, formulas_idx, n_clusters, repeats, seed):
        """Bootstrap the difference a-b by resampling formula clusters."""
        r = np.random.default_rng(seed)
        deltas = np.zeros(repeats)
        cluster_ids = np.unique(formulas_idx)
        for i in range(repeats):
            sampled = r.choice(cluster_ids, size=len(cluster_ids), replace=True)
            mask = np.zeros(len(a), dtype=bool)
            for c in sampled:
                mask |= formulas_idx == c
            deltas[i] = (a[mask].mean() - b[mask].mean()) * 100
        return deltas

    delta_samples = cluster_bootstrap_delta(
        s1_correct, wse_correct, formula_idx, len(unique_formulas), BOOTSTRAP, SEED)
    ci = np.percentile(delta_samples, [2.5, 97.5])

    # ---- Q2: mechanism ----
    # When margin selects the correct method, what's the margin distribution?
    # When it selects a wrong method, what happened?
    s1_wrong = ~s1_correct
    oracle_correct = np.zeros(n, dtype=bool)
    for c in correct.values():
        oracle_correct |= c

    # How often does each method get selected?
    selection_counts = {m: int((best_idx == i).sum())
                        for i, m in enumerate(methods)}
    # When selected, how often correct?
    selection_accuracy = {}
    for i, m in enumerate(methods):
        mask = best_idx == i
        if mask.sum() > 0:
            selection_accuracy[m] = {
                "selected": int(mask.sum()),
                "accuracy": round(float(correct[m][mask].mean()) * 100, 2)}

    # Key diagnostic: when S1 is wrong, was ANY method correct?
    s1_wrong_and_oracle_right = s1_wrong & oracle_correct
    # When S1 is wrong and no method is right → fundamentally impossible
    s1_wrong_and_oracle_wrong = s1_wrong & ~oracle_correct

    # ---- Q3: minimal method set ----
    # Try all subsets of size 2, 3, 4 and find which captures the most
    best_subsets = {}
    for size in (2, 3, 4, 5):
        best_gain = 0
        best_combo = None
        best_recall = 0
        for combo in itertools.combinations(methods, size):
            mm = np.stack([margins[m] for m in combo])
            bi = np.argmax(mm, axis=0)
            sc = np.zeros(n, dtype=bool)
            for i, m in enumerate(combo):
                mask = bi == i
                sc[mask] = correct[m][mask]
            recall = sc.mean() * 100
            if recall > best_recall:
                best_recall = recall
                best_combo = combo
                best_gain = recall - wse_correct.mean() * 100
        best_subsets[f"size_{size}"] = {
            "methods": list(best_combo) if best_combo else None,
            "recall1": round(best_recall, 2),
            "gain_vs_wse": round(best_gain, 2),
        }

    panel_report = {
        "S1_recall1": round(float(s1_correct.mean() * 100), 2),
        "WSE_recall1": round(float(wse_correct.mean() * 100), 2),
        "delta_pp": round(float(s1_correct.mean() - wse_correct.mean()) * 100, 2),
        "bootstrap_ci95": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
        "near_S1": round(float(s1_correct[near].mean() * 100), 2),
        "near_WSE": round(float(wse_correct[near].mean() * 100), 2),
        "selection_distribution": selection_accuracy,
        "s1_wrong_total": int(s1_wrong.sum()),
        "s1_wrong_but_oracle_right": int(s1_wrong_and_oracle_right.sum()),
        "s1_wrong_and_oracle_wrong": int(s1_wrong_and_oracle_wrong.sum()),
        "best_minimal_subsets": best_subsets,
    }
    report["panels"][panel] = panel_report

    print(f"\n{'='*60}")
    print(f"  {panel} (n={n:,})")
    print(f"{'='*60}")
    print(f"  S1 (margin selection): {panel_report['S1_recall1']:.2f}%")
    print(f"  WSE (best single):     {panel_report['WSE_recall1']:.2f}%")
    print(f"  Delta:                 {panel_report['delta_pp']:+.2f}pp")
    print(f"  Bootstrap CI 95%:      [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"  Near subset: S1={panel_report['near_S1']:.2f}% WSE={panel_report['near_WSE']:.2f}%")
    print(f"\n  When S1 is wrong ({panel_report['s1_wrong_total']} queries):")
    print(f"    Oracle could have saved: {panel_report['s1_wrong_but_oracle_right']}")
    print(f"    No method was correct:   {panel_report['s1_wrong_and_oracle_wrong']}")
    print(f"\n  Method selection frequency:")
    for m, stats in sorted(selection_accuracy.items(),
                            key=lambda x: -x[1]["selected"]):
        print(f"    {m:35s} selected {stats['selected']:5d}x  "
              f"accuracy {stats['accuracy']:.1f}%")
    print(f"\n  Minimal method sets:")
    for size, info in best_subsets.items():
        print(f"    {size}: {info['recall1']:.2f}% (+{info['gain_vs_wse']:.2f}pp) "
              f"← {', '.join(m.split('_')[0] for m in info['methods'])}")

out = Path("deliverables/GLM_gnps_article_ladder/run_local/s1_validation.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwritten: {out}")
