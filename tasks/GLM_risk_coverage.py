"""Risk-coverage analysis + failure-band structure.

Question 1: at matched coverage, what accuracy does each method achieve when
  allowed to abstain on its lowest-confidence queries (truth-blind gate)?
Question 2: in the band where the best single method (WSE) fails but the
  oracle succeeds, how confident is the winning method? If winners are
  low-confidence, no confidence-based router can recover the oracle
  headroom -- which is exactly what the leak-free router experiments show.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

OUT = ROOT / "deliverables/GLM_gnps_article_ladder/run_local"
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

report = {"status": "GLM_RISK_COVERAGE", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, top1 = per_query_stats(g, mol)
    N = rank.shape[1]
    correct = rank == 1
    wse_i = methods.index("weighted_spectral_entropy")
    oracle = correct.any(axis=0)

    # ---- risk-coverage per method -------------------------------------
    rc = {}
    for mi, m in enumerate(methods):
        order = np.argsort(-gap[mi], kind="stable")  # most confident first
        c = correct[mi][order]
        rc[m] = {f"cov{int(cov*100)}": round(float(c[: int(cov * N)].mean() * 100), 2)
                 for cov in COVERAGES}

    # ---- failure-band structure ---------------------------------------
    wse_wrong = ~correct[wse_i]
    recoverable = wse_wrong & oracle
    # among recoverable queries: percentile rank of the winning methods' gaps
    win_pct = []
    winners = []
    for i in np.flatnonzero(recoverable):
        for mi in range(len(methods)):
            if correct[mi, i]:
                gp = np.argsort(np.argsort(gap[mi]))[i] / (N - 1)
                win_pct.append(float(gp))
                winners.append(methods[mi])
    win_pct = np.array(win_pct)
    # for contrast: gap percentile of WSE on queries it gets right
    wse_right_pct = np.argsort(np.argsort(gap[wse_i]))[correct[wse_i]] / (N - 1)

    # correctness correlation across methods (mean pairwise phi)
    phis = []
    for a in range(len(methods)):
        for b in range(a + 1, len(methods)):
            ca, cb = correct[a], correct[b]
            p1, p2 = ca.mean(), cb.mean()
            p12 = (ca & cb).mean()
            cov_ = p12 - p1 * p2
            den = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
            phis.append(cov_ / den)
    band = {
        "wse_wrong_total": int(wse_wrong.sum()),
        "recoverable_by_oracle": int(recoverable.sum()),
        "winning_method_gap_percentile_mean": round(float(win_pct.mean()), 3),
        "winning_method_gap_percentile_below_60pct": round(
            float((win_pct < 0.6).mean() * 100), 1),
        "wse_correct_gap_percentile_mean": round(float(wse_right_pct.mean()), 3),
        "mean_pairwise_correctness_phi": round(float(np.mean(phis)), 3),
        "winner_counts": {m: int(sum(1 for w in winners if w == m))
                          for m in sorted(set(winners))},
    }

    report["panels"][panel] = {"risk_coverage": rc, "failure_band": band}
    print(f"=== {panel} (n={N}) ===")
    for m in ("weighted_spectral_entropy", "noise_v1", "entropy_raw_public",
              "denoising_search_public", "cosine_greedy"):
        print(f"  {m:28s} " + "  ".join(
            f"c{int(c*100)}={rc[m][f'cov{int(c*100)}']:5.2f}" for c in COVERAGES))
    print(f"  failure band: {band}")

(OUT / "risk_coverage.json").write_text(json.dumps(report, indent=2),
                                        encoding="utf-8")
print(f"\nwritten: {OUT / 'risk_coverage.json'}")
