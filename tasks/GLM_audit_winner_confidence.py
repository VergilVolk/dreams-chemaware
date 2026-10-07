"""Audit 1: is 'winners are unconfident' a conditioning artifact?

The reported statistic -- winning methods' gap batch-percentile averages
0.11 on recoverable queries -- conditions on HARD queries (WSE wrong), and
hard queries have low gaps for EVERY method (correlated difficulty). The
0.11 could therefore be an artifact of query hardness, not a property of
winners. This audit adds the within-query controls:

  C1  winner methods' gap percentile    vs  loser methods' gap percentile,
      both computed on the SAME recoverable queries (within-query contrast)
  C2  AUC of gap for separating winner-vs-loser method instances on
      recoverable queries (raw gap, and within-query rank of gap)
  C3  how often the argmax-gap method is a winner on recoverable queries
      (U1's hit rate exactly where the headroom lives)
  C4  the same controls on ALL queries (not conditioned on WSE failure)

If C1 shows winners ~ losers and C2 ~ 0.5, the claim survives in its
strong form: confidence cannot separate winners from losers even within
the same hard query. If winners are clearly above losers, the original
batch-percentile framing was misleading and must be corrected.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

report = {"status": "GLM_AUDIT_WINNER_CONFIDENCE_CONTROLS", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, _ = per_query_stats(g, mol)
    M, N = rank.shape
    correct = rank == 1
    pct = np.argsort(np.argsort(gap, axis=1), axis=1) / (N - 1)
    wse_i = methods.index("weighted_spectral_entropy")
    oracle = correct.any(axis=0)
    R = (~correct[wse_i]) & oracle

    # C1: within-recoverable-query winner vs loser percentiles
    win_pcts, lose_pcts = [], []
    win_gaps, lose_gaps = [], []
    win_inq_rank, lose_inq_rank = [], []  # within-query rank of gap (0=best)
    u1_hit = 0
    for q in np.flatnonzero(R):
        row_c, row_g, row_p = correct[:, q], gap[:, q], pct[:, q]
        am = int(np.argmax(row_g))
        u1_hit += int(row_c[am])
        order = np.argsort(-row_g, kind="stable")
        inq = np.empty(M)
        inq[order] = np.arange(M)
        win_pcts.append(row_p[row_c]); lose_pcts.append(row_p[~row_c])
        win_gaps.append(row_g[row_c]); lose_gaps.append(row_g[~row_c])
        win_inq_rank.append(inq[row_c]); lose_inq_rank.append(inq[~row_c])
    win_pcts = np.concatenate(win_pcts); lose_pcts = np.concatenate(lose_pcts)
    win_gaps = np.concatenate(win_gaps); lose_gaps = np.concatenate(lose_gaps)
    win_inq_rank = np.concatenate(win_inq_rank)
    lose_inq_rank = np.concatenate(lose_inq_rank)

    y = np.concatenate([np.ones(len(win_gaps)), np.zeros(len(lose_gaps))])
    auc_gap = roc_auc_score(y, np.concatenate([win_gaps, lose_gaps]))
    auc_pct = roc_auc_score(y, np.concatenate([win_pcts, lose_pcts]))
    auc_inq = roc_auc_score(y, -np.concatenate([win_inq_rank,
                                                 lose_inq_rank]))

    # C4: same on all queries
    w_all = pct[correct]; l_all = pct[~correct]
    y_all = np.concatenate([np.ones(correct.sum()), np.zeros((~correct).sum())])
    auc_pct_all = roc_auc_score(
        y_all, np.concatenate([gap[correct], gap[~correct]]))

    entry = {
        "n_recoverable": int(R.sum()),
        "batch_pct_winners_mean": round(float(win_pcts.mean()), 4),
        "batch_pct_losers_mean": round(float(lose_pcts.mean()), 4),
        "raw_gap_winners_mean": round(float(win_gaps.mean()), 4),
        "raw_gap_losers_mean": round(float(lose_gaps.mean()), 4),
        "within_query_gap_rank_winners_mean": round(
            float(win_inq_rank.mean()), 3),
        "within_query_gap_rank_losers_mean": round(
            float(lose_inq_rank.mean()), 3),
        "C2_auc_gap_winner_vs_loser_on_R": round(float(auc_gap), 4),
        "C2_auc_batch_pct_winner_vs_loser_on_R": round(float(auc_pct), 4),
        "C2_auc_withinquery_gap_rank_on_R": round(float(auc_inq), 4),
        "C3_u1_hit_rate_on_R": round(float(u1_hit / R.sum()), 4),
        "C4_auc_gap_correct_vs_incorrect_all": round(float(auc_pct_all), 4),
    }
    report["panels"][panel] = entry
    print(f"=== {panel} (R={R.sum()}) ===")
    for k, v in entry.items():
        print(f"  {k}: {v}")

out = Path("deliverables/GLM_gnps_article_ladder/run15/"
           "audit_winner_confidence.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
