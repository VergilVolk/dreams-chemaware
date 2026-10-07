"""Audit 2: true unique-win counts per method (corrects '独特贡献' wording).

winner_counts in risk_coverage.json counts among-winner membership on
recoverable queries, NOT unique wins. This audit reports, per method:
  unique_wins   queries where this method is the ONLY correct one of 15
  among_wins_R  recoverable queries where it is among the winners
  unique_share  unique_wins / total queries
Honest language: 'largest single contributor among winners', or 'most
unique wins' only where unique_wins ranks first.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

report = {"status": "GLM_AUDIT_UNIQUE_WINS", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, _, _ = per_query_stats(g, mol)
    correct = rank == 1
    M, N = rank.shape
    wse_i = methods.index("weighted_spectral_entropy")
    R = (~correct[wse_i]) & correct.any(axis=0)
    n_winners = correct.sum(axis=0)
    rows = {}
    for mi, m in enumerate(methods):
        only = correct[mi] & (n_winners == 1)
        rows[m] = {
            "recall1": round(float(correct[mi].mean() * 100), 2),
            "unique_wins": int(only.sum()),
            "unique_share_pct": round(float(only.mean() * 100), 3),
            "among_wins_on_recoverable": int((correct[mi] & R).sum()),
        }
    report["panels"][panel] = {"n_queries": N,
                               "n_recoverable": int(R.sum()),
                               "by_method": rows}
    print(f"=== {panel} ===")
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["unique_wins"]):
        print(f"  {m:28s} R@1 {v['recall1']:6.2f}  unique {v['unique_wins']:4d}"
              f"  among-winners-on-R {v['among_wins_on_recoverable']:4d}")

out = Path("deliverables/GLM_gnps_article_ladder/run15/audit_unique_wins.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
