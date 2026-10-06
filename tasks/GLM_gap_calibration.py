"""Diagnostic: why is the oracle headroom uncapturable?

Calibration analysis: per method, does the (truth-blind) top1-top2 gap
predict whether the method's top-1 is correct? Deciles of batch-percentile
gap -> P(correct), plus AUC. Flat curves / AUC near 0.5 mean confidence is
uninformative about correctness, which explains why every selection and
fusion strategy collapses to ~0 gain despite +4.6/+5.8pp oracle headroom.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

OUT = Path(os.environ.get(
    "GLM_OUT_DIR", "deliverables/GLM_gnps_article_ladder/run_local"))
report = {"status": "GLM_GAP_CALIBRATION", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, top1 = per_query_stats(g, mol)
    N = rank.shape[1]
    correct = rank == 1
    rows = []
    print(f"=== {panel} (n={N}) ===")
    print(f"{'method':30s} {'R@1':>6s}  P(correct|gap quintile Q1..Q5)  AUC")
    for mi, m in enumerate(methods):
        gp = np.argsort(np.argsort(gap[mi])) / (N - 1)
        qs = []
        for q in range(5):
            lo, hi = q / 5, (q + 1) / 5
            mask = (gp >= lo) & (gp < hi) if q < 4 else (gp >= lo)
            qs.append(round(float(correct[mi][mask].mean() * 100), 1))
        auc = float(roc_auc_score(correct[mi], gp))
        rows.append({"method": m, "recall1": round(float(correct[mi].mean() * 100), 2),
                     "p_correct_by_gap_quintile": qs, "auc_gap_to_correct": round(auc, 4)})
        print(f"{m:30s} {correct[mi].mean()*100:6.2f}  "
              + "  ".join(f"{x:5.1f}" for x in qs) + f"   {auc:.3f}")
    report["panels"][panel] = rows

(OUT / "gap_calibration.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
print(f"\nwritten: {OUT / 'gap_calibration.json'}")
