"""Paired CIs: truth-blind selection (U1) vs EACH individual method.

The original objective asked for S1-vs-each-method paired CIs. S1 in its
leaked form is retracted; its honest successor is U1 (select the method with
the largest raw top1-top2 gap). This table gives U1 vs every one of the 15
methods, both panels, paired-by-query percentile bootstrap (10k).
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
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats, r1  # noqa: E402

BOOT_N = 10000
RNG = np.random.default_rng(20261007)

report = {"status": "GLM_U1_VS_EACH_METHOD_CI", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, _ = per_query_stats(g, mol)
    correct = rank == 1
    M, N = rank.shape
    pick = np.argmax(gap, axis=0)
    u1 = correct[pick, np.arange(N)]
    idx = RNG.integers(0, N, size=(BOOT_N, N))
    u1_b = u1[idx].mean(axis=1)
    rows = {}
    for mi, m in enumerate(methods):
        c = correct[mi]
        d = u1_b - c[idx].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows[m] = {
            "method_recall1": r1(c),
            "u1_minus_method_pp": round(r1(u1) - r1(c), 2),
            "ci95": [round(float(lo * 100), 2), round(float(hi * 100), 2)],
            "u1_significantly_better": bool(lo > 0),
            "method_significantly_better": bool(hi < 0),
        }
    report["panels"][panel] = {"U1_recall1": r1(u1), "vs_each": rows}
    print(f"=== {panel}: U1 = {r1(u1)}% ===")
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["method_recall1"]):
        flag = ("U1_BETTER" if v["u1_significantly_better"]
                else "METHOD_BETTER" if v["method_significantly_better"
                ] else "n.s.")
        print(f"  {m:28s} {v['method_recall1']:6.2f}%  "
              f"U1-method {v['u1_minus_method_pp']:+6.2f} "
              f"[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]  {flag}")

out = Path("deliverables/GLM_gnps_article_ladder/run15/u1_vs_each_method.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
