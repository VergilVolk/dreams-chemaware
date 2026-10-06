"""Failure profile of queries by deployment-computable metadata.

Question: besides the spectral gap, do precursor m/z and candidate-set size
carry information about Top-1 failure? Descriptive analysis (no claims of
novelty): failure rate by precursor-mz decile and by candidate-count bucket,
for WSE and noise_v1, both panels. Feeds the escalation policy of the
identifiability gate (which queries to escalate even before scoring).
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

report = {"status": "GLM_FAILURE_PROFILE", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, _ = per_query_stats(g, mol)
    mz = np.load(Path("data/validation/GLM_gnps_identity_panel_reconstruction")
                 / f"panel_{panel}.npz")["query_precursor_mz"]
    n_mol = np.diff(g.query_ptr).astype(int)
    correct = rank == 1
    entry = {}
    for m in ("weighted_spectral_entropy", "noise_v1"):
        mi = methods.index(m)
        fail = ~correct[mi]
        # precursor-mz deciles
        qs = np.quantile(mz, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(qs, mz, side="right") - 1, 0, 9)
        by_mz = [round(float(fail[dec == d].mean() * 100), 1)
                 for d in range(10)]
        # candidate-count buckets 2,3,4-5,6-8,9+
        buckets = [(2, 2), (3, 3), (4, 5), (6, 8), (9, 10 ** 9)]
        by_n = []
        for lo, hi in buckets:
            mask = (n_mol >= lo) & (n_mol <= hi)
            by_n.append({"range": f"{lo}-{hi if hi < 10**9 else '+'}",
                         "n": int(mask.sum()),
                         "fail_pct": round(float(fail[mask].mean() * 100), 1)})
        entry[m] = {"fail_by_precursor_decile_pct": by_mz,
                    "fail_by_candidate_count": by_n,
                    "overall_fail_pct": round(float(fail.mean() * 100), 2)}
    report["panels"][panel] = entry
    print(f"=== {panel} ===")
    for m, v in entry.items():
        print(f"  {m}: overall {v['overall_fail_pct']}%")
        print(f"    by precursor decile: {v['fail_by_precursor_decile_pct']}")
        print(f"    by n_candidates: "
              + ", ".join(f"{b['range']}:{b['fail_pct']}%(n={b['n']})"
                          for b in v["fail_by_candidate_count"]))

out = Path("deliverables/GLM_gnps_article_ladder/run15/failure_profile.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
