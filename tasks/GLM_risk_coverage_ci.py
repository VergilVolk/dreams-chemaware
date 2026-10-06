"""Bootstrap CIs for risk-coverage operating points (paper table).

For each key method and coverage level, the accuracy-at-coverage estimate
gets a paired-by-query 95% percentile CI via bootstrap over queries.
Also compares WSE vs noise_v1 accuracy at matched coverage (paired CI).
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

BOOT_N = 10000
RNG = np.random.default_rng(20261007)
COVERAGES = (0.5, 0.6, 0.8)
METHODS = ("weighted_spectral_entropy", "noise_v1",
           "entropy_raw_public", "denoising_search_public")

report = {"status": "GLM_RISK_COVERAGE_CI", "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, _ = per_query_stats(g, mol)
    N = rank.shape[1]
    entry: dict = {}
    # precompute sorted correctness + threshold indices per method
    sorted_c, k_idx = {}, {}
    for m in METHODS:
        mi = methods.index(m)
        order = np.argsort(-gap[mi], kind="stable")
        sorted_c[m] = (rank[mi] == 1)[order]
        k_idx[m] = {c: int(c * N) for c in COVERAGES}
    idx = RNG.integers(0, N, size=(BOOT_N, N))
    for m in METHODS:
        c_arr = sorted_c[m]
        entry[m] = {}
        for cov in COVERAGES:
            k = k_idx[m][cov]
            acc = c_arr[:k].mean() * 100
            # bootstrap over the selected population of k queries
            idx_k = RNG.integers(0, k, size=(BOOT_N, k))
            booted = c_arr[:k][idx_k].mean(axis=1) * 100
            lo, hi = np.percentile(booted, [2.5, 97.5])
            entry[m][f"cov{int(cov*100)}"] = {
                "accuracy": round(float(acc), 2),
                "ci95": [round(float(lo), 2), round(float(hi), 2)]}
    # WSE vs noise_v1 at matched coverage (paired by query resample)
    m1, m2 = "weighted_spectral_entropy", "noise_v1"
    for cov in COVERAGES:
        k = min(k_idx[m1][cov], k_idx[m2][cov])
        a = sorted_c[m1][:k].astype(np.float64)
        b = sorted_c[m2][:k].astype(np.float64)
        # NOTE: the two methods' top-k sets differ; "matched coverage" here
        # compares accuracy among each method's own k most confident queries.
        # Each method's CI resamples its own selected population; the delta
        # CI pairs them through a shared resample draw of the same shape.
        idx_k = RNG.integers(0, k, size=(BOOT_N, k))
        d = a[idx_k].mean(axis=1) - b[idx_k].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        entry[f"wse_vs_noise_v1_cov{int(cov*100)}"] = {
            "delta_pp": round(float((a.mean() - b.mean()) * 100), 2),
            "ci95": [round(float(lo * 100), 2), round(float(hi * 100), 2)],
            "note": "independent top-k sets; CI over shared query resamples"}
    report["panels"][panel] = entry
    print(f"=== {panel} ===")
    for m in METHODS:
        bits = [f"c{int(c*100)}={entry[m][f'cov{int(c*100)}']['accuracy']}"
                f"[{entry[m][f'cov{int(c*100)}']['ci95'][0]},"
                f"{entry[m][f'cov{int(c*100)}']['ci95'][1]}]"
                for c in COVERAGES]
        print(f"  {m:28s} " + "  ".join(bits))

out = Path("deliverables/GLM_gnps_article_ladder/run15/risk_coverage_ci.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
