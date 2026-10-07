"""Audit 4: reconcile reconstruction-derived numbers against frozen tables.

The gap-calibration AUCs, risk-coverage points, and correctness correlation
were computed from the bundle via my own molecule-level reconstruction.
This audit recomputes four of them directly from the FROZEN per-query
tables (independent path) and checks equality with the published JSONs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation"
OUT = ROOT / "deliverables/GLM_gnps_article_ladder/run15"
METHODS = ["official_dreams", "noise_v1", "cosine_greedy", "modified_cosine",
           "weighted_spectral_entropy", "p2b_sqrt_cosine",
           "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine",
           "p2b_official_frozen", "p2b_noise_v1_frozen",
           "ms2deepscore_2x_public", "spec2vec_gnps_public",
           "entropy_raw_public", "denoising_search_public",
           "spec2vec_2026_retrained"]

panel = "identity_disjoint"
tabs = {}
for m in METHODS:
    df = pd.read_csv(RUN / f"queries_{panel}_{m}.csv.gz",
                     low_memory=False).sort_values("query_index", kind="stable")
    tabs[m] = df

checks = {}

# (i) WSE risk-coverage c60
gap = tabs["weighted_spectral_entropy"]["top1_top2_gap"].to_numpy()
corr = (tabs["weighted_spectral_entropy"]["rank"].to_numpy() == 1)
order = np.argsort(-gap, kind="stable")
c = corr[order]
n = len(c)
k = int(0.6 * n)
checks["wse_c60_identity"] = {
    "recomputed": round(float(c[:k].mean() * 100), 2),
    "published": json.loads((OUT / "risk_coverage_ci.json").read_text())[
        "panels"][panel]["weighted_spectral_entropy"]["cov60"]["accuracy"]}

# (ii) calibration AUC for entropy_raw
g2 = tabs["entropy_raw_public"]["top1_top2_gap"].to_numpy()
c2 = (tabs["entropy_raw_public"]["rank"].to_numpy() == 1)
auc = float(roc_auc_score(c2, np.argsort(np.argsort(g2)) / (n - 1)))
checks["entropy_raw_calibration_auc"] = {
    "recomputed": round(auc, 4),
    "published": [r for r in json.loads(
        (OUT / "gap_calibration.json").read_text())["panels"][panel]
        if r["method"] == "entropy_raw_public"][0]["auc_gap_to_correct"]}

# (iii) mean pairwise correctness phi
C = np.stack([(tabs[m]["rank"].to_numpy() == 1).astype(float)
              for m in METHODS])
phis = []
for a in range(len(METHODS)):
    for b in range(a + 1, len(METHODS)):
        p1, p2 = C[a].mean(), C[b].mean()
        p12 = (C[a] * C[b]).mean()
        den = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
        phis.append((p12 - p1 * p2) / den)
checks["mean_pairwise_phi"] = {
    "recomputed": round(float(np.mean(phis)), 3),
    "published": json.loads((OUT / "risk_coverage.json").read_text())[
        "panels"][panel]["failure_band"]["mean_pairwise_correctness_phi"]}

# (iv) U1 near recall
gaps = np.stack([tabs[m]["top1_top2_gap"].to_numpy() for m in METHODS])
ranks = np.stack([tabs[m]["rank"].to_numpy() for m in METHODS])
near = tabs[METHODS[0]]["near"].to_numpy(bool)
correct = ranks == 1
pick = np.argmax(gaps, axis=0)
u1 = correct[pick, np.arange(n)]
checks["u1_near_recall1"] = {
    "recomputed": round(float(u1[near].mean() * 100), 2),
    "published": json.loads((OUT / "truthblind_ensemble.json").read_text())[
        "panels"][panel]["U1_select_raw_gap"]["near_recall1"]}

ok = True
for k, v in checks.items():
    match = abs(v["recomputed"] - v["published"]) < 0.06
    ok &= match
    print(f"{k}: recomputed {v['recomputed']} vs published {v['published']}"
          f" -> {'MATCH' if match else 'MISMATCH'}")
print("AUDIT 4:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
