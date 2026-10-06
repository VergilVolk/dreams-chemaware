"""End-to-end spot audit of one number chain (independent recomputation).

Target claim: truthblind_ensemble.json identity U1_select_raw_gap = 87.33%
(delta -0.04pp vs WSE 87.37). Independent path: read the FROZEN per-query
tables directly (no bundle, no reduceat), compute per-method gap argmax
selection, strict-rank semantics already encoded in the tables' rank column.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation")
methods = ["official_dreams", "noise_v1", "cosine_greedy", "modified_cosine",
           "weighted_spectral_entropy", "p2b_sqrt_cosine",
           "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine",
           "p2b_official_frozen", "p2b_noise_v1_frozen",
           "ms2deepscore_2x_public", "spec2vec_gnps_public",
           "entropy_raw_public", "denoising_search_public",
           "spec2vec_2026_retrained"]
assert len(methods) == 15

gaps, ranks = [], []
for m in methods:
    df = pd.read_csv(RUN / f"queries_identity_disjoint_{m}.csv.gz",
                     low_memory=False).sort_values("query_index", kind="stable")
    assert (df["query_index"].to_numpy() == np.arange(len(df))).all()
    gaps.append(df["top1_top2_gap"].to_numpy())
    ranks.append(df["rank"].to_numpy())
G = np.stack(gaps)
R = np.stack(ranks)
correct = R == 1
pick = np.argmax(G, axis=0)
u1 = correct[pick, np.arange(G.shape[1])].mean() * 100
wse = correct[methods.index("weighted_spectral_entropy")].mean() * 100
oracle = correct.any(axis=0).mean() * 100

report = json.loads(Path("deliverables/GLM_gnps_article_ladder/run15/"
                         "truthblind_ensemble.json").read_text(encoding="utf-8"))
pan = report["panels"]["identity_disjoint"]
checks = {
    "U1_recomputed": round(float(u1), 2),
    "U1_reported": pan["U1_select_raw_gap"]["recall1"],
    "WSE_recomputed": round(float(wse), 2),
    "WSE_reported": pan["_best_single"]["recall1"],
    "oracle_recomputed": round(float(oracle), 2),
    "oracle_reported": pan["_oracle_any"]["recall1"],
}
ok = (checks["U1_recomputed"] == checks["U1_reported"]
      and checks["WSE_recomputed"] == checks["WSE_reported"]
      and checks["oracle_recomputed"] == checks["oracle_reported"])
print(json.dumps(checks, indent=2))
print("SPOT AUDIT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
