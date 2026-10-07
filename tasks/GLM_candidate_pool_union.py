"""Unified algorithm step 0: candidate-pool union analysis.

Per the corrected unified design (2026-10-07 discussion): the candidate pool
must be the UNION of the component encoders' top-M lists, not the top-M of a
single model, otherwise correct candidates recoverable by ChemAware/WSE/P2b
can never be rescued. This script measures, on both frozen GNPS panels:

  - Recall@M per component view (molecule-level, frozen max-pool semantics)
  - pairwise and triple unions (locally available views: Noise, WSE, P2b-on-
    Noise; ChemAware encoder slot is server-side and documented as such)
  - unrecoverable fraction: positives outside the union at budget M
  - rescue-pool availability: among queries where Noise top-1 is wrong, how
    often the positive sits inside WSE top-M / P2b top-M (i.e. rescuable by
    the pool, pending a pair-level decision)

No weights are chosen here; this is pool coverage accounting only.
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
from GLM_truthblind_fusion_analysis import load_panel  # noqa: E402

OUT = ROOT / "data/validation/GLM_candidate_differential_ledger"
VIEWS = ["noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen",
         "entropy_raw_public", "denoising_search_public",
         "spec2vec_gnps_public", "official_dreams"]
MS = (5, 10, 20)

report = {"status": "GLM_CANDIDATE_POOL_UNION",
          "chemaware_slot": ("ChemAware encoder scores on these panels are "
                             "server-side; the union numbers below understate "
                             "the full design pool by that slot"),
          "panels": {}}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    qptr, mptr = g.query_ptr, g.molecule_ptr
    n_q = g.n_queries
    top_sets = {m: {} for m in VIEWS}
    for m in VIEWS:
        mi = methods.index(m)
        mol_scores = mol[mi]  # load_panel already returns molecule-level scores
        for q in range(n_q):
            lo, hi = int(qptr[q]), int(qptr[q + 1])
            if hi <= lo:
                continue
            top_sets[m][q] = set(
                (lo + np.argsort(-mol_scores[lo:hi], kind="stable")[:max(MS)])
                .tolist())
    pos = {q: int(qptr[q]) for q in range(n_q)}  # positive is block-first

    entry = {"n_queries": n_q, "views": {}, "unions": {}, "rescue_pool": {}}
    for M in MS:
        entry["views"][f"M={M}"] = {
            m: round(float(np.mean([
                pos[q] in top_sets[m][q] for q in range(n_q)
                if q in top_sets[m]]) * 100), 2) for m in VIEWS}
        unions = {
            "noise+wse": ("noise_v1", "weighted_spectral_entropy"),
            "noise+p2bN": ("noise_v1", "p2b_noise_v1_frozen"),
            "wse+p2bN": ("weighted_spectral_entropy", "p2b_noise_v1_frozen"),
            "noise+wse+p2bN": ("noise_v1", "weighted_spectral_entropy",
                               "p2b_noise_v1_frozen"),
        }
        entry["unions"][f"M={M}"] = {}
        for name, views in unions.items():
            hit = np.mean([
                any(pos[q] in top_sets[v][q] for v in views)
                for q in range(n_q) if all(q in top_sets[v] for v in views)])
            entry["unions"][f"M={M}"][name] = round(float(hit * 100), 2)
        # unrecoverable under the design pool (local 3-view proxy)
        views3 = unions["noise+wse+p2bN"]
        unrec = [q for q in range(n_q)
                 if all(q in top_sets[v] for v in views3)
                 and not any(pos[q] in set(list(top_sets[v][q])[:M])
                             for v in views3)]
        entry["unions"][f"M={M}"]["unrecoverable_pct"] = round(
            100 * len(unrec) / max(1, n_q), 2)

    # rescue-pool availability: noise top-1 wrong -> positive in wse/p2b top-M
    mi_n = methods.index("noise_v1")
    mol_noise = mol[mi_n]
    wrong, wse_has, p2b_has, any_has = 0, 0, 0, 0
    for q in range(n_q):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi <= lo:
            continue
        top1 = lo + int(np.argmax(mol_noise[lo:hi]))
        if top1 == pos[q]:
            continue
        wrong += 1
        in_w = pos[q] in set(list(top_sets["weighted_spectral_entropy"][q])[:10])
        in_p = pos[q] in set(list(top_sets["p2b_noise_v1_frozen"][q])[:10])
        wse_has += in_w
        p2b_has += in_p
        any_has += (in_w or in_p)
    entry["rescue_pool"] = {
        "noise_top1_wrong": wrong,
        "positive_in_wse_top10_pct": round(100 * wse_has / max(1, wrong), 2),
        "positive_in_p2b_top10_pct": round(100 * p2b_has / max(1, wrong), 2),
        "positive_in_either_top10_pct": round(100 * any_has / max(1, wrong), 2),
    }
    report["panels"][panel] = entry
    print(f"=== {panel} ===")
    for M in MS:
        print(f" M={M}: " + "  ".join(
            f"{m.split('_')[0]}={v}" for m, v in entry["views"][f"M={M}"].items()))
        print(f"   unions: {entry['unions'][f'M={M}']}")
    print(f" rescue pool: {entry['rescue_pool']}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pool_union.json").write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")
print("written:", OUT / "pool_union.json")
