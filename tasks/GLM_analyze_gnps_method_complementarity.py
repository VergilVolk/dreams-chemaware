"""Complementarity analysis of the 12-method GNPS ladder (local, verified).

Reads the frozen per-query tables of the extended run, computes per-method
correct sets (rank == 1), pairwise overlaps, oracle unions and the best
method families, and quantifies how much non-redundant retrieval information
exists across the 12 methods on each sealed panel.  This is the empirical
fuel for the candidate-layer (router) rationale: every number derives from
the same verified tables as the ladder.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("data/validation/GLM_gnps_article_benchmark_public_models/run_local")
PANELS = ("identity_disjoint", "formula_disjoint")
BASELINE = "official_dreams"
TOP_FAMILIES = (
    ("weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen"),
    ("weighted_spectral_entropy", "noise_v1"),
    ("weighted_spectral_entropy", "spec2vec_gnps_public", "noise_v1"),
)

OUT = Path("deliverables/GLM_gnps_article_ladder/run_local")
report: dict = {"status": "GLM_GNPS_METHOD_COMPLEMENTARITY_COMPLETE", "panels": {}}
rows: list[dict] = []

for panel in PANELS:
    tables = {}
    for path in sorted((RUN / "evaluation").glob(f"queries_{panel}_*.csv.gz")):
        method = path.name.replace(f"queries_{panel}_", "").removesuffix(".csv.gz")
        table = pd.read_csv(path, low_memory=False)
        tables[method] = table
        assert len(table) == len(tables.get(BASELINE, table)), method
    methods = sorted(tables)
    base = tables[BASELINE].sort_values("query_index", kind="stable")
    n = len(base)
    near = base["near"].to_numpy(bool)

    correct: dict[str, np.ndarray] = {}
    for method, table in tables.items():
        aligned = table.sort_values("query_index", kind="stable")
        assert np.array_equal(aligned["query_index"].to_numpy(),
                              base["query_index"].to_numpy())
        correct[method] = aligned["rank"].to_numpy() == 1

    panel_block: dict = {"queries": int(n),
                         "per_method_correct": {m: int(c.sum())
                                                for m, c in correct.items()}}
    # pairwise: oracle union and disjointness of corrections vs baseline
    pair_rows = []
    for a, b in itertools.combinations(sorted(methods), 2):
        union = correct[a] | correct[b]
        inter = correct[a] & correct[b]
        pair_rows.append({
            "panel": panel, "method_a": a, "method_b": b,
            "oracle_union": int(union.sum()),
            "oracle_union_pp": float(union.mean() * 100),
            "jaccard_correct": float(inter.sum() / union.sum()) if union.sum() else 1.0,
        })
    rows.extend(pair_rows)
    best_pair = max(pair_rows, key=lambda r: r["oracle_union"])
    panel_block["best_pair"] = {k: best_pair[k] for k in
                                ("method_a", "method_b", "oracle_union_pp")}
    panel_block["best_pair"]["gain_vs_best_single_pp"] = (
        best_pair["oracle_union_pp"] -
        max(panel_block["per_method_correct"].values()) / n * 100.0)

    # pre-registered families + full union
    unions = {}
    for family in TOP_FAMILIES:
        mask = np.zeros(n, dtype=bool)
        for member in family:
            mask |= correct[member]
        unions["+".join(m.split("_")[0] for m in family)] = round(
            float(mask.mean() * 100), 2)
    full = np.zeros(n, dtype=bool)
    for mask in correct.values():
        full |= mask
    unions["all_12"] = round(float(full.mean() * 100), 2)
    panel_block["oracle_unions_pp"] = unions
    panel_block["best_single_pp"] = round(
        max(panel_block["per_method_correct"].values()) / n * 100, 2)

    # near-subset complementarity for the headline family
    fam = TOP_FAMILIES[0]
    mask = np.zeros(n, dtype=bool)
    for member in fam:
        mask |= correct[member]
    panel_block["near_family_oracle_pp"] = round(float(mask[near].mean() * 100), 2)
    panel_block["near_best_single_pp"] = round(
        max(int((correct[m][near]).sum()) for m in fam) / int(near.sum()) * 100, 2)
    report["panels"][panel] = panel_block

pd.DataFrame(rows).to_csv(
    OUT / "method_complementarity_pairs.csv", index=False)
(OUT / "method_complementarity.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
