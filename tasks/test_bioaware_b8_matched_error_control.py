#!/usr/bin/env python
"""Unit and static contract checks for the corrected BioAware B8 control."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from build_bioaware_b8_matched_error_routers import (  # noqa: E402
    GENERIC_ARM,
    GRAPH_ARM,
    NUMERIC_FEATURES,
    attach_training_graph,
    match_identity_clusters,
    matching_balance,
    minimum_cost_bipartite,
    route_frame,
    robust_scales,
)
from compare_bioaware_b8_matched_fold0 import exact_mcnemar, formula_bootstrap  # noqa: E402


def row(query: str, identity: str, formula: str, unit: str, margin: float,
        gap: float, candidates: int, repetitions: int, rank: int = 2) -> dict:
    return {
        "query_id": query,
        "truth_candidate_id": identity,
        "truth_formula": formula,
        "unit_id": unit,
        "truth_margin": margin,
        "baseline_gap": gap,
        "candidate_count": candidates,
        "positive_reference_count": 2,
        "identity_query_count": repetitions,
        "training_candidate_count": candidates,
        "training_baseline_rank_numeric": float(rank),
        "training_truth_margin": margin,
        "training_baseline_gap": gap,
        "log_candidate_count": float(np.log1p(candidates)),
        "log_positive_reference_count": float(np.log1p(2)),
        "log_identity_query_count": float(np.log1p(repetitions)),
        "baseline_rank_numeric": float(rank),
        "baseline_rank_bucket": 1,
        "candidate_count_bucket": 2,
        "identity_count_bucket": 1,
        "positive_reference_count_bucket": 1,
    }


def main() -> None:
    # The residual solver must repair a tempting greedy assignment.
    assert minimum_cost_bipartite(np.asarray([[1.0, 2.0], [1.1, 100.0]])) == [
        (0, 1), (1, 0)
    ]
    graph = pd.DataFrame([
        row("g1a", "G1", "GF1", "u1", -0.04, 0.02, 4, 2),
        row("g1b", "G1", "GF1", "u2", -0.03, 0.03, 4, 2),
        row("g2", "G2", "GF2", "u1", -0.08, 0.01, 3, 1),
    ])
    pool = pd.DataFrame([
        row("x1a", "X1", "XF1", "u1", -0.05, 0.02, 4, 2),
        row("x1b", "X1", "XF1", "u2", -0.02, 0.03, 4, 2),
        row("x2", "X2", "XF2", "u1", -0.07, 0.01, 3, 1),
        # Same-formula and wrong-unit decoys must not be used.
        row("bad_formula", "X3", "GF2", "u1", -0.08, 0.01, 3, 1),
        row("bad_unit", "X4", "XF4", "u9", -0.04, 0.02, 4, 2),
    ])
    scales = robust_scales(pd.concat((graph, pool), ignore_index=True))
    matches, unmatched = match_identity_clusters(graph, pool, scales)
    assert not unmatched
    # Exactly one representative query is retained per matched identity.
    assert len(matches) == 2
    assert matches["graph_identity"].nunique() == 2
    assert matches["generic_identity"].nunique() == 2
    assert "bad_formula" not in set(matches["generic_query_id"])
    assert "bad_unit" not in set(matches["generic_query_id"])
    lookup = pd.concat((graph, pool), ignore_index=True).set_index("query_id")
    for item in matches.itertuples(index=False):
        assert lookup.loc[item.graph_query_id, "unit_id"] == lookup.loc[item.generic_query_id, "unit_id"]
        assert lookup.loc[item.graph_query_id, "truth_formula"] != lookup.loc[item.generic_query_id, "truth_formula"]
    balance = matching_balance(matches, pd.concat((graph, pool), ignore_index=True))
    assert balance["identity_pairs"] == 2
    assert balance["balance_weighting"].startswith("one mean vector per matched")

    # If held-formula exclusion removes the only wrong candidate that beat the
    # truth, the query is no longer a corrective training example.
    filtered = attach_training_graph(
        pd.DataFrame({"query_id": ["q0"], "query_index": [0]}),
        {
            "query_ptr": np.asarray([0, 3]),
            "molecule_formula": np.asarray(["truth", "held", "other"]),
            "molecule_official_score": np.asarray([0.50, 0.60, 0.40]),
        },
        {"held"},
        0,
    )
    assert int(filtered.loc[0, "training_candidate_count"]) == 2
    assert int(filtered.loc[0, "training_baseline_rank_numeric"]) == 1
    assert not bool(filtered.loc[0, "training_eligible"])
    base_routes = pd.DataFrame({
        "query_id": ["z", "g2", "g1a"],
        "corrected": [False, True, True],
        "introduced": [True, False, False],
    })
    ordered = route_frame(base_routes, GRAPH_ARM, ["g1a", "g2"])
    assert ordered["query_id"].tolist()[:2] == ["g1a", "g2"]
    assert ordered["corrective_selected"].tolist() == [True, True, False]

    builder = (ROOT / "tasks/build_bioaware_b8_matched_error_routers.py").read_text(
        encoding="utf-8"
    )
    trainer = (ROOT / "tasks/train_bioaware_b4_direct_shared_embedding.py").read_text(
        encoding="utf-8"
    )
    sbatch = (ROOT / "tasks/run_bioaware_b8_matched_error_control.sbatch").read_text(
        encoding="utf-8"
    )
    ast.parse(builder)
    ast.parse(trainer)
    required_builder = (
        'GRAPH_ARM = "matched_graph_direct"',
        'GENERIC_ARM = "matched_generic_direct"',
        'identity_cluster_is_matching_unit',
        'one_representative_query_per_identity',
        'matching_graph_equals_backprop_graph',
        'biological_unit_exact_within_query_pairs',
        'same_graph_introduced_harm_queries_in_both_arms',
        '"P2b_used": False',
    )
    for token in required_builder:
        assert token in builder, token
    for token in (
        '"matched_graph_direct": (None, None)',
        '"matched_generic_direct": (None, None)',
        '"selection_column": "corrective_selected"',
        'args.arm in DIRECT_ROUTE_ARMS',
        'B8 matched arms require exactly one reference per candidate',
        'B8 matched arms require exactly one selected query per identity',
    ):
        assert token in trainer, token
    for token in (
        "#SBATCH --gpus=1",
        "matched_graph_direct",
        "matched_generic_direct",
        "--backbone-lr 2e-6",
        "--head-lr 1e-5",
        "--steps-per-epoch 128",
        "--references-per-molecule 1",
        "--no-amp",
        "set -euo pipefail",
    ):
        assert token in sbatch, token
    assert "#SBATCH --mem" not in sbatch
    assert all(name in graph for name in NUMERIC_FEATURES)
    assert GRAPH_ARM != GENERIC_ARM
    assert exact_mcnemar(0, 0) == 1.0
    assert np.isclose(exact_mcnemar(3, 0), 0.25)
    boot = formula_bootstrap(
        np.asarray([1.0, -1.0, 1.0]),
        np.asarray(["f1", "f1", "f2"]), 100, 7,
    )
    assert np.isclose(boot["mean_delta"], 1.0 / 3.0)
    assert boot["clusters"] == 2
    print("[BioAware B8 matched-error control checks] PASS")


if __name__ == "__main__":
    main()
