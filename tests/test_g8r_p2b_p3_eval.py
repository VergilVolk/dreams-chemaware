import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

from eval_g8r_p2b_on_sealed_p3 import (  # noqa: E402
    exact_mcnemar,
    macro_auc,
    manifest_hash,
    paired_cluster_bootstrap,
)


def test_macro_auc_handles_ties_and_multiple_negatives():
    assert macro_auc(np.asarray([0.5, 0.5, 0.2, 0.8])) == 0.5


def test_exact_mcnemar_is_symmetric_and_detects_clean_gain():
    assert exact_mcnemar(7, 2) == exact_mcnemar(2, 7)
    assert exact_mcnemar(20, 0) < 1e-4
    assert exact_mcnemar(0, 0) == 1.0


def test_manifest_hash_matches_builder_contract():
    body = {"panel": "x", "queries": [{"row": 1, "candidate_rows": [2, 3]}]}
    from eval_g8r_p2b_on_sealed_p3 import sha256_json
    body["query_manifest_sha256"] = sha256_json(body)
    assert manifest_hash(body) == body["query_manifest_sha256"]


def test_formula_cluster_bootstrap_preserves_paired_direction():
    result = paired_cluster_bootstrap(
        np.asarray(["A", "A", "B", "C"], dtype=object),
        np.asarray([1, 1, 1, 1]),
        np.asarray([0, 0, 0, 0]),
        200,
        7,
    )
    assert result["mean_delta"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0
