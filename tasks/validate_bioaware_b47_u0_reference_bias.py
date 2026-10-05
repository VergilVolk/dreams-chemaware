#!/usr/bin/env python
"""Independent fail-closed validation for the B47-U0 nuisance audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


EXPECTED_CONTRACTS = {
    "truth_opened": False,
    "phenotype_used": False,
    "reaction_network_used": False,
    "algorithm_outputs_opened": False,
    "model_fitted": False,
    "aggregation_selected": False,
    "queries_or_candidates_removed_after_scoring": False,
    "P2b_used": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    paths = {
        "report": directory / "report.json",
        "candidate_aggregates": directory / "candidate_aggregates.csv.gz",
        "per_query": directory / "per_query.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b47_u0_reference_multiplicity_adduct_audit_complete":
        raise RuntimeError("unexpected B47-U0 status")
    if report.get("protocol_version") != "B47-U0-v2-candidate-exposure":
        raise RuntimeError("unexpected B47-U0 protocol version")
    if report.get("formal") is not True or report.get("contracts") != EXPECTED_CONTRACTS:
        raise RuntimeError("B47-U0 formal/truth-blind contract changed")
    if not report.get("pass_to_u1_strong_unary_bakeoff"):
        raise RuntimeError("B47-U0 engineering gate failed")
    if not all(report.get("engineering_gates", {}).values()):
        raise RuntimeError("one or more B47-U0 engineering gates are false")
    provenance = report.get("provenance", {})
    for key in ("candidate_aggregates", "per_query"):
        if provenance.get(f"{key}_sha256") != sha256_file(paths[key]):
            raise RuntimeError(f"B47-U0 output hash mismatch: {key}")

    candidates = pd.read_csv(paths["candidate_aggregates"])
    queries = pd.read_csv(paths["per_query"])
    if len(candidates) != int(report["candidate_query_pairs"]):
        raise RuntimeError("B47-U0 candidate aggregate count mismatch")
    if candidates["candidate_id"].nunique() != int(report["unique_candidate_identities_global"]):
        raise RuntimeError("B47-U0 global candidate-identity count mismatch")
    if len(queries) != int(report["queries"]):
        raise RuntimeError("B47-U0 per-query count mismatch")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B47-U0 candidate keys are duplicated")
    if queries["query_id"].duplicated().any():
        raise RuntimeError("B47-U0 query keys are duplicated")
    required_candidate = {
        "reference_count", "max_score", "mean_score", "max_minus_mean",
        "reference_adduct", "expected_max_2", "expected_max_4", "expected_max_8",
    }
    if not required_candidate.issubset(candidates.columns):
        raise RuntimeError("B47-U0 candidate output columns are incomplete")
    finite = candidates[["max_score", "mean_score", "max_minus_mean"]].to_numpy(float)
    if not np.isfinite(finite).all():
        raise RuntimeError("B47-U0 candidate summaries contain non-finite core scores")
    if (candidates["reference_count"] < 1).any():
        raise RuntimeError("B47-U0 reference count below one")
    if (candidates["max_minus_mean"] < -1e-9).any():
        raise RuntimeError("B47-U0 max score below expected-single score")
    if not np.allclose(
        candidates["max_score"] - candidates["mean_score"],
        candidates["max_minus_mean"], atol=2e-7, rtol=0,
    ):
        raise RuntimeError("B47-U0 max-lift arithmetic mismatch")
    for k in (2, 4, 8):
        values = candidates[f"expected_max_{k}"]
        eligible = candidates["reference_count"].ge(k)
        if values[eligible].isna().any() or values[~eligible].notna().any():
            raise RuntimeError(f"B47-U0 expected-max-{k} eligibility mismatch")
        if ((values[eligible] < candidates.loc[eligible, "mean_score"] - 1e-9)
                | (values[eligible] > candidates.loc[eligible, "max_score"] + 1e-9)).any():
            raise RuntimeError(f"B47-U0 expected-max-{k} leaves score bounds")
    observed_both = int(queries["has_both_adduct_branches"].astype(bool).sum())
    if observed_both != int(report["adduct_pooling"]["queries_with_both_adduct_branches"]):
        raise RuntimeError("B47-U0 both-adduct query count mismatch")
    exposure = report.get("candidate_exposure", {})
    if not {
        "candidate_identities_per_query",
        "candidate_reference_spectra_per_query",
        "queries_per_candidate_identity",
        "by_adduct",
    }.issubset(exposure):
        raise RuntimeError("B47-U0 candidate exposure report is incomplete")
    if set(exposure["by_adduct"]) != {"[M+H]+", "[M+Na]+"}:
        raise RuntimeError("B47-U0 adduct exposure branches changed")
    print(
        f"[validate_bioaware_b47_u0_reference_bias] PASS "
        f"queries={len(queries):,} candidates={len(candidates):,} "
        f"both-adduct={observed_both:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
