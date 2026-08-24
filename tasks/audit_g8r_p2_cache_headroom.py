"""Read-only structural and headroom audit for the frozen P2 cache.

The RAW-feature oracle reported here is deliberately optimistic: it asks
whether at least one predeclared similarity feature could rank the positive
first for a query.  It is a necessary data-signal check, not deployable model
performance and not permission to inspect P3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from g8r_p2_listwise_core import deterministic_formula_fold, evaluate_query_scores


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data/validation/g8r_p2_listwise_cache.npz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2_cache_headroom.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--target-delta", type=float, default=0.04)
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray) -> dict:
    return {
        "min": int(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(np.max(values)),
    }


def main() -> None:
    a = parse_args()
    if a.folds < 3 or not 0 < a.target_delta < 1:
        raise ValueError("invalid audit parameter")
    cache_audit_path = a.cache.with_suffix(".json")
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(a.cache)
    if cache_audit.get("status") != "g8r_p2_listwise_cache_built":
        raise RuntimeError("cache build audit is invalid")
    if cache_audit.get("cache_sha256") != actual_hash:
        raise RuntimeError("cache hash does not match its build audit")

    with np.load(a.cache, allow_pickle=True) as body:
        feature_names = list(map(str, body["feature_names"]))
        features = np.asarray(body["features"], dtype=np.float64)
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
        molecule_label = np.asarray(body["molecule_label"], dtype=np.int8)
        query_ik14 = np.asarray(body["query_ik14"], dtype=object)
        query_formula = np.asarray(body["query_formula"], dtype=object)
        query_has_near = np.asarray(body["query_has_near"], dtype=bool)

    n_queries = len(query_ptr) - 1
    if n_queries != len(query_ik14) or n_queries != len(query_formula) or n_queries != len(query_has_near):
        raise RuntimeError("query arrays are not aligned")
    if query_ptr[0] != 0 or query_ptr[-1] != len(molecule_label):
        raise RuntimeError("query pointer is invalid")
    if molecule_ptr[0] != 0 or molecule_ptr[-1] != len(features):
        raise RuntimeError("molecule pointer is invalid")
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise RuntimeError("feature matrix is invalid")
    if np.any(~np.isfinite(features)):
        raise RuntimeError("non-finite P2 feature")

    base_top1 = np.zeros(n_queries, dtype=bool)
    base_mrr = np.zeros(n_queries, dtype=np.float64)
    base_margin = np.zeros(n_queries, dtype=np.float64)
    any_raw_correct = np.zeros(n_queries, dtype=bool)
    candidate_counts = np.zeros(n_queries, dtype=np.int64)
    positive_spectra = np.zeros(n_queries, dtype=np.int64)
    spectra_per_molecule = np.diff(molecule_ptr)
    per_feature_correct = np.zeros((n_queries, len(feature_names) - 1), dtype=bool)

    for query in range(n_queries):
        molecule_left, molecule_right = map(int, query_ptr[query:query + 2])
        labels = molecule_label[molecule_left:molecule_right]
        if len(labels) < 2 or labels.sum() != 1 or labels[0] != 1:
            raise RuntimeError(f"query {query} does not have exactly one first-position positive")
        pair_left = int(molecule_ptr[molecule_left])
        pair_right = int(molecule_ptr[molecule_right])
        local_ptr = molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        result = evaluate_query_scores(features[pair_left:pair_right, 0], local_ptr, 0)
        base_top1[query] = bool(result["top1"])
        base_mrr[query] = float(result["mrr"])
        base_margin[query] = float(result["margin"])
        candidate_counts[query] = molecule_right - molecule_left
        positive_spectra[query] = int(local_ptr[1] - local_ptr[0])
        for feature_index in range(1, len(feature_names)):
            feature_result = evaluate_query_scores(
                features[pair_left:pair_right, feature_index], local_ptr, 0,
            )
            per_feature_correct[query, feature_index - 1] = bool(feature_result["top1"])
        any_raw_correct[query] = bool(np.any(per_feature_correct[query]))

    fold = np.asarray([deterministic_formula_fold(str(value), a.folds) for value in query_formula])
    formula_to_fold = {}
    for formula, value in zip(query_formula, fold):
        previous = formula_to_fold.setdefault(str(formula), int(value))
        if previous != int(value):
            raise RuntimeError("one formula was assigned to multiple folds")

    required_corrections = int(math.ceil(a.target_delta * n_queries))
    baseline_errors = ~base_top1
    recoverable_errors = baseline_errors & any_raw_correct
    # A score residual bounded by +/-d can improve a positive-vs-negative
    # margin by at most 2d.  This is an exact necessary geometric condition,
    # independent of optimizer or model capacity.
    reachability = {}
    for bound in (0.03, 0.06, 0.10):
        reachable = baseline_errors & (base_margin > -2.0 * bound)
        reachable_with_raw_signal = reachable & any_raw_correct
        reachability[f"delta_bound_{bound:.2f}"] = {
            "geometrically_reachable_errors": int(reachable.sum()),
            "reachable_errors_with_raw_oracle_signal": int(reachable_with_raw_signal.sum()),
            "optimistic_delta_from_reachable_raw_signal": float(reachable_with_raw_signal.mean()),
        }
    per_fold = []
    for value in range(a.folds):
        mask = fold == value
        near = mask & query_has_near
        per_fold.append({
            "fold": value,
            "n_queries": int(mask.sum()),
            "n_formulas": int(len(set(query_formula[mask]))),
            "n_near": int(near.sum()),
            "baseline_recall1": float(base_top1[mask].mean()),
            "baseline_errors": int((mask & baseline_errors).sum()),
            "raw_oracle_recoverable_errors": int((mask & recoverable_errors).sum()),
        })

    feature_results = {}
    for index, name in enumerate(feature_names[1:]):
        correct = per_feature_correct[:, index]
        feature_results[name] = {
            "recall1": float(correct.mean()),
            "corrected_vs_dreams": int(np.sum(baseline_errors & correct)),
            "introduced_vs_dreams": int(np.sum(base_top1 & (~correct))),
        }

    report = {
        "status": "g8r_p2_cache_headroom_passed",
        "cache_sha256": actual_hash,
        "n_queries": n_queries,
        "n_identities": int(len(set(query_ik14))),
        "n_formulas": int(len(set(query_formula))),
        "n_near_queries": int(query_has_near.sum()),
        "baseline": {
            "recall1": float(base_top1.mean()),
            "mrr": float(base_mrr.mean()),
            "n_errors": int(baseline_errors.sum()),
            "near_recall1": float(base_top1[query_has_near].mean()),
            "near_errors": int(np.sum(query_has_near & baseline_errors)),
            "maximum_possible_delta": float(baseline_errors.mean()),
        },
        "four_point_target": {
            "target_delta": a.target_delta,
            "required_net_corrections": required_corrections,
            "available_baseline_errors": int(baseline_errors.sum()),
            "errors_correctly_ranked_by_at_least_one_raw_feature": int(recoverable_errors.sum()),
            "optimistic_raw_oracle_delta": float(recoverable_errors.mean()),
            "warning": "Per-query best-feature oracle uses the answer to choose a feature; it is headroom only, not a model result.",
        },
        "bounded_residual_reachability": reachability,
        "baseline_error_margin": {
            "p10": float(np.percentile(base_margin[baseline_errors], 10)),
            "median": float(np.median(base_margin[baseline_errors])),
            "p90": float(np.percentile(base_margin[baseline_errors], 90)),
        },
        "candidate_molecules_per_query": percentile(candidate_counts),
        "spectra_per_candidate_molecule": percentile(spectra_per_molecule),
        "positive_spectra_per_query": percentile(positive_spectra),
        "queries_per_identity": percentile(np.unique(query_ik14, return_counts=True)[1]),
        "per_formula_fold": per_fold,
        "single_raw_feature_diagnostics": feature_results,
    }
    gates = {
        "queries_ge_5000": n_queries >= 5000,
        "identities_ge_2500": len(set(query_ik14)) >= 2500,
        "formulas_ge_1000": len(set(query_formula)) >= 1000,
        "near_queries_ge_1500": int(query_has_near.sum()) >= 1500,
        "baseline_errors_cover_four_points": int(baseline_errors.sum()) >= required_corrections,
        "raw_oracle_headroom_ge_four_points": int(recoverable_errors.sum()) >= required_corrections,
        "bound_006_raw_reachable_ge_four_points": (
            reachability["delta_bound_0.06"]["reachable_errors_with_raw_oracle_signal"]
            >= required_corrections
        ),
        "every_fold_has_100_near": all(row["n_near"] >= 100 for row in per_fold),
        "every_fold_has_20_errors": all(row["baseline_errors"] >= 20 for row in per_fold),
    }
    gates["pass"] = all(gates.values())
    report["gates"] = gates
    report["status"] = "g8r_p2_cache_headroom_passed" if gates["pass"] else "g8r_p2_cache_headroom_failed"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gates["pass"]:
        raise RuntimeError(f"P2 cache/headroom audit failed; see {a.output}")
    print(f"[P2 cache/headroom] PASS: {a.output}")


if __name__ == "__main__":
    main()
