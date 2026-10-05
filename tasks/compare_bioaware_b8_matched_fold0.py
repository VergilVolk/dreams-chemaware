#!/usr/bin/env python
"""Compare B8 graph-selected and matched-generic shared-encoder fold-0 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    total = int(left_only + right_only)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1))
    return float(min(1.0, 2.0 * tail / (2.0 ** total)))


def formula_bootstrap(
    delta: np.ndarray,
    formula: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    unique = np.unique(formula.astype(str))
    by_formula = {
        value: np.flatnonzero(formula.astype(str) == value) for value in unique
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([delta[by_formula[str(value)]] for value in sampled])
        draws[repeat] = float(np.mean(values))
    return {
        "mean_delta": float(np.mean(delta)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(repeats),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--generic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def read_result(directory: Path, expected_arm: str) -> tuple[dict, pd.DataFrame]:
    report_path = directory / "report.json"
    query_path = directory / "held_per_query.csv.gz"
    for path in (report_path, query_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(query_path)
    if (report.get("status") != "bioaware_b4_direct_shared_embedding_fold_complete"
            or report.get("arm") != expected_arm
            or report.get("supervision_mode") != "direct_onehot"
            or int(report.get("outer_fold", -1)) != 0):
        raise RuntimeError(f"wrong B8 result identity for {expected_arm}")
    if report["official_replay"]["rank_mismatches"] != 0:
        raise RuntimeError(f"official replay failed for {expected_arm}")
    if set(frame["arm"].astype(str)) != {expected_arm} or frame["query_id"].duplicated().any():
        raise RuntimeError(f"invalid per-query result for {expected_arm}")
    return report, frame


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    router_path = args.router_dir / "report.json"
    if not router_path.is_file():
        raise FileNotFoundError(router_path)
    router = json.loads(router_path.read_text(encoding="utf-8"))
    if (router.get("status") != "bioaware_b8_matched_error_routers_frozen"
            or not router.get("pass_to_fold0_training")
            or not router.get("outer_folds", {}).get("0", {}).get("pass_to_training")):
        raise RuntimeError("B8 matched router did not pass fold 0")
    graph, graph_query = read_result(args.graph_dir, "matched_graph_direct")
    generic, generic_query = read_result(args.generic_dir, "matched_generic_direct")

    # The only permitted differences are the selected action arm and paths.
    ignored = {"arm", "output", "action_router_dir"}
    graph_optimization = {
        key: value for key, value in graph["optimization"].items() if key not in ignored
    }
    generic_optimization = {
        key: value for key, value in generic["optimization"].items() if key not in ignored
    }
    if graph_optimization != generic_optimization:
        raise RuntimeError("graph/generic optimizer or schedule configuration differs")
    mass_keys = (
        "held_queries", "held_formulas", "train_queries", "train_corrective_queries",
        "train_corrective_identities", "train_safety_queries", "train_known_harm_queries",
    )
    if any(graph[key] != generic[key] for key in mass_keys):
        raise RuntimeError("graph/generic training or held mass differs")
    if graph["baseline"] != generic["baseline"]:
        raise RuntimeError("graph/generic held baseline differs")

    left = graph_query.sort_values("query_id", kind="stable").reset_index(drop=True)
    right = generic_query.sort_values("query_id", kind="stable").reset_index(drop=True)
    if (not np.array_equal(left["query_id"].astype(str), right["query_id"].astype(str))
            or not np.array_equal(left["old_rank"].to_numpy(int), right["old_rank"].to_numpy(int))
            or not np.array_equal(left["truth_formula"].astype(str), right["truth_formula"].astype(str))):
        raise RuntimeError("held query universe or official ranks differ")
    graph_top = left["new_rank"].to_numpy(int) == 1
    generic_top = right["new_rank"].to_numpy(int) == 1
    graph_mrr = 1.0 / left["new_rank"].to_numpy(float)
    generic_mrr = 1.0 / right["new_rank"].to_numpy(float)
    r1_delta = graph_top.astype(float) - generic_top.astype(float)
    mrr_delta = graph_mrr - generic_mrr
    graph_only = int(np.sum(graph_top & ~generic_top))
    generic_only = int(np.sum(~graph_top & generic_top))
    graph_metrics = graph["shared_embedding"]
    generic_metrics = generic["shared_embedding"]
    comparison = {
        "graph_minus_generic_recall1": float(np.mean(r1_delta)),
        "graph_minus_generic_mrr": float(np.mean(mrr_delta)),
        "graph_only_correct": graph_only,
        "generic_only_correct": generic_only,
        "mcnemar_exact_p": exact_mcnemar(graph_only, generic_only),
        "recall1_formula_cluster_bootstrap": formula_bootstrap(
            r1_delta, left["truth_formula"].to_numpy(str),
            args.bootstrap_resamples, args.seed,
        ),
        "mrr_formula_cluster_bootstrap": formula_bootstrap(
            mrr_delta, left["truth_formula"].to_numpy(str),
            args.bootstrap_resamples, args.seed + 1,
        ),
        "risk_net_lambda2_advantage": int(
            graph_metrics["risk_net_lambda2"] - generic_metrics["risk_net_lambda2"]
        ),
        "introduced_advantage": int(
            generic_metrics["introduced"] - graph_metrics["introduced"]
        ),
    }
    gates = {
        "matched_router_passed": True,
        "training_mass_exactly_equal": True,
        "official_and_held_protocol_identical": True,
        "graph_recall1_strictly_better": comparison["graph_minus_generic_recall1"] > 0,
        "graph_mrr_noninferior": comparison["graph_minus_generic_mrr"] >= 0,
        "graph_risk_net_strictly_better": comparison["risk_net_lambda2_advantage"] > 0,
        "graph_introduces_no_more_errors": comparison["introduced_advantage"] >= 0,
        "both_preservation_ge_0_985": (
            graph_metrics["preservation"] >= 0.985
            and generic_metrics["preservation"] >= 0.985
        ),
    }
    report = {
        "status": "bioaware_b8_matched_fold0_comparison_complete",
        "formal": False,
        "protocol": (
            "paired identity-cluster graph-selected versus generic-hard-error "
            "shared-encoder fold-0 control"
        ),
        "graph_selected": graph_metrics,
        "matched_generic": generic_metrics,
        "comparison": comparison,
        "gates": {key: bool(value) for key, value in gates.items()},
        "pass_to_remaining_formula_folds": bool(all(gates.values())),
        "contracts": {
            "same_shared_encoder_training_recipe": True,
            "same_identity_and_query_training_mass": True,
            "same_held_queries": True,
            "direct_true_identity_supervision": True,
            "graph_score_not_in_loss": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "router_report_sha256": sha256_file(router_path),
            "graph_report_sha256": sha256_file(args.graph_dir / "report.json"),
            "graph_per_query_sha256": sha256_file(args.graph_dir / "held_per_query.csv.gz"),
            "generic_report_sha256": sha256_file(args.generic_dir / "report.json"),
            "generic_per_query_sha256": sha256_file(args.generic_dir / "held_per_query.csv.gz"),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "One opened formula fold. Passing only licenses the remaining OOF folds; "
            "it is not reaction-specific evidence or external performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
