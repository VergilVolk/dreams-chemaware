"""Fixed-configuration replay of the typed BioAware factor graph.

The two bundled cohorts are consumed development/mechanism audits.  This script
therefore reports transitions but never labels the result external or SOTA.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware_factor_graph import (
    UNKNOWN,
    CandidateVariable,
    PairFactor,
    TypedFactorGraph,
    relation_compatibility,
)


def strict_top(scores: np.ndarray, labels: np.ndarray) -> tuple[bool, int]:
    positive = np.flatnonzero(labels)
    if len(positive) != 1:
        raise RuntimeError("each query must have exactly one positive candidate")
    index = int(positive[0])
    rank = 1 + int(np.sum(scores[~labels] >= scores[index]))
    return rank == 1, rank


def logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return math.log(value / (1.0 - value))


def formula_bootstrap(transitions: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = transitions.groupby("truth_formula", sort=False).delta.mean()
    values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for i in range(repeats):
        draws[i] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "mean": float(transitions.delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "formula_clusters": int(len(values)),
        "resamples": int(repeats),
    }


def evaluate_cohort(root: Path, cohort: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    candidates = pd.read_csv(root / f"{cohort}__candidates.csv.gz")
    edges = pd.read_csv(root / f"{cohort}__edges.csv.gz")
    labels = pd.read_csv(root / f"{cohort}__labels.csv.gz")
    candidates = candidates.merge(
        labels[["query_id", "candidate_id", "truth_candidate_id", "truth_formula", "is_positive"]],
        on=["query_id", "candidate_id"], how="left", validate="one_to_one",
    )
    if candidates.is_positive.isna().any():
        raise RuntimeError(f"{cohort}: candidate/label mismatch")

    variables: list[CandidateVariable] = []
    query_variables: dict[str, CandidateVariable] = {}
    query_order: dict[str, pd.DataFrame] = {}
    for query_id, group in candidates.groupby("query_id", sort=False):
        group = group.reset_index(drop=True)
        values = group.spectral_score.to_numpy(float)
        centered = (values - values.max()) / args.spectral_temperature
        unknown = -abs(args.unknown_margin) / args.spectral_temperature
        variable = CandidateVariable(
            node_id=f"query:{query_id}",
            candidates=tuple(group.candidate_id.astype(str)) + (UNKNOWN,),
            unary_log_score=np.concatenate((centered, [unknown])),
        )
        variables.append(variable)
        query_variables[str(query_id)] = variable
        query_order[str(query_id)] = group

    seed_scores = edges.groupby(["seed_query_id", "seed_compound_id"], sort=False).experimental_support.max()
    seed_variables: dict[tuple[str, str], CandidateVariable] = {}
    for (seed_query_id, seed_compound_id), probability in seed_scores.items():
        key = (str(seed_query_id), str(seed_compound_id))
        variable = CandidateVariable(
            node_id=f"seed:{key[0]}:{key[1]}",
            candidates=(key[1], UNKNOWN),
            unary_log_score=np.asarray([0.0, -max(0.0, logit(float(probability)))]),
        )
        variables.append(variable)
        seed_variables[key] = variable

    factors: list[PairFactor] = []
    for index, row in enumerate(edges.itertuples(index=False)):
        query = query_variables[str(row.query_id)]
        seed = seed_variables[(str(row.seed_query_id), str(row.seed_compound_id))]
        confidence = (
            float(np.clip(row.path_confidence, 0, 1))
            * math.sqrt(float(np.clip(row.experimental_support, 0, 1)))
            * (0.25 + 0.75 * float(np.clip(row.reaction_completeness, 0, 1)))
            * (1.0 - float(np.clip(row.conflict, 0, 1)))
        )
        if confidence <= 0:
            continue
        supported = {tuple(sorted((str(row.candidate_id), str(row.seed_compound_id))))}
        compatibility = relation_compatibility(
            query, seed, supported, reward=args.relation_reward, unknown_value=0.0,
        )
        factors.append(PairFactor(
            factor_id=f"{cohort}:{index}:{row.dependency_key}",
            left=query.node_id, right=seed.node_id, compatibility=compatibility,
            family=str(row.relation_name), confidence=confidence,
        ))

    graph = TypedFactorGraph(variables, factors, damping=args.damping, message_cap=args.message_cap)
    inference = graph.infer(iterations=args.iterations, tolerance=args.tolerance)
    rows = []
    for query_id, group in query_order.items():
        labels_array = group.is_positive.to_numpy(bool)
        spectral = group.spectral_score.to_numpy(float)
        baseline_correct, baseline_rank = strict_top(spectral, labels_array)
        decision = inference["decisions"][f"query:{query_id}"]
        beliefs = np.asarray(decision["belief"][:-1], dtype=float)
        graph_correct, graph_rank = strict_top(beliefs, labels_array)
        winner = str(decision["candidate_id"])
        rows.append({
            "query_id": query_id,
            "truth_candidate_id": str(group.truth_candidate_id.iloc[0]),
            "truth_formula": str(group.truth_formula.iloc[0]),
            "baseline_rank": baseline_rank,
            "graph_rank": graph_rank,
            "baseline_correct": baseline_correct,
            "graph_correct": graph_correct,
            "corrected": (not baseline_correct) and graph_correct,
            "introduced": baseline_correct and (not graph_correct),
            "graph_abstained": bool(decision["abstained"]),
            "graph_winner": winner,
            "graph_margin": float(decision["margin"]),
            "delta": float(graph_correct) - float(baseline_correct),
        })
    transitions = pd.DataFrame(rows)
    result = {
        "queries": int(len(transitions)),
        "baseline_recall1": float(transitions.baseline_correct.mean()),
        "factor_graph_recall1": float(transitions.graph_correct.mean()),
        "delta_recall1": float(transitions.delta.mean()),
        "corrected": int(transitions.corrected.sum()),
        "introduced": int(transitions.introduced.sum()),
        "abstained": int(transitions.graph_abstained.sum()),
        "variables": int(len(variables)),
        "seed_variables": int(len(seed_variables)),
        "factors": int(len(factors)),
        "converged": bool(inference["converged"]),
        "iterations": int(inference["iterations"]),
        "formula_cluster_bootstrap": formula_bootstrap(transitions, args.bootstrap_resamples, args.seed),
    }
    return result, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/validation/bioaware_context_evidence_tensor_20260830"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_typed_factor_graph_replay_20260830"))
    parser.add_argument("--cohorts", nargs="+", default=["mtbls13729_expanded", "mtbls1905_auto"])
    parser.add_argument("--spectral-temperature", type=float, default=0.10)
    parser.add_argument("--unknown-margin", type=float, default=0.08)
    parser.add_argument("--relation-reward", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--message-cap", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for cohort in args.cohorts:
        result, transitions = evaluate_cohort(args.tensor_dir, cohort, args)
        results[cohort] = result
        transitions.to_csv(args.output_dir / f"{cohort}__transitions.csv.gz", index=False)
    report = {
        "status": "bioaware_typed_factor_graph_replay_complete",
        "formal": False,
        "protocol": "fixed typed max-product; explicit unknown; dependency-collapsed edges; consumed development cohorts",
        "results": results,
        "configuration": {
            "spectral_temperature": args.spectral_temperature,
            "unknown_margin": args.unknown_margin,
            "relation_reward": args.relation_reward,
            "damping": args.damping,
            "message_cap": args.message_cap,
            "iterations": args.iterations,
        },
        "contracts": {
            "phenotype_blind": True,
            "truth_not_used_by_inference": True,
            "identity_noop_filtered": True,
            "missing_relation_is_unknown": True,
            "consumed_development_only": True,
        },
        "claim_limit": "Mechanism replay on consumed cohorts; no external or SOTA claim is permitted.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
