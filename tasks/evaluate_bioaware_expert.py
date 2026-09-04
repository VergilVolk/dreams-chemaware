#!/usr/bin/env python
"""Leakage-safe BioAware v1 evaluation with degree-preserving network decoys."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import (  # noqa: E402
    BioAwareConfig,
    aggregate_query_support,
    build_one_hop_evidence,
    compound_reaction_degree,
    degree_preserving_reaction_decoy,
    fuse_candidates,
    top1_transition_table,
    validate_reaction_participants,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def cluster_bootstrap_delta(
    per_query: pd.DataFrame,
    clusters: pd.Series,
    *,
    resamples: int,
    seed: int,
) -> dict:
    values = per_query["final_correct"].astype(int).to_numpy() - per_query["baseline_correct"].astype(int).to_numpy()
    cluster_values = pd.Series(clusters.astype(str).to_numpy(), index=np.arange(len(per_query)))
    groups = {key: idx.to_numpy() for key, idx in cluster_values.groupby(cluster_values).groups.items()}
    keys = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    for i in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        boot[i] = float(values[indices].mean())
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "clusters": int(len(keys)),
        "resamples": int(resamples),
    }


def run_once(
    candidates: pd.DataFrame,
    participants: pd.DataFrame,
    seeds: pd.DataFrame,
    config: BioAwareConfig,
    truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    paths = build_one_hop_evidence(participants, seeds, config)
    supported, explanations = aggregate_query_support(
        candidates,
        paths,
        truth_col=truth_col,
        exclude_same_query=True,
        exclude_truth_identity=True,
    )
    scored, query_decisions = fuse_candidates(supported, config)
    per_query, summary = top1_transition_table(scored, truth_col=truth_col)
    return scored, query_decisions, explanations, {**summary, "evidence_paths": int(len(explanations))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-col", default="truth_candidate_id")
    parser.add_argument("--formula-col", default="truth_formula")
    parser.add_argument("--network-weight", type=float, default=0.15)
    parser.add_argument("--max-spectral-margin", type=float, default=0.05)
    parser.add_argument("--min-network-advantage", type=float, default=0.10)
    parser.add_argument("--min-seed-score", type=float, default=0.80)
    parser.add_argument("--max-seed-degree", type=int, default=250)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--decoy-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow fewer than 10 network decoys and mark the report non-formal",
    )
    args = parser.parse_args()

    if args.decoy_repeats < 0:
        raise ValueError("--decoy-repeats must be nonnegative")
    if not args.development and args.decoy_repeats < 10:
        raise RuntimeError("formal BioAware evaluation requires at least 10 network decoys")

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    candidates = read_table(args.candidates)
    participants = read_table(args.participants)
    seeds = read_table(args.seeds)
    if args.truth_col not in candidates:
        raise RuntimeError(f"candidates missing truth column {args.truth_col!r}")
    config = BioAwareConfig(
        directed=args.directed,
        network_weight=args.network_weight,
        maximum_spectral_margin_for_override=args.max_spectral_margin,
        minimum_network_advantage=args.min_network_advantage,
        minimum_seed_score=args.min_seed_score,
        maximum_seed_degree=args.max_seed_degree,
    )

    normalized_participants = validate_reaction_participants(participants)
    degree = compound_reaction_degree(normalized_participants)
    currency = set(
        normalized_participants.loc[normalized_participants["is_currency"], "compound_id"].astype(str)
    )
    seed_degree = seeds["seed_compound_id"].astype(str).map(degree).fillna(0).astype(int)
    seed_score = pd.to_numeric(seeds["seed_score"], errors="raise")
    eligible_seed_mask = (
        (seed_score >= config.minimum_seed_score)
        & (seed_degree > 0)
        & (seed_degree <= config.maximum_seed_degree)
        & (~seeds["seed_compound_id"].astype(str).isin(currency))
    )
    seed_audit = {
        "input_rows": int(len(seeds)),
        "input_compounds": int(seeds["seed_compound_id"].astype(str).nunique()),
        "score_eligible_rows": int((seed_score >= config.minimum_seed_score).sum()),
        "graph_degree_eligible_rows": int(((seed_degree > 0) & (seed_degree <= config.maximum_seed_degree)).sum()),
        "currency_rows": int(seeds["seed_compound_id"].astype(str).isin(currency).sum()),
        "fully_eligible_rows": int(eligible_seed_mask.sum()),
        "fully_eligible_compounds": int(
            seeds.loc[eligible_seed_mask, "seed_compound_id"].astype(str).nunique()
        ),
    }

    scored, decisions, explanations, real = run_once(
        candidates, participants, seeds, config, args.truth_col
    )
    per_query, _ = top1_transition_table(scored, truth_col=args.truth_col)
    query_meta = candidates.groupby("query_id", sort=False).first().reset_index()
    per_query = per_query.merge(query_meta, on="query_id", how="left", validate="one_to_one")
    if args.formula_col in per_query:
        clusters = per_query[args.formula_col].fillna(per_query["query_id"]).astype(str)
        cluster_name = args.formula_col
    else:
        clusters = per_query["query_id"].astype(str)
        cluster_name = "query_id"
    real["cluster_bootstrap"] = cluster_bootstrap_delta(
        per_query, clusters, resamples=args.bootstrap_resamples, seed=args.seed
    )
    corrected = int(real["corrected"])
    introduced = int(real["introduced"])
    real["mcnemar_exact_p"] = float(
        binomtest(min(corrected, introduced), corrected + introduced, 0.5).pvalue
    ) if corrected + introduced else 1.0
    real["intervention_rate"] = float(decisions["bioaware_applied"].mean())
    real["conflicts_abstained"] = int(
        (decisions["evidence_state"] == "spectral_strong_network_conflict").sum()
    )

    decoy_rows = []
    for repeat in range(args.decoy_repeats):
        decoy = degree_preserving_reaction_decoy(
            participants, seed=args.seed + repeat + 1, swaps_per_edge=5
        )
        _, _, _, decoy_summary = run_once(candidates, decoy, seeds, config, args.truth_col)
        decoy_rows.append(
            {
                "repeat": repeat,
                "delta_recall1": decoy_summary["delta_recall1"],
                "corrected": decoy_summary["corrected"],
                "introduced": decoy_summary["introduced"],
                "accepted_swaps": int(decoy.attrs.get("decoy_swaps_accepted", -1)),
            }
        )
        print(
            f"[decoy {repeat + 1}/{args.decoy_repeats}] delta={decoy_summary['delta_recall1']:+.4f}",
            flush=True,
        )
    decoys = pd.DataFrame(decoy_rows)
    decoy_95 = float(decoys["delta_recall1"].quantile(0.95)) if len(decoys) else None
    empirical_p = float(
        (1 + (decoys["delta_recall1"] >= real["delta_recall1"]).sum()) / (1 + len(decoys))
    ) if len(decoys) else None

    gates = {
        "corrected_gt_introduced": corrected > introduced,
        "formula_or_query_cluster_ci_positive": real["cluster_bootstrap"]["ci_low"] > 0,
        "beats_degree_preserving_decoy_p95": bool(
            len(decoys) and decoy_95 is not None and real["delta_recall1"] > decoy_95
        ),
        "decoy_empirical_p_le_0_10": bool(
            len(decoys) and empirical_p is not None and empirical_p <= 0.10
        ),
    }
    report = {
        "status": "bioaware_v1_evaluation_complete",
        "formal": not args.development,
        "protocol": "one-hop reaction-hypergraph evidence; leave-query-out and leave-truth-identity-out; phenotype-blind; fixed conservative override gate",
        "real_network": real,
        "degree_preserving_decoys": {
            "repeats": int(len(decoys)),
            "delta_mean": float(decoys["delta_recall1"].mean()) if len(decoys) else None,
            "delta_p95": decoy_95,
            "empirical_p": empirical_p,
        },
        "gates": {**gates, "pass": all(gates.values())},
        "cluster_column": cluster_name,
        "configuration": config.to_dict(),
        "seed_audit": seed_audit,
        "provenance": {
            "candidates": str(args.candidates.resolve()),
            "candidates_sha256": sha256(args.candidates.resolve()),
            "participants": str(args.participants.resolve()),
            "participants_sha256": sha256(args.participants.resolve()),
            "seeds": str(args.seeds.resolve()),
            "seeds_sha256": sha256(args.seeds.resolve()),
        },
        "claim_limit": "A passing result establishes incremental candidate-ranking evidence on this held-out protocol. It does not establish MSI Level 2 identity, flux, enzyme activity, or shared-embedding improvement.",
    }
    scored.to_csv(out / "candidate_scores.csv.gz", index=False)
    decisions.to_csv(out / "query_decisions.csv", index=False)
    explanations.to_csv(out / "evidence_paths.csv.gz", index=False)
    per_query.to_csv(out / "per_query_transitions.csv", index=False)
    decoys.to_csv(out / "degree_preserving_decoys.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
