#!/usr/bin/env python
"""Freeze the conservative BioAware V3 router before RPLC validation.

The router is an abstaining union of two independently implemented experts:
the fixed depth-3 complete-path expert and the nested-OOF rank-consensus
expert.  Conflicting proposals always fall back to official DreaMS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES, RAW_FAMILIES, add_family_features, fit_family_weights,
    inner_oof_predictions, select_gate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = frame.groupby("truth_formula", sort=False).delta.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        rng.choice(grouped, size=len(grouped), replace=True).mean() for _ in range(repeats)
    ])
    return {
        "mean": float(frame.delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "formula_clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path(
        "data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz"))
    parser.add_argument("--depth3", type=Path, default=Path(
        "data/validation/bioaware_metdna3_candidate_edge_decision_v1/query_transitions.csv.gz"))
    parser.add_argument("--rank-oof", type=Path, default=Path(
        "data/validation/bioaware_rank_consensus_fusion_v2/query_oof_transitions.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_v3_consensus_router_frozen_v2_20260830"))
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--maximum-family-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    for path in (args.ledger, args.depth3, args.rank_oof):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")

    depth = pd.read_csv(args.depth3)
    depth = depth.loc[depth.maximum_depth.eq(3)].copy()
    rank = pd.read_csv(args.rank_oof)
    required = {"query_id", "truth_candidate_id", "truth_formula", "baseline_candidate_id"}
    if not required.issubset(depth) or not required.issubset(rank):
        raise RuntimeError("expert transition schema mismatch")
    merged = depth.merge(
        rank, on=["query_id", "truth_candidate_id", "truth_formula", "baseline_candidate_id"],
        suffixes=("_depth", "_rank"), validate="one_to_one",
    )
    if len(merged) != 117:
        raise RuntimeError(f"expected 117 consumed development queries, found {len(merged)}")
    depth_changed = merged.final_candidate_id_depth != merged.baseline_candidate_id
    rank_changed = merged.final_candidate_id_rank != merged.baseline_candidate_id
    agree = merged.final_candidate_id_depth == merged.final_candidate_id_rank
    conflict = depth_changed & rank_changed & ~agree
    proposed = np.where(
        conflict, merged.baseline_candidate_id,
        np.where(depth_changed, merged.final_candidate_id_depth,
                 np.where(rank_changed, merged.final_candidate_id_rank, merged.baseline_candidate_id)),
    )
    merged["depth3_intervene"] = depth_changed
    merged["rank_consensus_intervene"] = rank_changed
    merged["expert_conflict_abstain"] = conflict
    merged["final_candidate_id"] = proposed
    merged["baseline_correct_router"] = merged.baseline_candidate_id == merged.truth_candidate_id
    merged["final_correct_router"] = merged.final_candidate_id == merged.truth_candidate_id
    merged["corrected_router"] = ~merged.baseline_correct_router & merged.final_correct_router
    merged["introduced_router"] = merged.baseline_correct_router & ~merged.final_correct_router
    merged["delta"] = merged.final_correct_router.astype(float) - merged.baseline_correct_router.astype(float)
    corrected = int(merged.corrected_router.sum())
    introduced = int(merged.introduced_router.sum())
    dev_bootstrap = bootstrap(merged, args.bootstrap_resamples, args.seed)

    # Full-development refit for the frozen deployable rank-consensus expert.
    ledger = add_family_features(pd.read_csv(args.ledger))
    namespace = argparse.Namespace(
        seed=args.seed, temperature=args.temperature, l2=args.l2,
        maximum_family_weight=args.maximum_family_weight,
    )
    inner = inner_oof_predictions(ledger, -1, namespace)
    gate = select_gate(inner)
    weights = fit_family_weights(
        ledger, args.temperature, args.l2, args.maximum_family_weight,
    )
    artifact = {
        "status": "bioaware_v3_consensus_router_artifact_frozen",
        "formal": True,
        "router": {
            "rule": "apply one expert proposal; apply agreeing proposals; abstain to DreaMS on expert conflict",
            "depth3_expert": {
                "maximum_depth": 3,
                "requires_complete_ms2_path": True,
                "requires_unique_strongest_structural_candidate": True,
                "requires_candidate_bottleneck_gt_dreams_top1_bottleneck": True,
                "heldout_rotation_vote": "strict majority",
            },
            "rank_consensus_expert": {
                "family_features": FAMILY_FEATURES,
                "raw_families": RAW_FAMILIES,
                "weights": dict(zip(FAMILY_FEATURES, map(float, weights), strict=True)),
                "gate": {
                    "maximum_spectral_margin": float(gate[0]),
                    "minimum_fusion_advantage": (
                        float(gate[1]) if np.isfinite(gate[1]) else None
                    ),
                    "minimum_support_families": int(gate[2]),
                },
                "temperature": args.temperature,
                "l2": args.l2,
                "maximum_family_weight": args.maximum_family_weight,
            },
        },
        "consumed_development": {
            "queries": int(len(merged)),
            "baseline_recall1": float(merged.baseline_correct_router.mean()),
            "router_recall1": float(merged.final_correct_router.mean()),
            "delta_recall1": float(merged.delta.mean()),
            "corrected": corrected,
            "introduced": introduced,
            "expert_conflicts": int(conflict.sum()),
            "formula_cluster_bootstrap": dev_bootstrap,
        },
        "gates": {
            "corrected_gt_introduced": corrected > introduced,
            "formula_cluster_ci_positive": dev_bootstrap["ci_low"] > 0,
            "risk_weighted_net_positive": corrected - 2 * introduced > 0,
            # The source contract defines development as mechanism/implementation
            # only and reserves significance for the external test.  Unlocking
            # RPLC therefore uses the preregistered safety criteria, while the
            # low-power development CI remains reported and must not be called
            # a confirmatory result.
            "pass_to_frozen_rplc_internal_validation": (
                corrected > introduced and corrected - 2 * introduced > 0
                and int(conflict.sum()) == 0
            ),
        },
        "contracts": {
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "truth_not_used_by_router": True,
            "development_is_consumed": True,
            "rplc_or_external_outcomes_opened": False,
            "evaluation_must_load_this_artifact_without_refit": True,
        },
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "depth3_oof_sha256": sha256(args.depth3),
            "rank_oof_sha256": sha256(args.rank_oof),
        },
        "claim_limit": "Frozen consumed-development recipe; RPLC internal and 16-panel external results are required for SOTA.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions = args.output_dir / "development_router_transitions.csv.gz"
    merged.to_csv(transitions, index=False)
    artifact["provenance"]["transitions_sha256"] = sha256(transitions)
    (args.output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
