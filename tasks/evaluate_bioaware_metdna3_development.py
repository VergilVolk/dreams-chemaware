#!/usr/bin/env python
"""Evaluate fixed BioAware v2 evidence on MetDNA3 HILIC development rotations.

This is deliberately a fixed-policy development audit.  It compares archived
raw path accumulation with dependency-corrected reaction hyperedges, using only
Level-1 seed identities assigned by the frozen 30/70 identity rotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.bioaware import (  # noqa: E402
    BioAwareConfig, build_one_hop_evidence, fuse_candidates, top1_transition_table,
)
from annotation.bioaware_context import extract_reaction_context_features  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def bootstrap_formula_delta(
    rows: pd.DataFrame, *, seed: int, resamples: int
) -> dict:
    grouped = {
        formula: group["delta"].to_numpy(float)
        for formula, group in rows.groupby("truth_formula", sort=False)
    }
    formulas = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        values[index] = np.concatenate([grouped[value] for value in sampled]).mean()
    return {
        "mean": float(rows["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "formula_clusters": int(len(formulas)),
        "resamples": int(resamples),
    }


def evaluate_policy(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    support_column: str,
    config: BioAwareConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    support = features[
        ["query_id", "candidate_id", support_column, "raw_path_count"]
    ].rename(columns={support_column: "network_support", "raw_path_count": "network_path_count"})
    attached = candidates.merge(support, on=["query_id", "candidate_id"], validate="one_to_one")
    scored, decisions = fuse_candidates(attached, config)
    transitions, result = top1_transition_table(scored, truth_col="truth_candidate_id")
    result["intervention_rate"] = float(decisions["bioaware_applied"].mean())
    result["queries_with_network_evidence"] = int(decisions["network_available"].sum())
    result["evidence_conflicts_abstained"] = int(
        decisions["evidence_state"].isin(
            ["spectral_strong_network_conflict", "network_ambiguous"]
        ).sum()
    )
    return transitions, decisions, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"),
    )
    parser.add_argument(
        "--dreams-report", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1/report.json"),
    )
    parser.add_argument(
        "--splits", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"),
    )
    parser.add_argument(
        "--participants", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_eval_v1"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    for path in (args.scores, args.dreams_report, args.splits, args.participants):
        if not path.exists():
            raise FileNotFoundError(path)
    dreams_report = json.loads(args.dreams_report.read_text(encoding="utf-8"))
    if not dreams_report.get("formal") or dreams_report["contracts"].get("reaction_network_used") is not False:
        raise RuntimeError("official DreaMS baseline contract failed")
    candidates = pd.read_csv(args.scores)
    split = pd.read_csv(args.splits)
    participants = pd.read_csv(args.participants)
    config = BioAwareConfig()
    graph_identities = set(participants["compound_id"].astype(str))
    all_transitions: dict[str, list[pd.DataFrame]] = {"raw": [], "dependency_corrected": []}
    fold_reports: list[dict] = []
    explanation_frames: list[pd.DataFrame] = []

    for fold in range(10):
        seed_identities = set(split[(split["fold"] == fold) & (split["role"] == "seed")]["ik14"])
        heldout_identities = set(split[(split["fold"] == fold) & (split["role"] == "heldout")]["ik14"])
        fold_candidates = candidates[candidates["truth_candidate_id"].isin(heldout_identities)].copy()
        if not len(fold_candidates):
            raise RuntimeError(f"fold {fold} has no held-out queries")
        if set(fold_candidates["truth_candidate_id"]) & seed_identities:
            raise RuntimeError(f"fold {fold} identity leakage")
        eligible_seeds = sorted(seed_identities & graph_identities)
        seeds = pd.DataFrame({
            "seed_query_id": [f"fold{fold}:seed:{identity}" for identity in eligible_seeds],
            "seed_compound_id": eligible_seeds,
            "seed_score": 1.0,
        })
        paths = build_one_hop_evidence(participants, seeds, config)
        features, details = extract_reaction_context_features(
            fold_candidates, paths, participants, seeds,
            exclude_truth_identity=False,
        )
        if len(details):
            details.insert(0, "fold", fold)
            explanation_frames.append(details)
        fold_result = {"fold": fold, "queries": int(fold_candidates["query_id"].nunique()),
                       "seed_identities": len(eligible_seeds), "evidence_paths": int(len(details))}
        for name, column in [
            ("raw", "raw_network_support"),
            ("dependency_corrected", "dependency_corrected_network_support"),
        ]:
            transitions, _, result = evaluate_policy(fold_candidates, features, column, config)
            transitions["fold"] = fold
            transitions = transitions.merge(
                fold_candidates[["query_id", "truth_formula"]].drop_duplicates(),
                on="query_id", validate="one_to_one",
            )
            transitions["delta"] = (
                transitions["final_correct"].astype(int)
                - transitions["baseline_correct"].astype(int)
            )
            all_transitions[name].append(transitions)
            fold_result[name] = result
        fold_reports.append(fold_result)
        print(
            f"[fold {fold}] queries={fold_result['queries']} seeds={len(eligible_seeds)} "
            f"dep C/I={fold_result['dependency_corrected']['corrected']}/"
            f"{fold_result['dependency_corrected']['introduced']}", flush=True,
        )

    combined_results: dict[str, dict] = {}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: development evaluation exists: {output}")
    for position, (name, frames) in enumerate(all_transitions.items()):
        frame = pd.concat(frames, ignore_index=True)
        corrected = int(frame["corrected"].sum())
        introduced = int(frame["introduced"].sum())
        result = {
            "query_rotation_instances": int(len(frame)),
            "unique_queries": int(frame["query_id"].nunique()),
            "baseline_recall1": float(frame["baseline_correct"].mean()),
            "bioaware_recall1": float(frame["final_correct"].mean()),
            "delta_recall1": float(frame["delta"].mean()),
            "corrected": corrected, "introduced": introduced,
            "mcnemar_exact_p": float(
                binomtest(min(corrected, introduced), corrected + introduced, 0.5).pvalue
            ) if corrected + introduced else 1.0,
            "formula_cluster_bootstrap": bootstrap_formula_delta(
                frame, seed=args.seed + position, resamples=args.bootstrap_resamples
            ),
        }
        combined_results[name] = result
        frame.to_csv(output / f"{name}_transitions.csv.gz", index=False, compression="gzip")
    if explanation_frames:
        pd.concat(explanation_frames, ignore_index=True).to_csv(
            output / "evidence_paths.csv.gz", index=False, compression="gzip"
        )
    primary = combined_results["dependency_corrected"]
    report = {
        "status": "bioaware_metdna3_hilic_development_complete", "formal": True,
        "protocol": "fixed BioAware v1 gate; 10 frozen 30/70 identity rotations; dependency-corrected Rhea hyperedges",
        "official_dreams_baseline": dreams_report["official_dreams_recall1"],
        "folds": fold_reports, "combined": combined_results,
        "configuration": config.to_dict(),
        "gates": {
            "dependency_corrected_gt_raw_net": (
                primary["corrected"] - primary["introduced"]
                > combined_results["raw"]["corrected"] - combined_results["raw"]["introduced"]
            ),
            "dependency_corrected_corrected_gt_introduced": primary["corrected"] > primary["introduced"],
            "dependency_corrected_formula_ci_positive": primary["formula_cluster_bootstrap"]["ci_low"] > 0,
        },
        "contracts": {
            "phenotype_blind": True, "heldout_truth_identity_absent_from_seeds": True,
            "P2b": "forbidden", "internal_validation_or_external_test_opened": False,
            "policy_or_threshold_fitted_here": False,
        },
        "provenance": {
            "scores_sha256": sha256(args.scores), "dreams_report_sha256": sha256(args.dreams_report),
            "splits_sha256": sha256(args.splits), "participants_sha256": sha256(args.participants),
        },
        "claim_limit": (
            "Consumed HILIC development evidence only. A positive result advances to locked RP internal validation; "
            "it is not an external SOTA, MSI identity, pathway, flux, or embedding claim."
        ),
    }
    report["gates"]["pass_to_locked_rp"] = all(report["gates"].values())
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

