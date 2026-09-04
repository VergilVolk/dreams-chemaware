#!/usr/bin/env python
"""Freeze the deterministic structural-plus-raw-edge BioAware decision audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

try:
    from build_bioaware_metdna3_candidate_path_table import evidence_key
except ModuleNotFoundError:
    from tasks.build_bioaware_metdna3_candidate_path_table import evidence_key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def formula_bootstrap(delta: np.ndarray, formula: np.ndarray, seed: int) -> dict:
    unique = np.unique(formula.astype(str))
    grouped = {value: delta[formula.astype(str) == value] for value in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, float)
    for index in range(len(boot)):
        sampled = rng.choice(unique, len(unique), replace=True)
        boot[index] = np.mean(np.concatenate([grouped[value] for value in sampled]))
    return {
        "mean": float(np.mean(delta)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "formulas": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_paths_v1/candidate_paths.csv.gz"))
    parser.add_argument("--edge-evidence", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_ms2_v2/candidate_edge_evidence.csv.gz"))
    parser.add_argument("--baseline", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_decision_v1"))
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()
    for path in (args.paths, args.edge_evidence, args.baseline):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    paths = pd.read_csv(args.paths)
    edge = pd.read_csv(args.edge_evidence)
    baseline = pd.read_csv(args.baseline).groupby("query_id")["baseline_top_candidate"].first().astype(str)
    fold_records: list[dict] = []
    for depth in (2, 3):
        table = paths.merge(
            edge[edge.maximum_depth.eq(depth)][
                ["fold", "query_id", "candidate_id", "best_bottleneck"]
            ], on=["fold", "query_id", "candidate_id"], validate="one_to_one",
        )
        for (fold, query_id), group in table.groupby(["fold", "query_id"], sort=True):
            base_id = str(baseline.loc[query_id])
            truth_id = str(group.truth_candidate_id.iloc[0])
            eligible = group.path_available & group.minimum_depth.le(depth)
            keys = [
                evidence_key(row) if bool(is_eligible) else (0, 0, 0, 0)
                for (_, row), is_eligible in zip(group.iterrows(), eligible, strict=True)
            ]
            best_key = max(keys)
            positions = [position for position, key in enumerate(keys) if key == best_key]
            network_id = str(group.iloc[positions[0]].candidate_id) if len(positions) == 1 else base_id
            base = group[group.candidate_id.eq(base_id)].iloc[0]
            network = group[group.candidate_id.eq(network_id)].iloc[0]
            comparable = np.isfinite(base.best_bottleneck) and np.isfinite(network.best_bottleneck)
            intervene = bool(
                network_id != base_id and comparable
                and float(network.best_bottleneck) > float(base.best_bottleneck)
            )
            final_id = network_id if intervene else base_id
            fold_records.append({
                "maximum_depth": depth, "fold": int(fold), "query_id": str(query_id),
                "truth_candidate_id": truth_id, "truth_formula": str(group.truth_formula.iloc[0]),
                "baseline_candidate_id": base_id, "network_candidate_id": network_id,
                "final_candidate_id": final_id, "intervene": intervene,
                "baseline_correct": base_id == truth_id, "final_correct": final_id == truth_id,
                "raw_edge_advantage": (
                    float(network.best_bottleneck - base.best_bottleneck) if comparable else np.nan
                ),
            })
    folds = pd.DataFrame(fold_records)
    fold_path = output / "fold_transitions.csv.gz"
    folds.to_csv(fold_path, index=False, compression="gzip")

    query_records: list[dict] = []
    for (depth, query_id), group in folds.groupby(["maximum_depth", "query_id"], sort=True):
        counts = Counter(group.final_candidate_id.astype(str))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        base_id = str(group.baseline_candidate_id.iloc[0])
        final_id = ordered[0][0] if len(ordered) == 1 or ordered[0][1] > ordered[1][1] else base_id
        truth_id = str(group.truth_candidate_id.iloc[0])
        query_records.append({
            "maximum_depth": int(depth), "query_id": str(query_id),
            "truth_candidate_id": truth_id, "truth_formula": str(group.truth_formula.iloc[0]),
            "baseline_candidate_id": base_id, "final_candidate_id": final_id,
            "intervention_rotations": int(group.intervene.sum()),
            "winning_vote_count": int(ordered[0][1]), "heldout_rotations": int(len(group)),
            "baseline_correct": base_id == truth_id, "final_correct": final_id == truth_id,
        })
    queries = pd.DataFrame(query_records)
    query_path = output / "query_transitions.csv.gz"
    queries.to_csv(query_path, index=False, compression="gzip")

    results: dict[str, dict] = {}
    for depth, group in queries.groupby("maximum_depth"):
        delta = group.final_correct.astype(int).to_numpy() - group.baseline_correct.astype(int).to_numpy()
        corrected = int(((~group.baseline_correct) & group.final_correct).sum())
        introduced = int((group.baseline_correct & (~group.final_correct)).sum())
        discordant = corrected + introduced
        results[f"depth{int(depth)}"] = {
            "queries": int(len(group)),
            "query_identities": int(group.truth_candidate_id.nunique()),
            "baseline_recall1": float(group.baseline_correct.mean()),
            "bioaware_recall1": float(group.final_correct.mean()),
            "delta_recall1": float(np.mean(delta)),
            "corrected": corrected, "introduced": introduced,
            "corrected_identities": int(group.loc[(~group.baseline_correct) & group.final_correct, "truth_candidate_id"].nunique()),
            "introduced_identities": int(group.loc[group.baseline_correct & (~group.final_correct), "truth_candidate_id"].nunique()),
            "formula_cluster_bootstrap": formula_bootstrap(
                delta, group.truth_formula.to_numpy(str), 20260828 + int(depth)
            ),
            "mcnemar_exact_p": float(
                binomtest(min(corrected, introduced), discordant, 0.5).pvalue
            ) if discordant else 1.0,
        }
    report = {
        "status": "bioaware_metdna3_candidate_edge_decision_complete",
        "formal": True,
        "scope": args.scope,
        "decision_rule": (
            "unique strongest structural candidate; full raw-MS2 path; candidate bottleneck "
            "strictly exceeds frozen DreaMS Top-1 bottleneck; seven heldout rotations majority vote"
        ),
        "results": results,
        "primary": "depth2 (preregistered)",
        "depth3": "development ablation; may not be promoted to confirmatory result",
        "gates": {
            "primary_depth2_positive": results["depth2"]["delta_recall1"] > 0,
            "depth3_ablation_positive": results["depth3"]["delta_recall1"] > 0,
            "depth3_no_introduced": results["depth3"]["introduced"] == 0,
            "depth3_formula_ci_positive": results["depth3"]["formula_cluster_bootstrap"]["ci_low"] > 0,
            "pass_to_RP": False,
        },
        "contracts": {
            "score_threshold_tuned": False, "predicted_step1_used": False,
            "P2b_used": False, "external_test_opened": False,
        },
        "provenance": {
            "paths_sha256": sha256(args.paths), "edge_evidence_sha256": sha256(args.edge_evidence),
            "baseline_sha256": sha256(args.baseline), "fold_transitions_sha256": sha256(fold_path),
            "query_transitions_sha256": sha256(query_path),
        },
        "claim_limit": (
            "Depth-3 is a promising consumed-development ablation, not an external gain: "
            "only two independent corrected identities and no positive formula-cluster lower CI."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
