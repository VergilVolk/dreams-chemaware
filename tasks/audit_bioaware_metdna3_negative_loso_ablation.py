#!/usr/bin/env python
"""Identity-purged ablation of the external negative-ion BioAware ranker.

The primary comparison is full BioAware versus a spectral-only pairwise model
under the *same* raw-edge safety gate.  This prevents supervised recalibration
of DreaMS cosine from being misreported as a network-prior improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Script execution adds tasks/ to sys.path; package import does not.
    from develop_bioaware_metdna3_negative_loso_ranker import (
        FEATURES,
        evaluate_fold,
        formula_bootstrap,
        summarize,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by pytest import mode
    from tasks.develop_bioaware_metdna3_negative_loso_ranker import (
        FEATURES,
        evaluate_fold,
        formula_bootstrap,
        summarize,
    )


KNOWN = [
    "known_mass_candidate_fraction",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
]
RAW_EDGE = [
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
]
ABLATIONS: dict[str, dict] = {
    "spectral_only_no_edge_gate": {
        "features": ["spectral_score"], "require_raw_step0_edge": False,
    },
    "spectral_only_same_edge_gate": {
        "features": ["spectral_score"], "require_raw_step0_edge": True,
    },
    "mass_membership_only_no_edge_gate": {
        "features": ["known_mass_candidate_fraction"], "require_raw_step0_edge": False,
    },
    "mass_membership_only_same_edge_gate": {
        "features": ["known_mass_candidate_fraction"], "require_raw_step0_edge": True,
    },
    "known_topology_only_same_edge_gate": {
        "features": KNOWN, "require_raw_step0_edge": True,
    },
    "known_topology_without_mass_same_edge_gate": {
        "features": KNOWN[1:], "require_raw_step0_edge": True,
    },
    "raw_step0_only_same_edge_gate": {
        "features": ["edge0_complete_fraction", "edge0_bottleneck_mean"],
        "require_raw_step0_edge": True,
    },
    "network_only_same_edge_gate": {
        "features": KNOWN + RAW_EDGE, "require_raw_step0_edge": True,
    },
    "spectral_plus_known_topology": {
        "features": ["spectral_score"] + KNOWN, "require_raw_step0_edge": True,
    },
    "spectral_plus_raw_edges": {
        "features": ["spectral_score"] + RAW_EDGE, "require_raw_step0_edge": True,
    },
    "full_no_edge_gate": {
        "features": FEATURES, "require_raw_step0_edge": False,
    },
    "full_bioaware": {
        "features": FEATURES, "require_raw_step0_edge": True,
    },
}


def paired_formula_bootstrap(
    left: pd.DataFrame, right: pd.DataFrame, repeats: int, seed: int
) -> dict:
    merged = left[["query_id", "truth_formula", "final_correct"]].merge(
        right[["query_id", "final_correct"]], on="query_id", suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged["paired_delta"] = (
        merged["final_correct_left"].astype(int) - merged["final_correct_right"].astype(int)
    )
    grouped = merged.groupby("truth_formula", sort=False)["paired_delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, float)
    for index in range(repeats):
        draw = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(merged["paired_delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "left_better": int((merged["paired_delta"] > 0).sum()),
        "right_better": int((merged["paired_delta"] < 0).sum()),
        "equal": int((merged["paired_delta"] == 0).sum()),
        "formulas": int(len(grouped)),
        "resamples": int(repeats),
    }


def run_ablation(candidates: pd.DataFrame, name: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    outputs: list[pd.DataFrame] = []
    folds: list[dict] = []
    for unit in sorted(candidates["unit_id"].unique()):
        test = candidates[candidates["unit_id"].eq(unit)].copy()
        test_truth = set(test["truth_candidate_id"].astype(str))
        train = candidates[
            (~candidates["unit_id"].eq(unit))
            & (~candidates["truth_candidate_id"].astype(str).isin(test_truth))
        ].copy()
        overlap = set(train["truth_candidate_id"].astype(str)) & test_truth
        if overlap:
            raise RuntimeError(f"{name}/{unit}: identity purge failed")
        result, fold = evaluate_fold(
            train,
            test,
            str(unit),
            features=spec["features"],
            require_raw_step0_edge=spec["require_raw_step0_edge"],
        )
        fold["training_truth_identity_overlap"] = 0
        fold["result"] = summarize(result)
        outputs.append(result)
        folds.append(fold)
    transitions = pd.concat(outputs, ignore_index=True).sort_values("query_id").reset_index(drop=True)
    expected_queries = int(candidates["query_id"].nunique())
    if transitions["query_id"].duplicated().any() or len(transitions) != expected_queries:
        raise RuntimeError(f"{name}: expected {expected_queries} unique external queries")
    report = {
        "features": spec["features"],
        "require_raw_step0_edge": bool(spec["require_raw_step0_edge"]),
        "pooled": summarize(transitions),
        "formula_cluster_bootstrap_vs_dreams": formula_bootstrap(
            transitions, 5000, 20260901
        ),
        "folds": folds,
    }
    return transitions, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--source-report", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ablation_v2"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    if not args.candidate_features.exists() or not args.source_report.exists():
        raise FileNotFoundError("identity-purged v3 artifacts are required")
    candidates = pd.read_csv(args.candidate_features)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source.get("protocol", "").find("truth IK14 purged") < 0:
        raise RuntimeError("source is not the identity-purged protocol")
    results: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    args.output_dir.mkdir(parents=True)
    for name, spec in ABLATIONS.items():
        transitions, report = run_ablation(candidates, name, spec)
        results[name] = transitions
        reports[name] = report
        transitions.to_csv(args.output_dir / f"{name}__transitions.csv.gz", index=False, compression="gzip")
        print(f"[ablation] {name}: {report['pooled']}", flush=True)
    full = results["full_bioaware"]
    comparisons = {}
    for name, transitions in results.items():
        if name == "full_bioaware":
            continue
        comparisons[f"full_bioaware_minus_{name}"] = paired_formula_bootstrap(
            full, transitions, args.bootstrap_resamples, 20260901
        )
    source_pooled = source["pooled"]
    replay = reports["full_bioaware"]["pooled"]
    for key in ("queries", "corrected", "introduced", "interventions"):
        if replay[key] != source_pooled[key]:
            raise RuntimeError(f"full-model replay mismatch for {key}: {replay[key]} != {source_pooled[key]}")
    report = {
        "status": "bioaware_metdna3_negative_loso_ablation_complete",
        "formal": True,
        "protocol": "identity-purged eight-unit LOSO; fixed thresholds; exact candidate graph; query-paired ablation",
        "models": reports,
        "paired_formula_cluster_comparisons": comparisons,
        "primary_network_increment": comparisons["full_bioaware_minus_spectral_only_same_edge_gate"],
        "contracts": {
            "full_v3_reproduced": True,
            "test_truth_identity_overlap": 0,
            "all_thresholds_fixed": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "shared_embedding_changed": False,
        },
        "decision_rule": "Call this BioAware increment only if full beats spectral-only under the identical raw-edge gate with paired formula-cluster CI lower bound > 0.",
        "claim_limit": "Development ablation on opened external units; a new independent cohort remains required.",
    }
    report["network_increment_pass"] = report["primary_network_increment"]["ci_low"] > 0
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
