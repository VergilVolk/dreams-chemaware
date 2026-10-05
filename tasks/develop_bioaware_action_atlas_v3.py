#!/usr/bin/env python
"""Develop and falsify BioAware candidate actions before any embedding fit.

This opened-development experiment has three narrow purposes:

1. exactly replay the strongest same-formula uncertainty action;
2. test whether its gain survives removing the network-degree feature;
3. recover path-reliability fields that older aggregation discarded and test
   fixed, preregistered low-capacity action recipes.

It also runs a degree-fixed, path-availability-conditioned joint permutation
of the remaining network block.  This is an action-discovery audit, never an
external confirmation or a shared-embedding result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    evaluate_fold,
    formula_bootstrap,
    summarize,
)
from develop_bioaware_same_formula_network_expert_v3 import (  # noqa: E402
    FEATURES as CURRENT_FEATURES,
)


RICH_EDGE_COLUMNS = [
    "path_available_fraction",
    "complete_path_fraction",
    "completion_ratio_mean",
    "log_identity_paths_mean",
    "log_complete_paths_mean",
    "log_node_combinations_mean",
    "multi_complete_fraction",
    "best_bottleneck_mean",
    "median_bottleneck_mean",
    "bottleneck_spread_mean",
    "truncation_fraction",
]

RICH_FEATURES = [
    *(f"edge0_{name}" for name in RICH_EDGE_COLUMNS),
    *(f"edge1_{name}" for name in RICH_EDGE_COLUMNS),
    "known_seed_per_degree",
    "known_path_per_degree",
    "edge0_reliability",
    "edge1_reliability",
    "edge_reliability_increment",
    "edge0_multipath_reliability",
    "edge1_multipath_reliability",
]

CURRENT_NO_DEGREE = [name for name in CURRENT_FEATURES if name != "known_log_degree"]

# Fixed before outcomes are recomputed.  This is a mechanism matrix, not a
# hyperparameter sweep.  Every recipe is retained in the report.
RECIPES: dict[str, list[str]] = {
    "spectral_only_control": ["spectral_score"],
    "spectral_plus_degree_control": ["spectral_score", "known_log_degree"],
    "current_v4_replay": list(CURRENT_FEATURES),
    "current_without_degree": CURRENT_NO_DEGREE,
    "path_reliability_without_degree": [
        "spectral_score",
        "known_path_fraction",
        "known_inverse_depth_mean",
        "known_seed_per_degree",
        "known_path_per_degree",
        "edge0_reliability",
        "edge1_reliability",
        "edge_reliability_increment",
        "edge0_multipath_reliability",
        "edge1_multipath_reliability",
        "edge0_bottleneck_spread_mean",
        "edge1_bottleneck_spread_mean",
    ],
    "current_plus_path_reliability_without_degree": [
        *CURRENT_NO_DEGREE,
        *RICH_FEATURES,
    ],
    "current_plus_path_reliability": [
        *CURRENT_FEATURES,
        *RICH_FEATURES,
    ],
}

NETWORK_BLOCK_FOR_NULL = [
    name
    for name in dict.fromkeys([*CURRENT_FEATURES, *RICH_FEATURES])
    if name not in {"spectral_score", "known_log_degree"}
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_rich_edge(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "query_id", "candidate_id", "maximum_depth", "identity_paths",
        "complete_ms2_paths", "node_combinations_evaluated", "path_truncated",
        "best_bottleneck", "median_bottleneck",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing edge columns {sorted(missing)}")
    frame = frame.loc[frame["maximum_depth"].eq(2)].copy()
    if frame.empty:
        raise RuntimeError(f"{path}: no maximum_depth=2 rows")
    identity_paths = pd.to_numeric(frame["identity_paths"], errors="raise").to_numpy(float)
    complete_paths = pd.to_numeric(
        frame["complete_ms2_paths"], errors="raise"
    ).to_numpy(float)
    combinations = pd.to_numeric(
        frame["node_combinations_evaluated"], errors="raise"
    ).to_numpy(float)
    best = pd.to_numeric(frame["best_bottleneck"], errors="coerce").fillna(0).to_numpy(float)
    median = pd.to_numeric(
        frame["median_bottleneck"], errors="coerce"
    ).fillna(0).to_numpy(float)
    if np.any(identity_paths < 0) or np.any(complete_paths < 0):
        raise RuntimeError(f"{path}: negative path count")
    if np.any(complete_paths > identity_paths):
        raise RuntimeError(f"{path}: complete paths exceed identity paths")
    frame["path_available"] = (identity_paths > 0).astype(float)
    frame["complete_available"] = (complete_paths > 0).astype(float)
    frame["completion_ratio"] = np.divide(
        complete_paths,
        identity_paths,
        out=np.zeros_like(complete_paths),
        where=identity_paths > 0,
    )
    frame["log_identity_paths"] = np.log1p(identity_paths)
    frame["log_complete_paths"] = np.log1p(complete_paths)
    frame["log_node_combinations"] = np.log1p(combinations)
    frame["multi_complete"] = (complete_paths >= 2).astype(float)
    frame["best_zero"] = best
    frame["median_zero"] = median
    frame["bottleneck_spread"] = np.maximum(0.0, best - median)
    frame["truncated"] = frame["path_truncated"].astype(bool).astype(float)
    aggregated = frame.groupby(["query_id", "candidate_id"], sort=False).agg(
        path_available_fraction=("path_available", "mean"),
        complete_path_fraction=("complete_available", "mean"),
        completion_ratio_mean=("completion_ratio", "mean"),
        log_identity_paths_mean=("log_identity_paths", "mean"),
        log_complete_paths_mean=("log_complete_paths", "mean"),
        log_node_combinations_mean=("log_node_combinations", "mean"),
        multi_complete_fraction=("multi_complete", "mean"),
        best_bottleneck_mean=("best_zero", "mean"),
        median_bottleneck_mean=("median_zero", "mean"),
        bottleneck_spread_mean=("bottleneck_spread", "mean"),
        truncation_fraction=("truncated", "mean"),
    ).reset_index()
    return aggregated.rename(
        columns={name: f"{prefix}_{name}" for name in RICH_EDGE_COLUMNS}
    )


def load_rich_edges(root: Path, prefix: str) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = sorted(root.glob("*/candidate_edge_evidence.csv.gz"))
    if len(paths) != 8:
        raise RuntimeError(f"{root}: expected eight unit edge tables, found {len(paths)}")
    tables = [aggregate_rich_edge(path, prefix) for path in paths]
    merged = pd.concat(tables, ignore_index=True)
    if merged.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError(f"{root}: duplicate candidate edge rows after unit aggregation")
    return merged, {str(path.relative_to(root)): sha256(path) for path in paths}


def enrich_candidates(
    candidates: pd.DataFrame, edge0_root: Path, edge1_root: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    required = {
        "query_id", "candidate_id", "unit_id", "truth_candidate_id",
        "truth_formula", "baseline_correct", "top_candidate_id", "is_positive",
        *CURRENT_FEATURES,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate table missing columns {sorted(missing)}")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate identities are not unique within query")
    edge0, provenance0 = load_rich_edges(edge0_root, "edge0")
    edge1, provenance1 = load_rich_edges(edge1_root, "edge1")
    output = candidates.merge(
        edge0, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    ).merge(
        edge1, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    )
    for column in RICH_FEATURES:
        if column not in output:
            output[column] = 0.0
    rich_edge_columns = [
        name for name in output if name.startswith("edge0_") or name.startswith("edge1_")
    ]
    for column in rich_edge_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    # Exact replay checks: the historical two-column aggregation must be a
    # projection of this richer table, not a newly defined protocol.
    for prefix in ("edge0", "edge1"):
        old_complete = output[f"{prefix}_complete_fraction"].to_numpy(float)
        rich_complete = output[f"{prefix}_complete_path_fraction"].to_numpy(float)
        old_best = output[f"{prefix}_bottleneck_mean"].to_numpy(float)
        rich_best = output[f"{prefix}_best_bottleneck_mean"].to_numpy(float)
        if not np.allclose(old_complete, rich_complete, rtol=0, atol=1e-12):
            raise RuntimeError(f"{prefix}: rich aggregation changed complete_fraction")
        if not np.allclose(old_best, rich_best, rtol=0, atol=1e-12):
            raise RuntimeError(f"{prefix}: rich aggregation changed bottleneck_mean")

    degree_scale = 1.0 + output["known_log_degree"].clip(lower=0.0)
    output["known_seed_per_degree"] = (
        output["known_log_seed_support_mean"].clip(lower=0.0) / degree_scale
    )
    output["known_path_per_degree"] = (
        output["known_path_fraction"].clip(lower=0.0) / degree_scale
    )
    for prefix in ("edge0", "edge1"):
        output[f"{prefix}_reliability"] = (
            output[f"{prefix}_completion_ratio_mean"]
            * output[f"{prefix}_median_bottleneck_mean"]
        )
        output[f"{prefix}_multipath_reliability"] = (
            output[f"{prefix}_log_complete_paths_mean"]
            * output[f"{prefix}_median_bottleneck_mean"]
            / degree_scale
        )
    output["edge_reliability_increment"] = (
        output["edge1_reliability"] - output["edge0_reliability"]
    )
    for name, features in RECIPES.items():
        missing_recipe = set(features) - set(output.columns)
        if missing_recipe:
            raise RuntimeError(f"recipe {name} missing features {sorted(missing_recipe)}")
        values = output[features].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"recipe {name} contains non-finite values")
    return output, {"edge0": provenance0, "edge1": provenance1}


def run_loso(
    candidates: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    outputs: list[pd.DataFrame] = []
    reports: list[dict] = []
    for unit in sorted(candidates["unit_id"].astype(str).unique()):
        test = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        identities = set(test["truth_candidate_id"].astype(str))
        formulas = set(test["truth_formula"].astype(str))
        train = candidates.loc[
            ~candidates["unit_id"].astype(str).eq(unit)
            & ~candidates["truth_candidate_id"].astype(str).isin(identities)
            & ~candidates["truth_formula"].astype(str).isin(formulas)
        ].copy()
        if train["query_id"].nunique() < 100:
            raise RuntimeError(f"{unit}: fewer than 100 formula-purged training queries")
        if set(train["truth_candidate_id"].astype(str)) & identities:
            raise RuntimeError(f"{unit}: truth identity leakage")
        if set(train["truth_formula"].astype(str)) & formulas:
            raise RuntimeError(f"{unit}: truth formula leakage")
        transitions, report = evaluate_fold(
            train,
            test,
            unit,
            features=features,
            require_raw_step0_edge=False,
            maximum_baseline_margin=0.05,
            minimum_proposal_probability=0.5,
        )
        report.update(
            {
                "test_truth_identities": len(identities),
                "test_truth_formulas": len(formulas),
                "training_truth_identity_overlap": 0,
                "training_truth_formula_overlap": 0,
                "result": summarize(transitions),
            }
        )
        outputs.append(transitions)
        reports.append(report)
    combined = pd.concat(outputs, ignore_index=True)
    if len(combined) != candidates["query_id"].nunique():
        raise RuntimeError("LOSO output changed query coverage")
    return combined, reports


def independent_counts(frame: pd.DataFrame) -> dict:
    corrected = frame.loc[frame["corrected"]]
    introduced = frame.loc[frame["introduced"]]
    intervened = frame.loc[frame["intervene"]]
    return {
        "intervention_identities": int(intervened["truth_candidate_id"].nunique()),
        "intervention_formulas": int(intervened["truth_formula"].nunique()),
        "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
        "corrected_formulas": int(corrected["truth_formula"].nunique()),
        "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
        "introduced_formulas": int(introduced["truth_formula"].nunique()),
    }


def paired_formula_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    columns = ["query_id", "truth_formula", "delta"]
    paired = left[columns].merge(
        right[columns], on=["query_id", "truth_formula"], suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    paired["difference"] = paired["delta_left"] - paired["delta_right"]
    grouped = paired.groupby("truth_formula", sort=False)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(paired["difference"].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "formulas": int(len(grouped)),
        "resamples": repeats,
    }


def degree_conditioned_permutation(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    output = frame.copy()
    degree_rank = output["known_log_degree"].rank(method="first")
    output["__degree_bin"] = pd.qcut(degree_rank, q=5, labels=False)
    output["__known_path"] = (output["known_path_fraction"] > 0).astype(int)
    output["__edge0"] = (output["edge0_complete_fraction"] > 0).astype(int)
    output["__edge1"] = (output["edge1_complete_fraction"] > 0).astype(int)
    strata = ["unit_id", "__degree_bin", "__known_path", "__edge0", "__edge1"]
    moved = 0
    eligible = 0
    for _, index in output.groupby(strata, sort=False).groups.items():
        locations = np.asarray(list(index), dtype=int)
        if len(locations) < 2:
            continue
        eligible += len(locations)
        permutation = rng.permutation(len(locations))
        if np.all(permutation == np.arange(len(locations))):
            permutation = np.roll(permutation, 1)
        source = output.loc[locations, NETWORK_BLOCK_FOR_NULL].to_numpy(copy=True)
        output.loc[locations, NETWORK_BLOCK_FOR_NULL] = source[permutation]
        moved += int(np.sum(permutation != np.arange(len(locations))))
    output = output.drop(columns=strata[1:])
    return output, {
        "eligible_rows": eligible,
        "moved_rows": moved,
        "moved_fraction": moved / len(output),
        "fixed_features": ["spectral_score", "known_log_degree"],
        "strata": strata,
    }


def replay_check(observed: pd.DataFrame, frozen: pd.DataFrame) -> dict:
    columns = [
        "query_id", "baseline_candidate_id", "proposed_candidate_id",
        "final_candidate_id", "baseline_correct", "final_correct", "corrected",
        "introduced", "intervene",
    ]
    merged = observed[columns].merge(
        frozen[columns], on="query_id", suffixes=("_new", "_frozen"),
        validate="one_to_one",
    )
    mismatches: dict[str, int] = {}
    for column in columns[1:]:
        mismatches[column] = int(
            (~merged[f"{column}_new"].eq(merged[f"{column}_frozen"])).sum()
        )
    if any(mismatches.values()):
        raise RuntimeError(f"current-v4 replay mismatch: {mismatches}")
    return {"queries": len(merged), "mismatches": mismatches, "pass": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path(
            "data/validation/bioaware_same_formula_network_expert_v3_development/"
            "candidate_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--frozen-current-transitions", type=Path,
        default=Path(
            "data/validation/bioaware_same_formula_uncertainty_expert_v4_development/"
            "query_transitions.csv.gz"
        ),
    )
    parser.add_argument(
        "--edge0-root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_edge_step0_v1"),
    )
    parser.add_argument(
        "--edge1-root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_edge_step1_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_action_atlas_v3_20260906"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is non-empty: {args.output_dir}")
    for path in (args.candidate_features, args.frozen_current_transitions):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates, edge_provenance = enrich_candidates(
        pd.read_csv(args.candidate_features), args.edge0_root, args.edge1_root
    )
    if len(candidates) != 1800 or candidates["query_id"].nunique() != 482:
        raise RuntimeError("same-formula candidate protocol changed")

    transitions: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for offset, (name, features) in enumerate(RECIPES.items()):
        result, folds = run_loso(candidates, features)
        summary = summarize(result)
        summary.update(independent_counts(result))
        bootstrap = formula_bootstrap(
            result, args.bootstrap_resamples, args.seed + offset
        )
        reports[name] = {
            "features": features,
            "pooled": summary,
            "formula_cluster_bootstrap": bootstrap,
            "all_units_nonnegative": all(
                fold["result"]["delta_recall1"] >= 0 for fold in folds
            ),
            "folds": folds,
        }
        transitions[name] = result
        print(
            f"[action {name}] delta={summary['delta_recall1']:+.4f} "
            f"C/I={summary['corrected']}/{summary['introduced']} "
            f"ids={summary['corrected_identities']}",
            flush=True,
        )

    frozen = pd.read_csv(args.frozen_current_transitions)
    replay = replay_check(transitions["current_v4_replay"], frozen)
    comparisons = {
        name: paired_formula_bootstrap(
            transitions[name],
            transitions["spectral_plus_degree_control"],
            args.bootstrap_resamples,
            args.seed + 100 + index,
        )
        for index, name in enumerate(RECIPES)
        if name not in {"spectral_only_control", "spectral_plus_degree_control"}
    }

    rng = np.random.default_rng(args.seed + 1000)
    null_deltas = []
    null_risk_nets = []
    null_moved = []
    for repeat in range(args.null_repeats):
        permuted, audit = degree_conditioned_permutation(candidates, rng)
        result, _ = run_loso(permuted, RECIPES["current_v4_replay"])
        summary = summarize(result)
        null_deltas.append(summary["delta_recall1"])
        null_risk_nets.append(summary["risk_weighted_net_lambda2"])
        null_moved.append(audit["moved_fraction"])
        if (repeat + 1) % 10 == 0 or repeat + 1 == args.null_repeats:
            print(f"[degree-conditioned null] {repeat + 1}/{args.null_repeats}", flush=True)

    observed = reports["current_v4_replay"]["pooled"]
    observed_delta = float(observed["delta_recall1"])
    observed_risk = float(observed["risk_weighted_net_lambda2"])
    degree_null = {
        "definition": (
            "spectral score and exact candidate degree fixed; joint remaining network block "
            "permuted within unit x degree-quintile x path/edge-availability strata"
        ),
        "repeats": args.null_repeats,
        "mean_moved_fraction": float(np.mean(null_moved)),
        "delta_recall1": {
            "observed": observed_delta,
            "null_mean": float(np.mean(null_deltas)),
            "null_p95": float(np.quantile(null_deltas, 0.95)),
            "empirical_p_ge_observed": float(
                (1 + np.sum(np.asarray(null_deltas) >= observed_delta))
                / (1 + args.null_repeats)
            ),
        },
        "risk_weighted_net_lambda2": {
            "observed": observed_risk,
            "null_mean": float(np.mean(null_risk_nets)),
            "null_p95": float(np.quantile(null_risk_nets, 0.95)),
            "empirical_p_ge_observed": float(
                (1 + np.sum(np.asarray(null_risk_nets) >= observed_risk))
                / (1 + args.null_repeats)
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output_dir / "candidate_features_rich.csv.gz"
    candidates.to_csv(candidate_path, index=False, compression="gzip")
    transition_hashes: dict[str, str] = {}
    for name, frame in transitions.items():
        path = args.output_dir / f"{name}__transitions.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        transition_hashes[name] = sha256(path)

    report = {
        "status": "bioaware_action_atlas_v3_complete",
        "formal": True,
        "opened_development": True,
        "model_scope": "candidate-group action discovery only; no embedding fit",
        "candidate_protocol": {
            "queries": int(candidates["query_id"].nunique()),
            "candidate_pairs": int(len(candidates)),
            "truth_identities": int(candidates["truth_candidate_id"].nunique()),
            "truth_formulas": int(candidates["truth_formula"].nunique()),
        },
        "fixed_recipes": reports,
        "current_v4_exact_replay": replay,
        "paired_formula_bootstrap_vs_degree_control": comparisons,
        "degree_fixed_path_availability_conditioned_null": degree_null,
        "gates": {
            "current_v4_exact_replay": replay["pass"],
            "current_v4_delta_ge_3pp": observed_delta >= 0.03,
            "current_v4_formula_ci_positive": (
                reports["current_v4_replay"]["formula_cluster_bootstrap"]["ci_low"] > 0
            ),
            "current_v4_corrected_gt_2x_introduced": (
                observed["corrected"] > 2 * observed["introduced"]
            ),
            "current_v4_corrected_identities_ge_20": (
                observed["corrected_identities"] >= 20
            ),
            "degree_conditioned_delta_null_p_le_0_05": (
                degree_null["delta_recall1"]["empirical_p_ge_observed"] <= 0.05
            ),
            "degree_conditioned_risk_null_p_le_0_05": (
                degree_null["risk_weighted_net_lambda2"]["empirical_p_ge_observed"] <= 0.05
            ),
        },
        "contracts": {
            "threshold_search": False,
            "all_fixed_recipes_reported": True,
            "outer_test_truth_identity_and_formula_purged": True,
            "candidate_identity_as_feature": False,
            "truth_or_formula_as_feature": False,
            "phenotype_used": False,
            "P2b_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "frozen_current_transitions_sha256": sha256(args.frozen_current_transitions),
            "rich_candidate_features_sha256": sha256(candidate_path),
            "edge_tables": edge_provenance,
            "transition_sha256": transition_hashes,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened action-mining result. A passing recipe identifies a candidate action for "
            "freezing; it is not independent confirmation, SOTA, or shared-embedding gain."
        ),
    }
    report["pass_current_action_specificity"] = bool(all(report["gates"].values()))
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
