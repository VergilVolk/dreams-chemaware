#!/usr/bin/env python
"""Mine a safer nonlinear BioAware catalogue-prior action.

This opened-development audit starts from the exactly reproducible B4 graph
opportunity action.  It asks one narrow question: does a small set of
deployment-visible interaction terms improve the negative-ion action without
using reaction identity, phenotype, candidate identity, or held-source
outcomes?

Model recipe and intervention gate are selected inside every outer
leave-one-source-out fold by an inner leave-one-source-out loop.  Truth
identities and formulae of every held fold are purged before fitting.  The
output therefore remains an action-discovery result, not external validation
and not a shared-embedding result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_b1_multisource_action import (  # noqa: E402
    EXPECTED_SOURCES,
    fit_ranker,
)


BASE_FEATURES = [
    "spectral_score",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
]

# This is a fixed mechanism matrix, not an unrestricted feature search.
# Query-constant variables only enter through interactions with candidate-level
# catalogue evidence, so they can change within-query ranking.
FEATURE_RECIPES: dict[str, list[str]] = {
    "linear_b4_replay": list(BASE_FEATURES),
    "spectral_catalog_interactions": [
        *BASE_FEATURES,
        "spectral_x_member",
        "spectral_x_degree",
        "spectral_x_mass_coverage",
        "member_x_mass_coverage",
        "degree_x_mass_coverage",
    ],
    "ambiguity_density_interactions": [
        *BASE_FEATURES,
        "spectral_x_member",
        "spectral_x_degree",
        "spectral_x_mass_coverage",
        "member_x_mass_coverage",
        "degree_x_mass_coverage",
        "gap_x_member",
        "gap_x_degree",
        "gap_x_mass_coverage",
        "log_candidates_x_member",
        "log_candidates_x_degree",
        "log_candidates_x_mass_coverage",
    ],
}

# Frozen before B11 outcomes.  Gate selection happens only on inner OOF rows.
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.04, 0.05, 0.08)
    for probability in (0.65, 0.70, 0.75)
)
REPLAY_GATE = (0.05, 0.75)
RISK_PENALTY = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_candidate_id", "baseline_correct",
        "baseline_gap", *BASE_FEATURES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"B11 candidate cache missing columns: {sorted(missing)}")
    output = frame.copy()
    numerical = ["spectral_score", "known_log_degree", "known_mass_candidate_fraction"]
    for name in numerical:
        output[name] = pd.to_numeric(output[name], errors="raise").astype(float)
    output["network_member"] = pd.to_numeric(
        output["network_member"], errors="raise"
    ).astype(float)
    if not set(output["network_member"].unique()).issubset({0.0, 1.0}):
        raise RuntimeError("network_member is not binary")
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate candidate identity within query")
    positive_count = output.groupby("query_id", sort=False).apply(
        lambda group: int(
            group["candidate_id"].astype(str).eq(
                group["truth_candidate_id"].astype(str)
            ).sum()
        ),
        include_groups=False,
    )
    if not positive_count.eq(1).all():
        raise RuntimeError("every query must contain exactly one truth candidate")

    spectral = output["spectral_score"]
    member = output["network_member"]
    degree = output["known_log_degree"].clip(lower=0.0)
    mass = output["known_mass_candidate_fraction"].clip(lower=0.0, upper=1.0)
    gap = pd.to_numeric(output["baseline_gap"], errors="raise").astype(float)
    candidate_count = output.groupby("query_id")["candidate_id"].transform("count")
    log_candidates = np.log(candidate_count.astype(float))

    interactions = {
        "spectral_x_member": spectral * member,
        "spectral_x_degree": spectral * degree,
        "spectral_x_mass_coverage": spectral * mass,
        "member_x_mass_coverage": member * mass,
        "degree_x_mass_coverage": degree * mass,
        "gap_x_member": gap * member,
        "gap_x_degree": gap * degree,
        "gap_x_mass_coverage": gap * mass,
        "log_candidates_x_member": log_candidates * member,
        "log_candidates_x_degree": log_candidates * degree,
        "log_candidates_x_mass_coverage": log_candidates * mass,
    }
    for name, values in interactions.items():
        output[name] = values.astype(float)
    all_features = sorted(set().union(*FEATURE_RECIPES.values()))
    if not np.isfinite(output[all_features].to_numpy(float)).all():
        raise RuntimeError("non-finite B11 model feature")
    return output


def outer_training_pool(frame: pd.DataFrame, held_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    held_all_polarities = frame.loc[frame["source"].eq(held_source)].copy()
    test = held_all_polarities.loc[
        held_all_polarities["polarity"].eq("negative")
    ].copy()
    # Exact B4 contract: purge the truth identities and formulae observed in
    # either polarity of the held biological source, then evaluate only the
    # held negative-ion queries.
    held_identities = set(held_all_polarities["truth_candidate_id"].astype(str))
    held_formulas = set(held_all_polarities["truth_formula"].astype(str))
    train = frame.loc[
        ~frame["source"].eq(held_source)
        & ~frame["truth_candidate_id"].astype(str).isin(held_identities)
        & ~frame["truth_formula"].astype(str).isin(held_formulas)
    ].copy()
    if set(train["truth_candidate_id"].astype(str)) & held_identities:
        raise RuntimeError(f"{held_source}: outer truth-identity leakage")
    if set(train["truth_formula"].astype(str)) & held_formulas:
        raise RuntimeError(f"{held_source}: outer truth-formula leakage")
    return train, test


def score_queries(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    held_label: str,
) -> tuple[pd.DataFrame, dict]:
    scaler, model = fit_ranker(train, features)
    local = test.copy()
    local["model_score"] = model.decision_function(
        scaler.transform(local[features].to_numpy(float))
    )
    rows: list[dict] = []
    for query_id, group in local.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline = str(group["baseline_candidate_id"].iloc[0])
        maximum = float(group["model_score"].max())
        top = group.loc[
            np.isclose(group["model_score"], maximum, rtol=0, atol=1e-12)
        ].sort_values("candidate_id", kind="stable")
        proposed = str(top["candidate_id"].iloc[0])
        proposed_row = top.iloc[0]
        baseline_row = group.loc[
            group["candidate_id"].astype(str).eq(baseline)
        ].iloc[0]
        baseline_score = float(
            baseline_row["model_score"]
        )
        advantage = maximum - baseline_score
        probability = float(1.0 / (1.0 + np.exp(-advantage)))
        rows.append({
            "query_id": str(query_id),
            "held_label": held_label,
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "proposed_candidate_id": proposed,
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "proposal_unique": bool(len(top) == 1),
            "proposal_probability": probability,
            "baseline_gap": float(group["baseline_gap"].iloc[0]),
            "candidate_count": int(len(group)),
            "delta_spectral_score": float(
                proposed_row["spectral_score"] - baseline_row["spectral_score"]
            ),
            "delta_network_member": float(
                proposed_row["network_member"] - baseline_row["network_member"]
            ),
            "delta_known_log_degree": float(
                proposed_row["known_log_degree"] - baseline_row["known_log_degree"]
            ),
            "delta_known_mass_candidate_fraction": float(
                proposed_row["known_mass_candidate_fraction"]
                - baseline_row["known_mass_candidate_fraction"]
            ),
        })
    scored = pd.DataFrame(rows)
    return scored, {
        "held_label": held_label,
        "train_queries": int(train["query_id"].nunique()),
        "train_identities": int(train["truth_candidate_id"].nunique()),
        "train_formulas": int(train["truth_formula"].nunique()),
        "test_queries": int(scored["query_id"].nunique()),
        "coefficients": {
            name: float(value)
            for name, value in zip(features, model.coef_[0], strict=True)
        },
    }


def apply_gate(scored: pd.DataFrame, margin: float, probability: float) -> pd.DataFrame:
    output = scored.copy()
    output["intervene"] = (
        output["proposal_unique"].astype(bool)
        & output["proposed_candidate_id"].astype(str).ne(
            output["baseline_candidate_id"].astype(str)
        )
        & output["baseline_gap"].astype(float).le(float(margin) + 1e-15)
        & output["proposal_probability"].astype(float).ge(float(probability) - 1e-15)
    )
    output["final_candidate_id"] = np.where(
        output["intervene"],
        output["proposed_candidate_id"],
        output["baseline_candidate_id"],
    )
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = (
        output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    )
    output["gate_margin"] = float(margin)
    output["gate_probability"] = float(probability)
    return output


def summarize(frame: pd.DataFrame) -> dict:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "recall1": float(frame["final_correct"].mean()),
        "delta_recall1": float(frame["delta"].mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - RISK_PENALTY * introduced,
        "interventions": int(frame["intervene"].sum()),
        "intervention_rate": float(frame["intervene"].mean()),
    }


def choose_configuration(inner_scores: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for recipe, scored in inner_scores.items():
        for margin, probability in GATE_GRID:
            result = apply_gate(scored, margin, probability)
            summary = summarize(result)
            by_source = {
                source: summarize(result.loc[result["source"].eq(source)])
                for source in sorted(result["source"].unique())
            }
            every_source_nonnegative = all(
                item["risk_net_lambda2"] >= 0 for item in by_source.values()
            )
            ledger.append({
                "recipe": recipe,
                "margin": margin,
                "probability": probability,
                **summary,
                "every_inner_source_risk_nonnegative": every_source_nonnegative,
            })
    eligible = [
        row for row in ledger
        if row["every_inner_source_risk_nonnegative"]
        and row["corrected"] > RISK_PENALTY * row["introduced"]
    ]
    if not eligible:
        # Fail safe: the exact B4 replay is the only allowed fallback.
        selected = next(
            row for row in ledger
            if row["recipe"] == "linear_b4_replay"
            and row["margin"] == REPLAY_GATE[0]
            and row["probability"] == REPLAY_GATE[1]
        )
        selected = {**selected, "selection_reason": "no_safe_inner_configuration__b4_fallback"}
        return selected, ledger
    selected = max(
        eligible,
        key=lambda row: (
            row["risk_net_lambda2"] / max(1, row["queries"]),
            -row["introduced"] / max(1, row["queries"]),
            row["delta_recall1"],
            -row["intervention_rate"],
            row["recipe"] == "linear_b4_replay",
        ),
    )
    selected = {**selected, "selection_reason": "maximum_safe_inner_oof_risk_net_rate"}
    return selected, ledger


def cluster_bootstrap(frame: pd.DataFrame, column: str, repeats: int, seed: int) -> dict:
    grouped = frame.groupby(column, sort=False)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--frozen-b4-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b4_opportunity_decomposition_local_20260906/mixed_polarity__graph_only__transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (args.candidate_features, args.frozen_b4_transitions):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = enrich(pd.read_csv(args.candidate_features))
    queries = candidates[[
        "query_id", "truth_candidate_id", "truth_formula", "source",
        "polarity", "baseline_correct",
    ]].drop_duplicates()
    negative = queries.loc[queries["polarity"].eq("negative")]
    if len(queries) != 1426 or len(negative) != 548:
        raise RuntimeError(
            f"B11 protocol changed: all={len(queries)} negative={len(negative)}"
        )

    outer_results: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    replay_results: list[pd.DataFrame] = []
    for outer_source in EXPECTED_SOURCES:
        outer_train, outer_test = outer_training_pool(candidates, outer_source)
        scored_by_recipe: dict[str, pd.DataFrame] = {}
        inner_by_recipe: dict[str, list[pd.DataFrame]] = {
            name: [] for name in FEATURE_RECIPES
        }
        inner_fit_reports: dict[str, list[dict]] = {
            name: [] for name in FEATURE_RECIPES
        }
        for recipe, features in FEATURE_RECIPES.items():
            outer_scored, outer_fit = score_queries(
                outer_train, outer_test, features, outer_source
            )
            scored_by_recipe[recipe] = outer_scored
            for inner_source in EXPECTED_SOURCES:
                if inner_source == outer_source:
                    continue
                inner_all_polarities = outer_train.loc[
                    outer_train["source"].eq(inner_source)
                ].copy()
                inner_test = inner_all_polarities.loc[
                    inner_all_polarities["polarity"].eq("negative")
                ].copy()
                inner_ids = set(
                    inner_all_polarities["truth_candidate_id"].astype(str)
                )
                inner_formulas = set(
                    inner_all_polarities["truth_formula"].astype(str)
                )
                inner_train = outer_train.loc[
                    ~outer_train["source"].eq(inner_source)
                    & ~outer_train["truth_candidate_id"].astype(str).isin(inner_ids)
                    & ~outer_train["truth_formula"].astype(str).isin(inner_formulas)
                ].copy()
                # Formula purging for the outer source can legitimately leave
                # a small inner source.  Ten is the preregistered identifiability
                # floor; every such source is still required to have
                # non-negative risk net during configuration selection.
                if inner_test["query_id"].nunique() < 10:
                    raise RuntimeError(
                        f"{outer_source}/{inner_source}: fewer than 10 inner test queries"
                    )
                scored, fit_report = score_queries(
                    inner_train,
                    inner_test,
                    features,
                    f"outer={outer_source}|inner={inner_source}",
                )
                inner_by_recipe[recipe].append(scored)
                inner_fit_reports[recipe].append(fit_report)
            inner_by_recipe[recipe] = pd.concat(
                inner_by_recipe[recipe], ignore_index=True
            )
            inner_fit_reports[recipe].append({"outer_fit": outer_fit})

        inner_scores = {
            recipe: table for recipe, table in inner_by_recipe.items()
        }
        selected, selection_ledger = choose_configuration(inner_scores)
        outer_result = apply_gate(
            scored_by_recipe[selected["recipe"]],
            float(selected["margin"]),
            float(selected["probability"]),
        )
        outer_result["selected_recipe"] = selected["recipe"]
        outer_results.append(outer_result)

        replay_results.append(apply_gate(
            scored_by_recipe["linear_b4_replay"], *REPLAY_GATE
        ))
        fold_reports.append({
            "outer_source": outer_source,
            "outer_test_queries": int(outer_test["query_id"].nunique()),
            "outer_train_queries_after_identity_formula_purge": int(
                outer_train["query_id"].nunique()
            ),
            "selected": selected,
            "outer_result": summarize(outer_result),
            "inner_selection_ledger": selection_ledger,
            "fit_reports": inner_fit_reports,
        })
        print(
            f"[B11 {outer_source}] {selected['recipe']} "
            f"margin={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"{summarize(outer_result)}",
            flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    replay = pd.concat(replay_results, ignore_index=True)
    if len(result) != 548 or result["query_id"].nunique() != 548:
        raise RuntimeError("outer OOF result changed negative-query coverage")

    frozen = pd.read_csv(args.frozen_b4_transitions)
    frozen = frozen.loc[frozen["polarity"].eq("negative")].copy()
    replay_check = replay[[
        "query_id", "proposed_candidate_id", "intervene", "final_correct",
        "corrected", "introduced",
    ]].merge(
        frozen[[
            "query_id", "proposed_candidate_id", "intervene", "gated_correct",
            "corrected", "introduced",
        ]],
        on="query_id", suffixes=("_new", "_frozen"), validate="one_to_one",
    )
    mismatch = {
        "proposed_candidate_id": int(
            replay_check["proposed_candidate_id_new"].astype(str).ne(
                replay_check["proposed_candidate_id_frozen"].astype(str)
            ).sum()
        ),
        "intervene": int(
            replay_check["intervene_new"].astype(bool).ne(
                replay_check["intervene_frozen"].astype(bool)
            ).sum()
        ),
        "final_correct": int(
            replay_check["final_correct"].astype(bool).ne(
                replay_check["gated_correct"].astype(bool)
            ).sum()
        ),
        "corrected": int(
            replay_check["corrected_new"].astype(bool).ne(
                replay_check["corrected_frozen"].astype(bool)
            ).sum()
        ),
        "introduced": int(
            replay_check["introduced_new"].astype(bool).ne(
                replay_check["introduced_frozen"].astype(bool)
            ).sum()
        ),
    }
    if any(mismatch.values()):
        raise RuntimeError(f"B4 replay mismatch: {mismatch}")

    summary = summarize(result)
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 1
    )
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 2
    )
    by_source = {
        source: summarize(result.loc[result["source"].eq(source)])
        for source in EXPECTED_SOURCES
    }
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    selected_counts = Counter(result["selected_recipe"].astype(str))
    gates = {
        "gain_ge_3pp": summary["delta_recall1"] >= 0.03,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": summary["corrected"] > 2 * summary["introduced"],
        "corrected_identities_ge_20": corrected["truth_candidate_id"].nunique() >= 20,
        "all_outer_sources_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_source.values()
        ),
        "b4_replay_exact": not any(mismatch.values()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "nested_source_loso_transitions.csv.gz"
    result.to_csv(transitions_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b11_catalog_interaction_action_complete",
        "formal": True,
        "protocol": (
            "negative-ion outer source-LOSO with truth identity/formula purge; "
            "recipe and gate selected by inner source-LOSO"
        ),
        "candidate_protocol": {
            "all_queries": int(len(queries)),
            "negative_queries": int(len(negative)),
            "negative_identities": int(negative["truth_candidate_id"].nunique()),
            "negative_formulas": int(negative["truth_formula"].nunique()),
            "sources": int(negative["source"].nunique()),
        },
        "feature_recipes": FEATURE_RECIPES,
        "gate_grid": [
            {"margin": margin, "probability": probability}
            for margin, probability in GATE_GRID
        ],
        "nested_oof": {
            **summary,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "formula_cluster_bootstrap": formula_ci,
            "identity_cluster_bootstrap": identity_ci,
            "by_source": by_source,
            "selected_recipe_query_counts": dict(selected_counts),
        },
        "folds": fold_reports,
        "b4_exact_replay": {
            "summary": summarize(replay),
            "mismatches": mismatch,
        },
        "gates": gates,
        "pass_to_external_catalog_action_validation": bool(all(gates.values())),
        "contracts": {
            "held_source_outcomes_used_for_recipe_or_gate_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "candidate_identity_as_feature": False,
            "truth_or_formula_as_feature": False,
            "reaction_edge_as_feature": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "frozen_b4_transitions_sha256": sha256(args.frozen_b4_transitions),
            "transitions_sha256": sha256(transitions_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened-development catalogue-prior action mining.  Passing supports "
            "a candidate-ranking action for new external validation; it is not "
            "reaction specificity, SOTA, biological mechanism, or shared-embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_external_catalog_action_validation"]:
        raise RuntimeError(f"B11 scientific gate failed: {gates}")


if __name__ == "__main__":
    main()
