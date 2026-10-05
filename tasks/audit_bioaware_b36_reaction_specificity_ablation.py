#!/usr/bin/env python
"""Measure whether BioAware gains are specific to reaction context.

This is an opened-development ablation on the frozen six-domain B12/B16
candidate universe.  Every evidence arm uses the same shallow pairwise model,
the same outer leave-domain-out split, the same truth identity/formula purge,
and an inner leave-domain-out intervention-gate selection.  Therefore the
pre-registered primary contrasts change evidence, not model capacity or the
evaluation graph.

Two feature-only negative controls derange reaction-context vectors without
using outcomes: within-query cyclic derangement and cross-query derangement
inside coarse catalogue strata.  These are association-null controls; they
are not a substitute for a future raw Rhea degree-preserving edge rewire.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    atomic_json,
    sha256,
    summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    split_domain,
)
from audit_bioaware_b16_pairwise_nonlinear_action import fit_model  # noqa: E402


RISK_PENALTY = 2
B16_CATALOG_REPLAY = {
    "queries": 860,
    "baseline_recall1": 0.6604651162790698,
    "recall1": 0.708139534883721,
    "delta_recall1": 0.047674418604651166,
    "corrected": 44,
    "introduced": 3,
}
SPECTRAL_FEATURES = ["spectral_score"]
CATALOG_FEATURES = [
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
    "log_reference_spectra",
]
REACTION_FEATURES = [
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
    "b36_known_path_present",
    "b36_edge0_present",
]
REAL_ARMS: dict[str, list[str]] = {
    "spectral_only": SPECTRAL_FEATURES,
    "spectral_plus_catalog": SPECTRAL_FEATURES + CATALOG_FEATURES,
    "spectral_plus_reaction": SPECTRAL_FEATURES + REACTION_FEATURES,
    "spectral_plus_catalog_plus_reaction": (
        SPECTRAL_FEATURES + CATALOG_FEATURES + REACTION_FEATURES
    ),
}
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.04, 0.05, 0.08)
    for probability in (0.55, 0.60, 0.65, 0.70, 0.75)
)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def prepare_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_candidate_id", "baseline_correct",
        "baseline_gap", "is_positive", *SPECTRAL_FEATURES,
        *CATALOG_FEATURES, *(feature for feature in REACTION_FEATURES
                            if not feature.startswith("b36_")),
    }
    if missing := required - set(frame.columns):
        raise RuntimeError(f"B36 candidate table misses: {sorted(missing)}")
    output = frame.copy()
    numerical = sorted(
        set(SPECTRAL_FEATURES + CATALOG_FEATURES + [
            feature for feature in REACTION_FEATURES
            if not feature.startswith("b36_")
        ])
    )
    for column in numerical:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["b36_known_path_present"] = (
        output["known_path_fraction"].astype(float) > 0
    ).astype(float)
    output["b36_edge0_present"] = (
        (output["edge0_complete_fraction"].astype(float) > 0)
        | (output["edge0_bottleneck_mean"].astype(float) > 0)
    ).astype(float)
    features = sorted(set().union(*REAL_ARMS.values()))
    if not np.isfinite(output[features].to_numpy(float)).all():
        raise RuntimeError("B36 evidence features contain non-finite values")
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B36 candidate identity is not unique within query")
    positives = output.groupby("query_id", sort=False)["is_positive"].sum()
    if not positives.eq(1).all():
        raise RuntimeError("B36 requires exactly one positive candidate per query")
    return output


def _catalog_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcome-free, absolute (not quantile-fitted) catalogue strata."""
    strata = pd.DataFrame(index=frame.index)
    strata["source"] = frame["source"].astype(str)
    strata["member"] = (frame["network_member"].astype(float) > 0.5).astype(int)
    strata["degree"] = np.floor(
        frame["known_log_degree"].astype(float).clip(0, 8)
    ).astype(int)
    strata["mass"] = np.floor(
        frame["known_mass_candidate_fraction"].astype(float).clip(0, 0.999999) * 4
    ).astype(int)
    strata["references"] = np.floor(
        frame["log_reference_spectra"].astype(float).clip(0, 8)
    ).astype(int)
    return strata


def derange_reaction_context(
    frame: pd.DataFrame,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Derange reaction vectors without reading identity, truth, or outcome."""
    if method not in {"within_query", "catalog_stratum"}:
        raise ValueError(method)
    output = frame.copy()
    original = output[REACTION_FEATURES].to_numpy(float, copy=True)
    positions = np.arange(len(output), dtype=int)
    if method == "within_query":
        keys = pd.DataFrame({"query_id": output["query_id"].astype(str).to_numpy()})
    else:
        keys = _catalog_strata(output).reset_index(drop=True)
    assignments = positions.copy()
    groups = 0
    movable_groups = 0
    singleton_rows = 0
    for key, local in keys.groupby(list(keys.columns), sort=True, dropna=False):
        destination = local.index.to_numpy(dtype=int)
        groups += 1
        if len(destination) < 2:
            singleton_rows += len(destination)
            continue
        movable_groups += 1
        rng = np.random.default_rng(stable_seed(seed, method, key))
        order = destination[rng.permutation(len(destination))]
        shift = 1 + stable_seed(seed, "shift", method, key) % (len(order) - 1)
        source = np.roll(order, int(shift))
        assignments[order] = source
    output.loc[:, REACTION_FEATURES] = original[assignments]
    moved = assignments != positions
    changed = np.any(np.abs(original[assignments] - original) > 1e-12, axis=1)

    # Exact multiset preservation is the scientific contract of the null.
    before = pd.concat([keys, frame[REACTION_FEATURES].reset_index(drop=True)], axis=1)
    after = pd.concat([keys, output[REACTION_FEATURES].reset_index(drop=True)], axis=1)
    group_columns = list(keys.columns)
    before_sum = before.groupby(group_columns, dropna=False, sort=True)[
        REACTION_FEATURES
    ].sum().sort_index()
    after_sum = after.groupby(group_columns, dropna=False, sort=True)[
        REACTION_FEATURES
    ].sum().sort_index()
    maximum_sum_error = float(
        np.max(np.abs(before_sum.to_numpy(float) - after_sum.to_numpy(float)), initial=0)
    )
    if maximum_sum_error > 1e-9:
        raise RuntimeError(f"{method}: reaction-feature multiset was not preserved")
    return output, {
        "method": method,
        "rows": int(len(output)),
        "groups": int(groups),
        "movable_groups": int(movable_groups),
        "singleton_rows": int(singleton_rows),
        "assigned_from_other_row": int(moved.sum()),
        "value_changed_rows": int(changed.sum()),
        "value_changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "maximum_group_feature_sum_error": maximum_sum_error,
    }


def arm_specifications(null_repeats: int, seed: int) -> list[dict]:
    if null_repeats < 1:
        raise ValueError("null_repeats must be positive")
    arms = [
        {"arm": name, "features": features, "null_method": None, "null_seed": None}
        for name, features in REAL_ARMS.items()
    ]
    full = REAL_ARMS["spectral_plus_catalog_plus_reaction"]
    for repeat in range(null_repeats):
        for method in ("within_query", "catalog_stratum"):
            arms.append({
                "arm": f"full_{method}_deranged_r{repeat:02d}",
                "features": full,
                "null_method": method,
                "null_seed": stable_seed(seed, "null", method, repeat),
            })
    return arms


def transformed_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specification: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    method = specification["null_method"]
    if method is None:
        return train, test, []
    seed = int(specification["null_seed"])
    train_null, train_audit = derange_reaction_context(train, method, seed)
    test_null, test_audit = derange_reaction_context(test, method, seed + 1)
    train_audit["partition"] = "train"
    test_audit["partition"] = "test"
    return train_null, test_null, [train_audit, test_audit]


def score_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specification: dict,
    held_label: str,
    model_seed: int,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    local_train, local_test, transform_audits = transformed_frames(
        train, test, specification
    )
    features = list(specification["features"])
    model, fit_report = fit_model(local_train, features, model_seed)
    rows: list[dict] = []
    for query_id, group in local_test.groupby("query_id", sort=False):
        baseline = str(group["baseline_candidate_id"].iloc[0])
        baseline_row = group.loc[group["candidate_id"].astype(str).eq(baseline)]
        if len(baseline_row) != 1:
            raise RuntimeError(f"{query_id}: baseline candidate is not unique")
        baseline_vector = baseline_row[features].to_numpy(float)[0]
        candidate_vectors = group[features].to_numpy(float)
        probabilities = model.predict_proba(candidate_vectors - baseline_vector)[:, 1]
        local = group.copy()
        local["preference_probability"] = probabilities
        maximum = float(local["preference_probability"].max())
        top = local.loc[
            np.isclose(local["preference_probability"], maximum, rtol=0, atol=1e-12)
        ].sort_values("candidate_id", kind="stable")
        rows.append({
            "query_id": str(query_id),
            "held_label": held_label,
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": str(group["truth_candidate_id"].iloc[0]),
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "proposed_candidate_id": str(top.iloc[0]["candidate_id"]),
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "proposal_unique": bool(len(top) == 1),
            "proposal_probability": maximum,
            "baseline_gap": float(group["baseline_gap"].iloc[0]),
            "candidate_count": int(len(group)),
        })
    return pd.DataFrame(rows), {"held_label": held_label, **fit_report}, transform_audits


def no_op(scored: pd.DataFrame) -> pd.DataFrame:
    output = scored.copy()
    output["intervene"] = False
    output["final_candidate_id"] = output["baseline_candidate_id"].astype(str)
    output["final_correct"] = output["baseline_correct"].astype(bool)
    output["corrected"] = False
    output["introduced"] = False
    output["delta"] = 0
    output["gate_margin"] = -1.0
    output["gate_probability"] = 1.0
    output["gate_name"] = "no_op"
    return output


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
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    output["gate_margin"] = float(margin)
    output["gate_probability"] = float(probability)
    output["gate_name"] = f"margin={margin:.2f}|probability={probability:.2f}"
    return output


def choose_gate(scored: pd.DataFrame) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    configurations: list[tuple[str, pd.DataFrame]] = [("no_op", no_op(scored))]
    configurations.extend(
        (
            f"margin={margin:.2f}|probability={probability:.2f}",
            apply_gate(scored, margin, probability),
        )
        for margin, probability in GATE_GRID
    )
    for name, result in configurations:
        overall = summarize(result)
        by_domain = {
            domain: summarize(result.loc[result["source"].eq(domain)])
            for domain in sorted(result["source"].unique())
        }
        safe = all(item["risk_net_lambda2"] >= 0 for item in by_domain.values())
        ledger.append({
            "gate_name": name,
            "margin": float(result["gate_margin"].iloc[0]),
            "probability": float(result["gate_probability"].iloc[0]),
            **overall,
            "every_inner_domain_risk_nonnegative": bool(safe),
        })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and (
            item["gate_name"] == "no_op"
            or item["corrected"] > RISK_PENALTY * item["introduced"]
        )
    ]
    if not eligible:
        raise RuntimeError("no-op must always be an eligible gate")
    selected = max(eligible, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"],
        -item["intervention_rate"],
        item["gate_name"] == "no_op",
    ))
    return {**selected, "selection_reason": "maximum_safe_inner_oof_risk_net_rate"}, ledger


def apply_selected_gate(scored: pd.DataFrame, selected: dict) -> pd.DataFrame:
    if selected["gate_name"] == "no_op":
        return no_op(scored)
    return apply_gate(scored, float(selected["margin"]), float(selected["probability"]))


def cluster_ci_values(
    values: np.ndarray,
    clusters: Iterable[str],
    repeats: int,
    seed: int,
) -> dict:
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster": list(clusters)})
    grouped = frame.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def summarize_arm(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    base = summarize(frame)
    return {
        **base,
        "identity_cluster_bootstrap": cluster_ci_values(
            frame["delta"].to_numpy(float),
            frame["truth_candidate_id"].astype(str), repeats, seed,
        ),
        "formula_cluster_bootstrap": cluster_ci_values(
            frame["delta"].to_numpy(float),
            frame["truth_formula"].astype(str), repeats, seed + 1,
        ),
        "mcnemar_exact_p": exact_mcnemar(base["corrected"], base["introduced"]),
    }


def exact_mcnemar(corrected: int, introduced: int) -> float:
    discordant = int(corrected + introduced)
    if discordant == 0:
        return 1.0
    lower = min(int(corrected), int(introduced))
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * tail))


def paired_contrast(
    all_results: pd.DataFrame,
    left_arm: str,
    right_arm: str,
    repeats: int,
    seed: int,
) -> dict:
    columns = [
        "query_id", "truth_candidate_id", "truth_formula", "source",
        "baseline_correct", "final_correct",
    ]
    left = all_results.loc[all_results["arm"].eq(left_arm), columns]
    right = all_results.loc[all_results["arm"].eq(right_arm), columns]
    joined = left.merge(
        right, on="query_id", how="inner", validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(joined) != 860:
        raise RuntimeError(f"{left_arm} vs {right_arm}: incomplete query pairing")
    for column in ("truth_candidate_id", "truth_formula", "source", "baseline_correct"):
        if not joined[f"{column}_left"].astype(str).equals(
            joined[f"{column}_right"].astype(str)
        ):
            raise RuntimeError(f"{left_arm} vs {right_arm}: {column} mismatch")
    effect = (
        joined["final_correct_left"].astype(int)
        - joined["final_correct_right"].astype(int)
    ).to_numpy(float)
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "mean_top1_difference": float(effect.mean()),
        "left_better": int((effect > 0).sum()),
        "right_better": int((effect < 0).sum()),
        "identity_cluster_bootstrap": cluster_ci_values(
            effect, joined["truth_candidate_id_left"].astype(str), repeats, seed,
        ),
        "formula_cluster_bootstrap": cluster_ci_values(
            effect, joined["truth_formula_left"].astype(str), repeats, seed + 1,
        ),
        "mcnemar_exact_p": exact_mcnemar(
            int((effect > 0).sum()), int((effect < 0).sum())
        ),
    }


def empirical_p(real: float, null: list[float]) -> float:
    return float((1 + sum(value >= real - 1e-15 for value in null)) / (1 + len(null)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = prepare_candidates(candidates)
    specifications = arm_specifications(args.null_repeats, args.seed)
    outer_tables: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    selection_rows: list[dict] = []
    transform_rows: list[dict] = []

    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        for arm_index, specification in enumerate(specifications):
            arm = str(specification["arm"])
            # Common random numbers across evidence arms: an arm may change
            # features, never the model RNG.  This also exactly replays B16.
            model_seed = args.seed + 100 * outer_index
            outer_scored, outer_fit, audits = score_arm(
                outer_train, outer_test, specification, outer_domain, model_seed
            )
            for audit in audits:
                transform_rows.append({
                    "outer_domain": outer_domain,
                    "inner_domain": "__outer_fit__",
                    "arm": arm,
                    **audit,
                })
            inner_parts: list[pd.DataFrame] = []
            inner_fit_reports: list[dict] = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                if inner_test["query_id"].nunique() < 10:
                    raise RuntimeError(f"{outer_domain}/{inner_domain}: too few inner queries")
                inner_scored, inner_fit, inner_audits = score_arm(
                    inner_train, inner_test, specification,
                    f"outer={outer_domain}|inner={inner_domain}",
                    model_seed + inner_index + 1,
                )
                inner_parts.append(inner_scored)
                inner_fit_reports.append(inner_fit)
                for audit in inner_audits:
                    transform_rows.append({
                        "outer_domain": outer_domain,
                        "inner_domain": inner_domain,
                        "arm": arm,
                        **audit,
                    })
            inner_scored = pd.concat(inner_parts, ignore_index=True)
            selected, ledger = choose_gate(inner_scored)
            for row in ledger:
                selection_rows.append({
                    "outer_domain": outer_domain,
                    "arm": arm,
                    **row,
                })
            result = apply_selected_gate(outer_scored, selected)
            result["arm"] = arm
            result["null_method"] = specification["null_method"] or "none"
            result["outer_domain"] = outer_domain
            outer_tables.append(result)
            outer_summary = summarize(result)
            fold_reports.append({
                "outer_domain": outer_domain,
                "arm": arm,
                "selected_gate": selected,
                "outer_result": outer_summary,
                "outer_fit": outer_fit,
                "inner_fits": inner_fit_reports,
            })
            print(
                f"[B36 {outer_domain} {arm}] gate={selected['gate_name']} "
                f"dR1={outer_summary['delta_recall1']:+.4f} "
                f"C/I={outer_summary['corrected']}/{outer_summary['introduced']}",
                flush=True,
            )

    results = pd.concat(outer_tables, ignore_index=True)
    expected_queries = 860
    for arm in [specification["arm"] for specification in specifications]:
        local = results.loc[results["arm"].eq(arm)]
        if len(local) != expected_queries or local["query_id"].nunique() != expected_queries:
            raise RuntimeError(f"{arm}: outer-domain OOF coverage changed")
    baseline_matrix = results.pivot(
        index="query_id", columns="arm", values="baseline_correct"
    )
    if not baseline_matrix.nunique(axis=1).eq(1).all():
        raise RuntimeError("baseline correctness changed across B36 arms")

    arm_reports: dict[str, dict] = {}
    for index, specification in enumerate(specifications):
        arm = str(specification["arm"])
        local = results.loc[results["arm"].eq(arm)].copy()
        arm_reports[arm] = {
            **summarize_arm(
                local, args.bootstrap_resamples, args.seed + 100000 + index * 10
            ),
            "by_domain": {
                domain: summarize(local.loc[local["source"].eq(domain)])
                for domain in EXPECTED_DOMAINS
            },
        }

    catalog_replay = arm_reports["spectral_plus_catalog"]
    replay_mismatches = {
        key: {"expected": expected, "observed": catalog_replay[key]}
        for key, expected in B16_CATALOG_REPLAY.items()
        if not (
            int(catalog_replay[key]) == int(expected)
            if key in {"queries", "corrected", "introduced"}
            else np.isclose(float(catalog_replay[key]), float(expected), atol=1e-15, rtol=0)
        )
    }

    full_arm = "spectral_plus_catalog_plus_reaction"
    catalog_arm = "spectral_plus_catalog"
    reaction_arm = "spectral_plus_reaction"
    spectral_arm = "spectral_only"
    full_vs_catalog = paired_contrast(
        results, full_arm, catalog_arm, args.bootstrap_resamples, args.seed + 200001
    )
    reaction_vs_spectral = paired_contrast(
        results, reaction_arm, spectral_arm, args.bootstrap_resamples, args.seed + 200002
    )
    full_vs_spectral = paired_contrast(
        results, full_arm, spectral_arm, args.bootstrap_resamples, args.seed + 200003
    )

    null_reports: dict[str, dict] = {}
    null_by_method: dict[str, list[str]] = {"within_query": [], "catalog_stratum": []}
    for index, specification in enumerate(specifications):
        if specification["null_method"] is None:
            continue
        arm = str(specification["arm"])
        method = str(specification["null_method"])
        null_by_method[method].append(arm)
        null_reports[arm] = paired_contrast(
            results, full_arm, arm, args.bootstrap_resamples,
            args.seed + 300000 + index * 10,
        )

    null_summaries: dict[str, dict] = {}
    full_delta = float(arm_reports[full_arm]["delta_recall1"])
    full_risk = float(arm_reports[full_arm]["risk_net_lambda2"])
    for method, arms in null_by_method.items():
        deltas = [float(arm_reports[arm]["delta_recall1"]) for arm in arms]
        risks = [float(arm_reports[arm]["risk_net_lambda2"]) for arm in arms]
        null_summaries[method] = {
            "arms": arms,
            "delta_recall1": {
                "minimum": float(min(deltas)),
                "median": float(np.median(deltas)),
                "maximum": float(max(deltas)),
                "real": full_delta,
                "empirical_one_sided_p": empirical_p(full_delta, deltas),
            },
            "risk_net_lambda2": {
                "minimum": float(min(risks)),
                "median": float(np.median(risks)),
                "maximum": float(max(risks)),
                "real": full_risk,
                "empirical_one_sided_p": empirical_p(full_risk, risks),
            },
            "full_minus_null_formula_ci_positive_count": int(sum(
                null_reports[arm]["formula_cluster_bootstrap"]["ci_low"] > 0
                for arm in arms
            )),
        }

    gates = {
        "frozen_B16_catalog_reproduction": not replay_mismatches,
        "full_gain_ge_3pp": full_delta >= 0.03,
        "full_corrected_gt_2x_introduced": (
            arm_reports[full_arm]["corrected"]
            > 2 * arm_reports[full_arm]["introduced"]
        ),
        "full_every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0
            for item in arm_reports[full_arm]["by_domain"].values()
        ),
        "full_vs_catalog_identity_ci_positive": (
            full_vs_catalog["identity_cluster_bootstrap"]["ci_low"] > 0
        ),
        "full_vs_catalog_formula_ci_positive": (
            full_vs_catalog["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "reaction_vs_spectral_identity_ci_positive": (
            reaction_vs_spectral["identity_cluster_bootstrap"]["ci_low"] > 0
        ),
        "reaction_vs_spectral_formula_ci_positive": (
            reaction_vs_spectral["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "full_beats_every_within_query_null": (
            full_delta > null_summaries["within_query"]["delta_recall1"]["maximum"]
        ),
        "full_beats_every_catalog_stratum_null": (
            full_delta > null_summaries["catalog_stratum"]["delta_recall1"]["maximum"]
        ),
        "within_query_null_empirical_p_le_0_10": (
            null_summaries["within_query"]["delta_recall1"]["empirical_one_sided_p"] <= 0.10
        ),
        "catalog_stratum_null_empirical_p_le_0_10": (
            null_summaries["catalog_stratum"]["delta_recall1"]["empirical_one_sided_p"] <= 0.10
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "nested_domain_loso_arm_transitions.csv.gz"
    selection_path = args.output_dir / "inner_gate_selection_ledger.csv.gz"
    transform_path = args.output_dir / "null_transform_audit.csv.gz"
    results.to_csv(transitions_path, index=False, compression="gzip")
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False, compression="gzip")
    pd.DataFrame(transform_rows).to_csv(transform_path, index=False, compression="gzip")

    report = {
        "status": "bioaware_b36_reaction_specificity_ablation_complete",
        "formal": True,
        "scientific_pass": bool(all(gates.values())),
        "protocol": (
            "six opened development domains; fixed evidence ablation; symmetric "
            "pairwise shallow-HGB; outer leave-domain-out with held truth identity "
            "and formula purge; inner leave-domain-out gate selection with explicit no-op"
        ),
        "frozen_B16_catalog_reproduction": {
            **B16_CATALOG_REPLAY,
            "observed": {
                key: catalog_replay[key] for key in B16_CATALOG_REPLAY
            },
            "mismatches": replay_mismatches,
            "pass": not replay_mismatches,
        },
        "universe": {
            "queries": 860,
            "domains": list(EXPECTED_DOMAINS),
            "candidate_rows": int(len(candidates.loc[candidates["polarity"].eq("negative")])),
        },
        "real_arms": {
            arm: arm_reports[arm] for arm in REAL_ARMS
        },
        "primary_contrasts": {
            "full_vs_catalog_only": full_vs_catalog,
            "reaction_vs_spectral_only": reaction_vs_spectral,
            "full_vs_spectral_only": full_vs_spectral,
        },
        "null_controls": {
            "repeats_per_method": int(args.null_repeats),
            "arm_results": {
                arm: arm_reports[arm]
                for arms in null_by_method.values()
                for arm in arms
            },
            "summaries": null_summaries,
            "paired_full_minus_null": null_reports,
            "interpretation": (
                "Feature-association nulls only. A passing result still requires a "
                "future raw-Rhea degree-preserving edge-rewire confirmation."
            ),
        },
        "feature_contract": {
            "spectral": SPECTRAL_FEATURES,
            "catalog": CATALOG_FEATURES,
            "reaction": REACTION_FEATURES,
            "same_model_capacity_within_each_reported_feature_set": True,
            "outcome_used_to_construct_null": False,
        },
        "folds": fold_reports,
        "gates": gates,
        "contracts": {
            "all_domains_are_opened_development": True,
            "outer_outcomes_used_for_model_or_gate_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "candidate_or_truth_identity_as_feature": False,
            "phenotype_used": False,
            "P2b_used": False,
            "shared_embedding_changed": False,
            "scientific_failure_returns_valid_artifact": True,
        },
        "provenance": {
            **provenance["provenance"],
            "scikit_learn_version": sklearn.__version__,
            "transitions_sha256": sha256(transitions_path),
            "selection_ledger_sha256": sha256(selection_path),
            "null_transform_audit_sha256": sha256(transform_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A passing B36 result would establish reaction-context-specific action "
            "signal on opened development domains, not an independent blind result, "
            "SOTA, biological mechanism, or improved shared embedding. A failing "
            "result assigns the current gain to spectral/catalogue priors and blocks "
            "reaction-context embedding fine-tuning."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    console_summary = {
        "status": report["status"],
        "scientific_pass": report["scientific_pass"],
        "frozen_B16_catalog_reproduction": report["frozen_B16_catalog_reproduction"],
        "real_arms": {
            arm: {
                key: values[key]
                for key in (
                    "baseline_recall1", "recall1", "delta_recall1",
                    "corrected", "introduced", "risk_net_lambda2",
                )
            }
            for arm, values in report["real_arms"].items()
        },
        "primary_contrasts": report["primary_contrasts"],
        "null_summaries": report["null_controls"]["summaries"],
        "gates": report["gates"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(console_summary, indent=2, sort_keys=True), flush=True)
    print(
        "[B36 scientific decision]",
        "PASS_TO_RAW_GRAPH_REWIRE" if report["scientific_pass"] else "REACTION_SPECIFICITY_NOT_ESTABLISHED",
        flush=True,
    )


if __name__ == "__main__":
    main()
