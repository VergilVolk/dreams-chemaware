#!/usr/bin/env python
"""Test catalogue topology after removing catalogue-membership shortcuts.

B45 is a development-only nested leave-source-out experiment on the already
opened B42 universe.  A candidate may challenge the DreaMS baseline only when
both have exactly the same KEGG/Rhea membership signature.  Training likewise
uses only truth-negative pairs with equal membership signatures.  Consequently
the model cannot gain by promoting a catalogued molecule over an uncatalogued
one; only within-coverage topology degree can add evidence.

Three deterministic, within-query and within-membership permutations of the
degree pair are evaluated with the same nested protocol.  B45 does not read B44.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    split_domain,
)
from audit_bioaware_b36_reaction_specificity_ablation import (  # noqa: E402
    prepare_candidates,
)


RISK_PENALTY = 2
TOPOLOGY = ["independent_log_degree_mean", "independent_log_degree_min"]
SIGNATURE = ["independent_member_count", "independent_member_intersection"]
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.02, 0.04, 0.05, 0.08)
    for probability in (0.55, 0.60, 0.65, 0.70, 0.75)
)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    # scikit-learn estimators require a NumPy-compatible uint32 seed.  Four
    # digest bytes retain deterministic separation without version-dependent
    # overflow or parameter-validation failures.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_degree_permutation(
    frame: pd.DataFrame, repeat: int, seed: int
) -> pd.DataFrame:
    output = frame.copy()
    renamed = {feature: f"null{repeat}_{feature}" for feature in TOPOLOGY}
    for source, target in renamed.items():
        output[target] = output[source].to_numpy(float)
    for key, group in output.groupby(["query_id", *SIGNATURE], sort=False):
        if len(group) < 2:
            continue
        positions = group.index.to_numpy()
        rng = np.random.default_rng(stable_seed(seed, "B45-null", repeat, *key))
        shuffled = positions[rng.permutation(len(positions))]
        shift = 1 + stable_seed(seed, "B45-shift", repeat, *key) % (len(positions) - 1)
        source = np.roll(shuffled, int(shift))
        output.loc[shuffled, list(renamed.values())] = output.loc[
            source, TOPOLOGY
        ].to_numpy(float)
    for feature in TOPOLOGY:
        real = (
            output.groupby(["query_id", *SIGNATURE], sort=False)[feature]
            .apply(lambda values: sorted(values.astype(float).tolist()))
            .to_dict()
        )
        null = (
            output.groupby(["query_id", *SIGNATURE], sort=False)[renamed[feature]]
            .apply(lambda values: sorted(values.astype(float).tolist()))
            .to_dict()
        )
        if real != null:
            raise RuntimeError(f"null {repeat} does not preserve {feature} multisets")
    return output


def pairwise_training(
    frame: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    query_meta = frame[["query_id", "truth_candidate_id"]].drop_duplicates()
    identity_counts = query_meta["truth_candidate_id"].astype(str).value_counts()
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    eligible_queries = 0
    eligible_pairs = 0
    eligible_identities: set[str] = set()
    for _, group in frame.groupby("query_id", sort=False):
        positive = group.loc[group["is_positive"].astype(bool)]
        if len(positive) != 1:
            raise RuntimeError("B45 requires exactly one truth per query")
        positive_row = positive.iloc[0]
        negative = group.loc[~group["is_positive"].astype(bool)].copy()
        for column in SIGNATURE:
            negative = negative.loc[
                negative[column].astype(float).eq(float(positive_row[column]))
            ]
        if negative.empty:
            continue
        difference = positive_row[features].to_numpy(float) - negative[features].to_numpy(float)
        identity = str(positive_row["truth_candidate_id"])
        safety = 2.0 if bool(positive_row["baseline_correct"]) else 1.0
        weight = safety / (float(identity_counts[identity]) * len(negative))
        x_rows.extend((difference, -difference))
        y_rows.extend((np.ones(len(difference), dtype=int), np.zeros(len(difference), dtype=int)))
        weights.extend((np.full(len(difference), weight), np.full(len(difference), weight)))
        eligible_queries += 1
        eligible_pairs += len(difference)
        eligible_identities.add(identity)
    if not x_rows or eligible_queries < 20:
        raise RuntimeError(f"insufficient coverage-neutral training pairs: {eligible_queries}")
    x = np.vstack(x_rows)
    y = np.concatenate(y_rows)
    sample_weight = np.concatenate(weights)
    if not np.isfinite(x).all() or not np.isfinite(sample_weight).all():
        raise RuntimeError("non-finite B45 training matrix")
    return x, y, sample_weight, {
        "eligible_queries": int(eligible_queries),
        "eligible_pairs": int(eligible_pairs),
        "symmetric_training_rows": int(len(y)),
        "eligible_identities": int(len(eligible_identities)),
    }


def fit_model(
    frame: pd.DataFrame, features: list[str], seed: int
) -> tuple[HistGradientBoostingClassifier, dict]:
    x, y, weight, report = pairwise_training(frame, features)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    ).fit(x, y, sample_weight=weight)
    return model, report


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def score_queries(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    model, fit_report = fit_model(train, features, seed)
    zero_raw = float(model.decision_function(np.zeros((1, len(features))))[0])
    rows: list[dict] = []
    for query_id, group in test.groupby("query_id", sort=False):
        positive = group.loc[group["is_positive"].astype(bool)]
        if len(positive) != 1:
            raise RuntimeError(f"{query_id}: truth is not unique")
        truth = str(positive["candidate_id"].iloc[0])
        baseline = str(group["baseline_candidate_id"].iloc[0])
        baseline_rows = group.loc[group["candidate_id"].astype(str).eq(baseline)]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"{query_id}: baseline candidate is not unique")
        baseline_row = baseline_rows.iloc[0]
        eligible = group.copy()
        for column in SIGNATURE:
            eligible = eligible.loc[
                eligible[column].astype(float).eq(float(baseline_row[column]))
            ]
        if baseline not in set(eligible["candidate_id"].astype(str)):
            raise RuntimeError(f"{query_id}: baseline excluded from its own coverage stratum")
        baseline_vector = baseline_row[features].to_numpy(float)
        difference = eligible[features].to_numpy(float) - baseline_vector
        centered_raw = model.decision_function(difference) - zero_raw
        probabilities = sigmoid(centered_raw)
        maximum = float(np.max(probabilities))
        top = np.flatnonzero(np.isclose(probabilities, maximum, rtol=0, atol=1e-12))
        proposed = sorted(eligible.iloc[top]["candidate_id"].astype(str))[0]
        proposal_unique = len(top) == 1
        rows.append(
            {
                "query_id": str(query_id),
                "source": str(group["source"].iloc[0]),
                "truth_candidate_id": truth,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "baseline_candidate_id": baseline,
                "baseline_correct": bool(baseline == truth),
                "baseline_gap": float(group["baseline_gap"].iloc[0]),
                "candidate_count": int(len(group)),
                "coverage_neutral_candidate_count": int(len(eligible)),
                "truth_coverage_matches_baseline": bool(
                    all(
                        float(positive[column].iloc[0]) == float(baseline_row[column])
                        for column in SIGNATURE
                    )
                ),
                "proposed_candidate_id": proposed,
                "proposal_probability": maximum,
                "proposal_unique": bool(proposal_unique),
            }
        )
    return pd.DataFrame(rows), fit_report


def no_op(scored: pd.DataFrame) -> pd.DataFrame:
    output = scored.copy()
    output["gate_margin"] = -1.0
    output["gate_probability"] = 1.0
    output["gate_name"] = "no_op"
    output["intervene"] = False
    output["final_candidate_id"] = output["baseline_candidate_id"]
    output["final_correct"] = output["baseline_correct"].astype(bool)
    output["corrected"] = False
    output["introduced"] = False
    output["delta"] = 0
    return output


def apply_gate(scored: pd.DataFrame, margin: float, probability: float) -> pd.DataFrame:
    output = scored.copy()
    output["intervene"] = (
        output["proposal_unique"].astype(bool)
        & output["proposed_candidate_id"].astype(str).ne(
            output["baseline_candidate_id"].astype(str)
        )
        & output["baseline_gap"].astype(float).le(margin)
        & output["proposal_probability"].astype(float).ge(probability)
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
        "risk_net_lambda2": int(corrected - RISK_PENALTY * introduced),
        "interventions": int(frame["intervene"].sum()),
        "intervention_rate": float(frame["intervene"].mean()),
    }


def choose_gate(scored: pd.DataFrame) -> tuple[dict, list[dict]]:
    configurations = [("no_op", no_op(scored))]
    configurations.extend(
        (f"margin={m:.2f}|probability={p:.2f}", apply_gate(scored, m, p))
        for m, p in GATE_GRID
    )
    ledger: list[dict] = []
    for name, result in configurations:
        by_source = {
            source: summarize(group)
            for source, group in result.groupby("source", sort=True)
        }
        safe = all(item["risk_net_lambda2"] >= 0 for item in by_source.values())
        ledger.append(
            {
                "gate_name": name,
                "margin": float(result["gate_margin"].iloc[0]),
                "probability": float(result["gate_probability"].iloc[0]),
                **summarize(result),
                "every_inner_source_risk_nonnegative": bool(safe),
            }
        )
    eligible = [
        row
        for row in ledger
        if row["every_inner_source_risk_nonnegative"]
        and (row["gate_name"] == "no_op" or row["corrected"] > 2 * row["introduced"])
    ]
    if not eligible:
        raise RuntimeError("B45 no-op must always be eligible")
    selected = max(
        eligible,
        key=lambda row: (
            row["risk_net_lambda2"] / max(1, row["queries"]),
            -row["introduced"] / max(1, row["queries"]),
            row["delta_recall1"],
            -row["intervention_rate"],
            row["gate_name"] == "no_op",
        ),
    )
    return selected, ledger


def cluster_bootstrap(
    values: Iterable[float], clusters: Iterable[str], repeats: int, seed: int
) -> dict:
    frame = pd.DataFrame({"value": list(values), "cluster": list(clusters)})
    grouped = frame.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(frame["value"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def self_check() -> None:
    toy = pd.DataFrame(
        {
            "query_id": ["q"] * 4,
            "independent_member_count": [1.0] * 4,
            "independent_member_intersection": [0.0] * 4,
            "independent_log_degree_mean": [1.0, 2.0, 3.0, 4.0],
            "independent_log_degree_min": [0.0, 1.0, 2.0, 3.0],
        }
    )
    null = deterministic_degree_permutation(toy, 0, 7)
    for feature in TOPOLOGY:
        if sorted(toy[feature]) != sorted(null[f"null0_{feature}"]):
            raise AssertionError("B45 permutation self-check failed")
    if np.array_equal(
        toy[TOPOLOGY].to_numpy(float),
        null[[f"null0_{feature}" for feature in TOPOLOGY]].to_numpy(float),
    ):
        raise AssertionError("B45 permutation self-check did not change the assignment")
    print("[BioAware B45 unit checks] PASS", flush=True)


def main() -> None:
    self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates",
        type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates",
        type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries",
        type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates",
        type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds",
        type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument(
        "--b42-dir",
        type=Path,
        default=ROOT / "data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 10000 or args.null_repeats != 3:
        raise ValueError("formal B45 requires 10,000 bootstraps and three nulls")
    for path in (
        args.internal_candidates,
        args.st_candidates,
        args.st_queries,
        args.kgmn_candidates,
        args.kgmn_seeds,
        args.b42_dir / "candidate_catalog_features.csv.gz",
        args.b42_dir / "report.json",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    universe, universe_report = build_universe(args)
    candidates = prepare_candidates(universe)
    topology = pd.read_csv(args.b42_dir / "candidate_catalog_features.csv.gz")
    topology_columns = ["query_id", "candidate_id", *SIGNATURE, *TOPOLOGY]
    candidates = candidates.merge(
        topology[topology_columns],
        on=["query_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if candidates[topology_columns[2:]].isna().any().any():
        raise RuntimeError("B45 topology join is incomplete")
    if len(candidates) != 6695 or candidates["query_id"].nunique() != 1738:
        raise RuntimeError("B45 B42 universe coverage changed")
    for repeat in range(args.null_repeats):
        candidates = deterministic_degree_permutation(candidates, repeat, args.seed)

    evaluated = candidates.loc[candidates["polarity"].eq("negative")].copy()
    if len(evaluated["query_id"].unique()) != 860:
        raise RuntimeError("B45 negative evaluation denominator changed")
    arms = {
        "spectral_only": ["spectral_score"],
        "degree_real": ["spectral_score", *TOPOLOGY],
        **{
            f"degree_permuted_r{repeat:02d}": [
                "spectral_score",
                *(f"null{repeat}_{feature}" for feature in TOPOLOGY),
            ]
            for repeat in range(args.null_repeats)
        },
    }

    outer_results: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}
    fold_reports: list[dict] = []
    gate_rows: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        fold_report = {
            "outer_domain": outer_domain,
            "outer_train_queries": int(outer_train["query_id"].nunique()),
            "outer_test_queries": int(outer_test["query_id"].nunique()),
            "arms": {},
        }
        for arm_index, (arm, features) in enumerate(arms.items()):
            inner_parts: list[pd.DataFrame] = []
            inner_fits: list[dict] = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                scored, fit_report = score_queries(
                    inner_train,
                    inner_test,
                    features,
                    stable_seed(args.seed, arm, outer_domain, inner_domain, inner_index),
                )
                inner_parts.append(scored)
                inner_fits.append({"inner_domain": inner_domain, **fit_report})
            inner_scores = pd.concat(inner_parts, ignore_index=True)
            selected, ledger = choose_gate(inner_scores)
            for row in ledger:
                gate_rows.append(
                    {"outer_domain": outer_domain, "arm": arm, **row}
                )
            outer_scored, outer_fit = score_queries(
                outer_train,
                outer_test,
                features,
                stable_seed(args.seed, arm, outer_domain, outer_index),
            )
            if selected["gate_name"] == "no_op":
                outer_final = no_op(outer_scored)
            else:
                outer_final = apply_gate(
                    outer_scored,
                    float(selected["margin"]),
                    float(selected["probability"]),
                )
            outer_final["arm"] = arm
            outer_final["outer_domain"] = outer_domain
            outer_results[arm].append(outer_final)
            fold_report["arms"][arm] = {
                "selected_gate": selected,
                "outer_fit": outer_fit,
                "inner_fits": inner_fits,
                "outer_result": summarize(outer_final),
            }
        fold_reports.append(fold_report)
        print(f"[B45 outer {outer_domain}] complete", flush=True)

    combined = {
        arm: pd.concat(parts, ignore_index=True).sort_values("query_id").reset_index(drop=True)
        for arm, parts in outer_results.items()
    }
    query_order = combined["degree_real"]["query_id"].astype(str).tolist()
    for arm, frame in combined.items():
        if len(frame) != 860 or frame["query_id"].astype(str).tolist() != query_order:
            raise RuntimeError(f"B45 {arm} OOF coverage/order mismatch")

    arm_reports: dict[str, dict] = {}
    for arm_index, (arm, frame) in enumerate(combined.items()):
        arm_reports[arm] = {
            **summarize(frame),
            "formula_cluster_bootstrap": cluster_bootstrap(
                frame["delta"], frame["truth_formula"], args.bootstrap_resamples,
                args.seed + 4500 + arm_index,
            ),
            "identity_cluster_bootstrap": cluster_bootstrap(
                frame["delta"], frame["truth_candidate_id"], args.bootstrap_resamples,
                args.seed + 4600 + arm_index,
            ),
            "by_source": {
                source: summarize(group)
                for source, group in frame.groupby("source", sort=True)
            },
            "eligible_action_queries": int(
                frame["coverage_neutral_candidate_count"].gt(1).sum()
            ),
            "recoverable_baseline_errors": int(
                (
                    ~frame["baseline_correct"].astype(bool)
                    & frame["truth_coverage_matches_baseline"].astype(bool)
                ).sum()
            ),
        }

    real = combined["degree_real"]
    contrasts: dict[str, dict] = {}
    for index, control in enumerate(
        ["spectral_only", *(f"degree_permuted_r{r:02d}" for r in range(args.null_repeats))]
    ):
        difference = (
            real["final_correct"].astype(int)
            - combined[control]["final_correct"].astype(int)
        )
        contrasts[f"degree_real_minus_{control}"] = {
            "mean_delta": float(difference.mean()),
            "formula_cluster_bootstrap": cluster_bootstrap(
                difference, real["truth_formula"], args.bootstrap_resamples,
                args.seed + 4700 + index,
            ),
            "identity_cluster_bootstrap": cluster_bootstrap(
                difference, real["truth_candidate_id"], args.bootstrap_resamples,
                args.seed + 4800 + index,
            ),
        }

    real_report = arm_reports["degree_real"]
    null_contrasts = [
        contrasts[f"degree_real_minus_degree_permuted_r{repeat:02d}"]
        for repeat in range(args.null_repeats)
    ]
    gates = {
        "coverage_neutral_queries_ge_400": bool(real_report["eligible_action_queries"] >= 400),
        "recoverable_errors_ge_100": bool(real_report["recoverable_baseline_errors"] >= 100),
        "overall_gain_ge_3pp": bool(real_report["delta_recall1"] >= 0.03),
        "formula_ci_low_positive": bool(
            real_report["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "identity_ci_low_positive": bool(
            real_report["identity_cluster_bootstrap"]["ci_low"] > 0
        ),
        "corrected_gt_2x_introduced": bool(
            real_report["corrected"] > 2 * real_report["introduced"]
        ),
        "every_source_nonnegative": bool(
            all(item["delta_recall1"] >= 0 for item in real_report["by_source"].values())
        ),
        "real_beats_every_degree_permutation_formula_ci": bool(
            all(item["formula_cluster_bootstrap"]["ci_low"] > 0 for item in null_contrasts)
        ),
    }
    pass_to_new_external = bool(all(gates.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "nested_loso_transitions.csv.gz"
    ledger_path = args.output_dir / "inner_gate_ledger.csv.gz"
    report_path = args.output_dir / "report.json"
    atomic_csv(transitions_path, pd.concat(combined.values(), ignore_index=True))
    atomic_csv(ledger_path, pd.DataFrame(gate_rows))
    report = {
        "status": "bioaware_b45_coverage_neutral_topology_complete",
        "formal": True,
        "protocol": (
            "development-only nested leave-source-out; held truth identities and "
            "formulas purged; training pairs and candidate actions require exact "
            "KEGG/Rhea membership-signature equality"
        ),
        "arms": arm_reports,
        "contrasts": contrasts,
        "folds": fold_reports,
        "gates": gates,
        "pass_to_new_external": pass_to_new_external,
        "decision": (
            "retain within-coverage static topology only if every gate passes; "
            "otherwise stop candidate-static topology and move to sample-context evidence"
        ),
        "contracts": {
            "B44_read": False,
            "catalogue_membership_changes_forbidden": True,
            "outcome_used_for_degree_permutation": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_report["provenance"],
            "b42_candidate_features": sha256(
                args.b42_dir / "candidate_catalog_features.csv.gz"
            ),
            "b42_report": sha256(args.b42_dir / "report.json"),
            "transitions": sha256(transitions_path),
            "inner_gate_ledger": sha256(ledger_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B45 can establish development-only within-coverage topology signal. "
            "It cannot recover B44, prove sample-context biology, or establish external gain."
        ),
    }
    atomic_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "degree_real": real_report,
        "contrasts": contrasts,
        "gates": gates,
        "pass_to_new_external": pass_to_new_external,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
