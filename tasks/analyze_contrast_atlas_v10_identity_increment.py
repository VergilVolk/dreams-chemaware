#!/usr/bin/env python3
"""Measure decision and identity increment on the frozen Atlas v10 axis.

This is a post-scoring analysis.  It opens library identities only after all
method-specific complete-block decisions have been written.  It does not use
response effects, phenotype labels, registered biological relations, or WSE
edges to alter any method's decision.

The primary identity utility is +1 for a correct answer, -2 for a wrong answer,
and 0 for abstention.  Every method is evaluated on the same truth-eligible
query-block denominator.  Raw edge counts and answer coverage are descriptive
and cannot establish an increment by themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FULL_KEY = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
METHOD_ORDER = (
    "weighted_spectral_entropy",
    "official_dreams",
    "noise_v1",
    "p2b_noise_v1_frozen",
    "noise_msg_fusion_stack_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
    "full23_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_v10_complete_axis_full23_run_2433291"
        ),
    )
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_model_independent_identity_axis_v6_"
            "msv000080918_20261031"
        ),
    )
    parser.add_argument(
        "--library-evidence",
        type=Path,
        default=Path("tasks/contrast_atlas_v10_entity_library_evidence.tsv.gz"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_method_id(method_dir: Path) -> str:
    report = json.loads((method_dir / "report.json").read_text(encoding="utf-8"))
    method = report.get("method", {})
    method_id = method.get("method_id") if isinstance(method, dict) else None
    if not method_id:
        raise RuntimeError(f"method_id missing from {method_dir / 'report.json'}")
    return str(method_id)


def discover_methods(package_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for report_path in sorted(package_root.glob("*/report.json")):
        method_dir = report_path.parent
        method_id = read_method_id(method_dir)
        if method_id in found:
            raise RuntimeError(f"duplicate method package: {method_id}")
        required = (
            "method_pair_scores.tsv.gz",
            "method_block_decisions.tsv.gz",
            "reciprocal_edges.tsv.gz",
        )
        missing = [name for name in required if not (method_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{method_id}: missing {missing}")
        found[method_id] = method_dir
    missing_methods = [method for method in METHOD_ORDER if method not in found]
    if missing_methods:
        raise RuntimeError(f"missing method packages: {missing_methods}")
    return {method: found[method] for method in METHOD_ORDER}


def bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def load_exact_identity(path: Path) -> pd.DataFrame:
    evidence = pd.read_csv(path, sep="\t", low_memory=False, dtype={
        "study_id": str, "entity_id": str, "library_inchikey": str,
    })
    supported = bool_column(evidence["library_supported"])
    full = evidence["library_inchikey"].fillna("").astype(str).str.match(FULL_KEY)
    exact = evidence.loc[
        supported & full,
        ["study_id", "entity_id", "library_inchikey"],
    ].copy()
    conflicts = (
        exact.groupby(["study_id", "entity_id"])["library_inchikey"]
        .nunique()
        .loc[lambda values: values > 1]
    )
    if len(conflicts):
        raise RuntimeError(
            f"{len(conflicts)} entities have conflicting exact library identities"
        )
    return exact.drop_duplicates(["study_id", "entity_id"], keep="first")


def add_identity_keys(
    pairs: pd.DataFrame, exact: pd.DataFrame
) -> pd.DataFrame:
    query_keys = exact.rename(columns={
        "study_id": "query_study_id",
        "entity_id": "query_entity_id",
        "library_inchikey": "query_inchikey",
    })
    candidate_keys = exact.rename(columns={
        "study_id": "candidate_study_id",
        "entity_id": "candidate_entity_id",
        "library_inchikey": "candidate_inchikey",
    })
    output = pairs.merge(
        query_keys,
        on=["query_study_id", "query_entity_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        candidate_keys,
        on=["candidate_study_id", "candidate_entity_id"],
        how="left",
        validate="many_to_one",
    )
    output["is_exact_truth_candidate"] = (
        output["query_inchikey"].notna()
        & output["candidate_inchikey"].notna()
        & output["query_inchikey"].eq(output["candidate_inchikey"])
    )
    return output


def truth_blocks(pairs: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    grouped = pairs.groupby("query_block_id", sort=False).agg(
        query_inchikey=("query_inchikey", "first"),
        exact_truth_candidates=("is_exact_truth_candidate", "sum"),
    )
    eligible = grouped.loc[
        grouped["query_inchikey"].notna()
        & grouped["exact_truth_candidates"].gt(0)
    ].reset_index()
    eligible = eligible.merge(
        blocks[[
            "query_block_id", "query_block_index", "query_study_id",
            "query_entity_id", "candidate_study_id", "eligible_candidates",
        ]],
        on="query_block_id",
        how="left",
        validate="one_to_one",
    )
    eligible["source_dyad"] = (
        eligible["query_study_id"].astype(str)
        + "||"
        + eligible["candidate_study_id"].astype(str)
    )
    return eligible


def edge_key(frame: pd.DataFrame) -> pd.Series:
    left = (
        frame["left_study_id"].astype(str)
        + "::"
        + frame["left_entity_id"].astype(str)
    )
    right = (
        frame["right_study_id"].astype(str)
        + "::"
        + frame["right_entity_id"].astype(str)
    )
    return pd.Series(
        np.where(left <= right, left + "||" + right, right + "||" + left),
        index=frame.index,
    )


def reciprocal_identity_ppv(edges: pd.DataFrame, exact: pd.DataFrame) -> tuple[int, int]:
    left = exact.rename(columns={
        "study_id": "left_study_id", "entity_id": "left_entity_id",
        "library_inchikey": "left_inchikey",
    })
    right = exact.rename(columns={
        "study_id": "right_study_id", "entity_id": "right_entity_id",
        "library_inchikey": "right_inchikey",
    })
    joined = edges.merge(
        left, on=["left_study_id", "left_entity_id"], how="left", validate="many_to_one"
    ).merge(
        right, on=["right_study_id", "right_entity_id"], how="left", validate="many_to_one"
    )
    evaluable = joined["left_inchikey"].notna() & joined["right_inchikey"].notna()
    correct = evaluable & joined["left_inchikey"].eq(joined["right_inchikey"])
    return int(correct.sum()), int(evaluable.sum())


def method_truth_outcomes(
    method: str,
    method_dir: Path,
    pairs: pd.DataFrame,
    truth: pd.DataFrame,
    exact: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(
        method_dir / "method_block_decisions.tsv.gz",
        sep="\t",
        dtype={
            "query_block_id": str, "selected_candidate_entity_id": str,
        },
    )
    decisions["answered"] = bool_column(decisions["answered"])
    selected = decisions[[
        "query_block_id", "answered", "decision_status",
        "selected_candidate_entity_id", "selected_raw_score",
    ]].copy()
    outcomes = truth.merge(selected, on="query_block_id", how="left", validate="one_to_one")
    outcomes["answered"] = outcomes["answered"].fillna(False).astype(bool)
    selected_keys = exact.rename(columns={
        "study_id": "candidate_study_id",
        "entity_id": "selected_candidate_entity_id",
        "library_inchikey": "selected_inchikey",
    })
    outcomes = outcomes.merge(
        selected_keys,
        on=["candidate_study_id", "selected_candidate_entity_id"],
        how="left",
        validate="many_to_one",
    )
    outcomes["outcome"] = np.select(
        [
            ~outcomes["answered"],
            outcomes["selected_inchikey"].eq(outcomes["query_inchikey"]),
        ],
        ["abstain", "correct"],
        default="wrong",
    )
    outcomes["utility"] = outcomes["outcome"].map(
        {"correct": 1.0, "wrong": -2.0, "abstain": 0.0}
    )
    outcomes["method"] = method

    scores = pd.read_csv(
        method_dir / "method_pair_scores.tsv.gz",
        sep="\t",
        usecols=["directed_pair_id", "query_block_id", "raw_score", "score_available"],
        dtype={"directed_pair_id": str, "query_block_id": str},
    )
    scores["score_available"] = bool_column(scores["score_available"])
    ranked = pairs[[
        "directed_pair_id", "query_block_id", "is_exact_truth_candidate",
    ]].merge(scores, on=["directed_pair_id", "query_block_id"], validate="one_to_one")
    truth_ids = set(truth["query_block_id"].astype(str))
    ranked = ranked.loc[ranked["query_block_id"].isin(truth_ids)].copy()
    complete = ranked.groupby("query_block_id")["score_available"].all()
    ranked["average_rank"] = ranked.groupby("query_block_id")["raw_score"].rank(
        ascending=False, method="average", na_option="bottom"
    )
    best_truth_rank = (
        ranked.loc[ranked["is_exact_truth_candidate"]]
        .groupby("query_block_id")["average_rank"]
        .min()
    )
    outcomes["truth_rank"] = outcomes["query_block_id"].map(best_truth_rank)
    outcomes["complete_pair_scores"] = outcomes["query_block_id"].map(complete).fillna(False)
    outcomes["reciprocal_rank"] = np.where(
        outcomes["complete_pair_scores"] & outcomes["truth_rank"].gt(0),
        1.0 / outcomes["truth_rank"],
        0.0,
    )

    counts = outcomes["outcome"].value_counts()
    correct = int(counts.get("correct", 0))
    wrong = int(counts.get("wrong", 0))
    abstain = int(counts.get("abstain", 0))
    denominator = len(outcomes)
    answered = correct + wrong

    edges = pd.read_csv(
        method_dir / "reciprocal_edges.tsv.gz",
        sep="\t",
        dtype={
            "left_study_id": str, "left_entity_id": str,
            "right_study_id": str, "right_entity_id": str,
        },
    )
    edges["edge_key"] = edge_key(edges)
    exact_edges, evaluable_edges = reciprocal_identity_ppv(edges, exact)
    metrics: dict[str, object] = {
        "method": method,
        "truth_blocks": denominator,
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "unconditional_recall_at_1": correct / denominator if denominator else np.nan,
        "conditional_accuracy": correct / answered if answered else np.nan,
        "unconditional_mrr": float(outcomes["reciprocal_rank"].mean()),
        "micro_utility": float(outcomes["utility"].mean()),
        "dyad_macro_utility": float(
            outcomes.groupby("source_dyad")["utility"].mean().mean()
        ),
        "reciprocal_edges": len(edges),
        "known_known_reciprocal_edges": evaluable_edges,
        "exact_key_reciprocal_edges": exact_edges,
        "known_known_reciprocal_exact_key_ppv": (
            exact_edges / evaluable_edges if evaluable_edges else np.nan
        ),
    }
    return outcomes, metrics, decisions, edges


def bootstrap_full23_deltas(
    outcomes: dict[str, pd.DataFrame], replicates: int, seed: int
) -> pd.DataFrame:
    target = "full23_v1"
    target_frame = outcomes[target][["query_block_id", "source_dyad", "utility"]].rename(
        columns={"utility": "target_utility"}
    )
    comparisons: dict[str, pd.Series] = {}
    leave_one_min: dict[str, float] = {}
    sources_by_block = outcomes[target].set_index("query_block_id")[[
        "query_study_id", "candidate_study_id"
    ]]
    for baseline, frame in outcomes.items():
        if baseline == target:
            continue
        joined = target_frame.merge(
            frame[["query_block_id", "utility"]].rename(columns={"utility": "baseline_utility"}),
            on="query_block_id",
            validate="one_to_one",
        )
        joined["delta"] = joined["target_utility"] - joined["baseline_utility"]
        dyad = joined.groupby("source_dyad")["delta"].mean().sort_index()
        comparisons[baseline] = dyad

        merged = joined.join(sources_by_block, on="query_block_id")
        all_sources = sorted(
            set(merged["query_study_id"].astype(str))
            | set(merged["candidate_study_id"].astype(str))
        )
        values = []
        for source in all_sources:
            keep = (
                merged["query_study_id"].astype(str).ne(source)
                & merged["candidate_study_id"].astype(str).ne(source)
            )
            reduced = merged.loc[keep].groupby("source_dyad")["delta"].mean()
            if len(reduced):
                values.append(float(reduced.mean()))
        leave_one_min[baseline] = min(values) if values else np.nan

    common_dyads = sorted(set.intersection(*(set(values.index) for values in comparisons.values())))
    if not common_dyads:
        raise RuntimeError("no common source dyads for Full23 comparisons")
    names = list(comparisons)
    matrix = np.column_stack([
        comparisons[name].reindex(common_dyads).to_numpy(float) for name in names
    ])
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, len(names)), dtype=float)
    for start in range(0, replicates, 500):
        stop = min(start + 500, replicates)
        index = rng.integers(0, len(common_dyads), size=(stop - start, len(common_dyads)))
        boot[start:stop] = matrix[index].mean(axis=1)
    max_abs_error = np.max(np.abs(boot - observed[None, :]), axis=1)
    simultaneous_radius = float(np.quantile(max_abs_error, 0.95))
    rows = []
    for column, baseline in enumerate(names):
        rows.append({
            "target_method": target,
            "baseline_method": baseline,
            "source_dyads": len(common_dyads),
            "dyad_macro_utility_delta": float(observed[column]),
            "bootstrap_ci95_low": float(np.quantile(boot[:, column], 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boot[:, column], 0.975)),
            "simultaneous_ci95_low": float(observed[column] - simultaneous_radius),
            "simultaneous_ci95_high": float(observed[column] + simultaneous_radius),
            "leave_one_source_min_delta": leave_one_min[baseline],
            "bootstrap_replicates": replicates,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    axis_dir = args.axis_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    required_inputs = {
        "method packages": run_dir / "method_packages",
        "frozen candidate pairs": axis_dir / "directed_complete_candidate_pairs.tsv.gz",
        "frozen query blocks": axis_dir / "query_target_source_blocks.tsv.gz",
        "hidden exact-identity evidence": args.library_evidence.resolve(),
    }
    missing = [f"{label}: {path}" for label, path in required_inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs before analysis:\n" + "\n".join(missing))
    out_dir.mkdir(parents=True)

    methods = discover_methods(run_dir / "method_packages")
    pairs = pd.read_csv(
        axis_dir / "directed_complete_candidate_pairs.tsv.gz",
        sep="\t",
        low_memory=False,
        dtype={
            "directed_pair_id": str, "query_block_id": str,
            "query_study_id": str, "query_entity_id": str,
            "candidate_study_id": str, "candidate_entity_id": str,
        },
    )
    blocks = pd.read_csv(
        axis_dir / "query_target_source_blocks.tsv.gz",
        sep="\t",
        low_memory=False,
        dtype={
            "query_block_id": str, "query_study_id": str,
            "query_entity_id": str, "candidate_study_id": str,
        },
    )
    exact = load_exact_identity(args.library_evidence.resolve())
    pairs = add_identity_keys(pairs, exact)
    truth = truth_blocks(pairs, blocks)
    if not len(truth):
        raise RuntimeError("no truth-eligible blocks on the frozen axis")

    outcomes_by_method: dict[str, pd.DataFrame] = {}
    decisions_by_method: dict[str, pd.DataFrame] = {}
    edges_by_method: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, object]] = []
    pair_ids: np.ndarray | None = None
    common_available: np.ndarray | None = None
    coverage_rows: list[dict[str, object]] = []

    for method, method_dir in methods.items():
        pair_frame = pd.read_csv(
            method_dir / "method_pair_scores.tsv.gz",
            sep="\t",
            usecols=["directed_pair_id", "score_available"],
            dtype={"directed_pair_id": str},
        )
        current_ids = pair_frame["directed_pair_id"].to_numpy(str)
        available = bool_column(pair_frame["score_available"]).to_numpy(bool)
        if pair_ids is None:
            pair_ids = current_ids
            common_available = available.copy()
        else:
            if not np.array_equal(pair_ids, current_ids):
                raise RuntimeError(f"pair order differs for {method}")
            common_available &= available

        outcomes, metrics, decisions, edges = method_truth_outcomes(
            method, method_dir, pairs, truth, exact
        )
        outcomes_by_method[method] = outcomes
        decisions_by_method[method] = decisions
        edges_by_method[method] = edges
        metrics_rows.append(metrics)
        coverage_rows.append({
            "method": method,
            "directed_pairs": len(available),
            "available_pair_scores": int(available.sum()),
            "available_pair_fraction": float(available.mean()),
            "nonempty_blocks": len(decisions),
            "answered_blocks": int(decisions["answered"].sum()),
            "answered_fraction_nonempty": float(decisions["answered"].mean()),
            "reciprocal_edges": len(edges),
        })

    assert common_available is not None
    pairs["common_available"] = common_available
    common_block = pairs.groupby("query_block_id")["common_available"].all()
    common_block_ids = set(common_block.loc[common_block].index.astype(str))
    for row in coverage_rows:
        method = str(row["method"])
        decisions = decisions_by_method[method]
        common = decisions.loc[decisions["query_block_id"].isin(common_block_ids)]
        row["common_support_nonempty_blocks"] = len(common)
        row["answered_common_support_blocks"] = int(common["answered"].sum())
        row["answered_fraction_common_support"] = (
            float(common["answered"].mean()) if len(common) else np.nan
        )

    baseline = "weighted_spectral_entropy"
    baseline_decisions = decisions_by_method[baseline][[
        "query_block_id", "answered", "selected_candidate_entity_id"
    ]].rename(columns={
        "answered": "baseline_answered",
        "selected_candidate_entity_id": "baseline_selected_candidate_entity_id",
    })
    block_transition_rows = []
    for method, decisions in decisions_by_method.items():
        if method == baseline:
            continue
        joined = baseline_decisions.merge(
            decisions[["query_block_id", "answered", "selected_candidate_entity_id"]],
            on="query_block_id", validate="one_to_one"
        )
        state = np.select(
            [
                ~joined["baseline_answered"] & ~joined["answered"],
                ~joined["baseline_answered"] & joined["answered"],
                joined["baseline_answered"] & ~joined["answered"],
                joined["baseline_answered"] & joined["answered"]
                & joined["baseline_selected_candidate_entity_id"].eq(
                    joined["selected_candidate_entity_id"]
                ),
            ],
            ["both_abstain", "gained_answer", "lost_answer", "retained_same"],
            default="changed_answer",
        )
        for label, count in pd.Series(state).value_counts().items():
            block_transition_rows.append({
                "baseline_method": baseline, "method": method,
                "transition": label, "blocks": int(count),
            })

    baseline_edges = set(edges_by_method[baseline]["edge_key"].astype(str))
    edge_transition_rows = []
    for method, edges in edges_by_method.items():
        if method == baseline:
            continue
        method_edges = set(edges["edge_key"].astype(str))
        edge_transition_rows.append({
            "baseline_method": baseline,
            "method": method,
            "retained_edges": len(baseline_edges & method_edges),
            "gained_edges": len(method_edges - baseline_edges),
            "lost_edges": len(baseline_edges - method_edges),
        })

    identity_transition_rows = []
    full = outcomes_by_method["full23_v1"][["query_block_id", "outcome"]].rename(
        columns={"outcome": "full23_outcome"}
    )
    for method, outcomes in outcomes_by_method.items():
        if method == "full23_v1":
            continue
        joined = outcomes[["query_block_id", "outcome"]].merge(
            full, on="query_block_id", validate="one_to_one"
        )
        counts = joined.groupby(["outcome", "full23_outcome"]).size()
        for (before, after), count in counts.items():
            identity_transition_rows.append({
                "baseline_method": method,
                "full23_method": "full23_v1",
                "baseline_outcome": before,
                "full23_outcome": after,
                "truth_blocks": int(count),
            })

    deltas = bootstrap_full23_deltas(
        outcomes_by_method, args.bootstrap_replicates, args.seed
    )

    compression = {"method": "gzip", "mtime": 0}
    pd.DataFrame(coverage_rows).to_csv(
        out_dir / "coverage_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(metrics_rows).to_csv(
        out_dir / "identity_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(block_transition_rows).to_csv(
        out_dir / "block_transitions_vs_wse.tsv", sep="\t", index=False
    )
    pd.DataFrame(edge_transition_rows).to_csv(
        out_dir / "reciprocal_edge_transitions_vs_wse.tsv", sep="\t", index=False
    )
    pd.DataFrame(identity_transition_rows).to_csv(
        out_dir / "full23_identity_outcome_transitions.tsv", sep="\t", index=False
    )
    deltas.to_csv(
        out_dir / "full23_identity_utility_deltas.tsv", sep="\t", index=False
    )
    truth.to_csv(
        out_dir / "truth_eligible_blocks.tsv.gz",
        sep="\t", index=False, compression=compression,
    )
    pd.concat(outcomes_by_method.values(), ignore_index=True).to_csv(
        out_dir / "method_truth_block_outcomes.tsv.gz",
        sep="\t", index=False, compression=compression,
    )

    full_delta = deltas.set_index("baseline_method")
    pass_identity = bool(
        len(full_delta)
        and (full_delta["simultaneous_ci95_low"] > 0).all()
        and (full_delta["leave_one_source_min_delta"] > 0).all()
    )
    report = {
        "schema": "contrast_atlas_v10_identity_increment_v1",
        "status": "DEVELOPMENT_CONSUMED_IDENTITY_INCREMENT_ANALYZED",
        "data_role": "development_consumed",
        "denominators": {
            "directed_candidate_pairs": int(len(pairs)),
            "query_target_source_blocks": int(len(blocks)),
            "nonempty_blocks": int((blocks["eligible_candidates"] > 0).sum()),
            "truth_eligible_blocks": int(len(truth)),
            "truth_source_dyads": int(truth["source_dyad"].nunique()),
        },
        "primary_utility": {
            "correct": 1,
            "wrong": -2,
            "abstain": 0,
            "aggregation": "source_dyad_macro",
            "full23_passes_identity_increment_gate": pass_identity,
            "gate": (
                "Full23 simultaneous 95% CI lower bound > 0 versus every other method "
                "and every leave-one-source minimum delta > 0"
            ),
        },
        "claim_boundary": [
            "Coverage, answer count, and reciprocal-edge count are not accuracy endpoints.",
            "Library-supported full InChIKeys are hidden development truth, not MSI level-1 identities.",
            "This analysis evaluates same-observed-precursor 10-ppm identity hypotheses only.",
            "It does not evaluate cross-mass chemical relations or biological response retrieval.",
            "No external or sealed biological validation is claimed.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
