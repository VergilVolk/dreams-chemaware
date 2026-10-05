#!/usr/bin/env python
"""Post-hoc, read-only root-cause audit of the consumed BioAware B44 result.

This script never refits a model or changes an evaluation decision.  It checks
the downloaded B44 ledgers against the sealed report and compares their action
mechanism with the frozen B42 development OOF predictions.  Results must be
written to a new directory outside the one-time B44 result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = [
    "independent_member_count",
    "independent_member_intersection",
    "independent_log_degree_mean",
    "independent_log_degree_min",
]


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
    suffix = ".csv.gz" if path.name.endswith(".csv.gz") else ".csv"
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression="gzip" if path.name.endswith(".gz") else None,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cluster_bootstrap(
    values: pd.Series,
    clusters: pd.Series,
    repeats: int,
    seed: int,
) -> dict:
    frame = pd.DataFrame(
        {"value": values.to_numpy(float), "cluster": clusters.astype(str).to_numpy()}
    )
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


def add_candidate_features(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_column: str,
    truth_column: str,
    baseline_column: str,
    proposal_column: str,
) -> pd.DataFrame:
    index = candidates.set_index(["query_id", candidate_column])
    output = queries.copy()
    for prefix, column in (
        ("truth", truth_column),
        ("baseline", baseline_column),
        ("proposal", proposal_column),
    ):
        keys = pd.MultiIndex.from_arrays([output["query_id"], output[column]])
        for feature in TOPOLOGY:
            values = index.reindex(keys)[feature]
            if values.isna().any():
                raise RuntimeError(f"missing {prefix} candidate feature: {feature}")
            output[f"{prefix}_{feature}"] = values.to_numpy(float)
    return output


def coverage_strata(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_column: str,
) -> pd.DataFrame:
    local = candidates.copy()
    local["mapped"] = local["independent_member_count"].astype(float).gt(0)
    grouped = local.groupby("query_id", sort=False).agg(
        candidate_rows=(candidate_column, "size"),
        mapped_candidates=("mapped", "sum"),
    )
    grouped["coverage_stratum"] = np.select(
        [
            grouped["mapped_candidates"].eq(0),
            grouped["mapped_candidates"].eq(grouped["candidate_rows"]),
        ],
        ["none_mapped", "all_mapped"],
        default="mixed",
    )
    return queries.merge(
        grouped[["mapped_candidates", "coverage_stratum"]],
        left_on="query_id",
        right_index=True,
        validate="one_to_one",
    )


def summarize_strata(frame: pd.DataFrame, repeats: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for (stratum, truth_mapped), group in frame.groupby(
        ["coverage_stratum", "truth_catalogue_member"], sort=True
    ):
        rows.append(
            {
                "coverage_stratum": str(stratum),
                "truth_catalogue_member": bool(truth_mapped),
                "queries": int(len(group)),
                "formulas": int(group["truth_formula"].nunique()),
                "baseline_recall1": float(group["baseline_correct"].mean()),
                "final_recall1": float(group["final_correct"].mean()),
                "delta_recall1": float(group["delta_top1"].mean()),
                "interventions": int(group["intervene"].sum()),
                "corrected": int(group["corrected"].sum()),
                "introduced": int(group["introduced"].sum()),
                "risk_net_lambda2": int(
                    group["corrected"].sum() - 2 * group["introduced"].sum()
                ),
                "formula_cluster_bootstrap": cluster_bootstrap(
                    group["delta_top1"], group["truth_formula"], repeats, seed + len(rows)
                ),
            }
        )
    return rows


def intervention_mechanisms(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    truth = output["truth_independent_member_count"].gt(0)
    baseline = output["baseline_independent_member_count"].gt(0)
    proposal = output["proposal_independent_member_count"].gt(0)
    output["mechanism"] = np.select(
        [
            output["intervene"] & truth & ~baseline & proposal,
            output["intervene"] & ~truth & ~baseline & proposal,
            output["intervene"] & truth & baseline & proposal,
            output["intervene"],
        ],
        [
            "coverage_truth_rescue",
            "coverage_false_promotion",
            "mapped_degree_competition",
            "other",
        ],
        default="no_intervention",
    )
    output["outcome"] = np.select(
        [output["corrected"], output["introduced"]],
        ["corrected", "introduced"],
        default="neutral",
    )
    return output


def mapped_rates(frame: pd.DataFrame) -> dict:
    mapped = frame["independent_member_count"].astype(float).gt(0)
    truth = frame["is_positive"].astype(bool)
    return {
        "truth": float(mapped[truth].mean()),
        "wrong": float(mapped[~truth].mean()),
        "truth_minus_wrong": float(mapped[truth].mean() - mapped[~truth].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b42-dir",
        type=Path,
        default=ROOT / "data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2",
    )
    parser.add_argument(
        "--b44-dir",
        type=Path,
        default=ROOT / "data/validation/bioaware_b44_massbank_once_2336677",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 10000:
        raise ValueError("formal audit requires at least 10,000 bootstrap resamples")

    b44_report_path = args.b44_dir / "report.json"
    b44_query_path = args.b44_dir / "per_query.csv.gz"
    b44_candidate_path = args.b44_dir / "candidate_scores.csv.gz"
    b42_transition_path = args.b42_dir / "cross_catalog_transitions.csv.gz"
    b42_candidate_path = args.b42_dir / "candidate_catalog_features.csv.gz"
    for path in (
        b44_report_path,
        b44_query_path,
        b44_candidate_path,
        b42_transition_path,
        b42_candidate_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    sealed = json.loads(b44_report_path.read_text(encoding="utf-8"))
    if sealed.get("status") != "bioaware_b44_massbank_one_time_evaluation_complete":
        raise RuntimeError("B44 report is not the consumed one-time evaluation")
    if sealed.get("pass_external_confirmation") is not False:
        raise RuntimeError("B44 external decision unexpectedly changed")
    if sha256(b44_query_path) != sealed["provenance"]["per_query"]:
        raise RuntimeError("B44 per-query SHA256 mismatch")
    if sha256(b44_candidate_path) != sealed["provenance"]["candidate_scores"]:
        raise RuntimeError("B44 candidate-score SHA256 mismatch")

    b44_queries = pd.read_csv(b44_query_path)
    b44_candidates = pd.read_csv(b44_candidate_path)
    if len(b44_queries) != 893 or b44_queries["query_id"].nunique() != 893:
        raise RuntimeError("B44 query coverage changed")
    if not b44_candidates.groupby("query_id")["is_positive"].sum().eq(1).all():
        raise RuntimeError("B44 does not have exactly one truth per query")
    counts = b44_candidates.groupby("query_id").size()
    if not b44_queries.set_index("query_id")["candidate_count"].eq(counts).all():
        raise RuntimeError("B44 query/candidate counts disagree")
    if int(b44_queries["delta_top1"].sum()) != int(
        b44_queries["corrected"].sum() - b44_queries["introduced"].sum()
    ):
        raise RuntimeError("B44 transition reconciliation failed")

    transitions = pd.read_csv(b42_transition_path)
    b42_queries = transitions.loc[transitions["arm"].eq("kegg_rhea_consensus")].copy()
    if len(b42_queries) != 860 or b42_queries["query_id"].nunique() != 860:
        raise RuntimeError("B42 consensus OOF coverage changed")
    b42_queries["fixed_intervene"] = (
        b42_queries["proposal_unique"].astype(bool)
        & b42_queries["proposed_candidate_id"].astype(str).ne(
            b42_queries["baseline_candidate_id"].astype(str)
        )
        & b42_queries["baseline_gap"].astype(float).le(0.05)
        & b42_queries["proposal_probability"].astype(float).ge(0.55)
    )
    b42_queries["fixed_corrected"] = (
        b42_queries["fixed_intervene"]
        & ~b42_queries["baseline_correct"].astype(bool)
        & b42_queries["proposed_candidate_id"].astype(str).eq(
            b42_queries["truth_candidate_id"].astype(str)
        )
    )
    b42_queries["fixed_introduced"] = (
        b42_queries["fixed_intervene"]
        & b42_queries["baseline_correct"].astype(bool)
        & b42_queries["proposed_candidate_id"].astype(str).ne(
            b42_queries["truth_candidate_id"].astype(str)
        )
    )
    if (
        int(b42_queries["fixed_intervene"].sum()),
        int(b42_queries["fixed_corrected"].sum()),
        int(b42_queries["fixed_introduced"].sum()),
    ) != (111, 65, 5):
        raise RuntimeError("B42 frozen deployment-gate replay failed")

    b42_candidates = pd.read_csv(b42_candidate_path)
    b42_candidates = b42_candidates.loc[
        b42_candidates["query_id"].isin(set(b42_queries["query_id"]))
    ].copy()
    if b42_candidates["query_id"].nunique() != 860:
        raise RuntimeError("B42 candidate feature coverage changed")

    b44_enriched = add_candidate_features(
        b44_queries,
        b44_candidates,
        "candidate_ik14",
        "truth_ik14",
        "baseline_candidate",
        "proposed_candidate",
    )
    b44_enriched = coverage_strata(b44_enriched, b44_candidates, "candidate_ik14")
    b44_enriched = intervention_mechanisms(b44_enriched)

    strata = summarize_strata(
        b44_enriched, args.bootstrap_resamples, args.seed + 4400
    )
    mechanism_rows = []
    for (polarity, mechanism), group in b44_enriched.groupby(
        ["ion_mode", "mechanism"], sort=True
    ):
        mechanism_rows.append(
            {
                "ion_mode": str(polarity),
                "mechanism": str(mechanism),
                "queries": int(len(group)),
                "corrected": int(group["corrected"].sum()),
                "introduced": int(group["introduced"].sum()),
                "net_top1": int(group["delta_top1"].sum()),
            }
        )

    b42_errors = int((~b42_queries["baseline_correct"].astype(bool)).sum())
    b42_right = int(b42_queries["baseline_correct"].astype(bool).sum())
    b44_errors = int((~b44_queries["baseline_correct"].astype(bool)).sum())
    b44_right = int(b44_queries["baseline_correct"].astype(bool).sum())
    b42_corrected = int(b42_queries["fixed_corrected"].sum())
    b42_introduced = int(b42_queries["fixed_introduced"].sum())
    b44_corrected = int(b44_queries["corrected"].sum())
    b44_introduced = int(b44_queries["introduced"].sum())

    b44_interventions = b44_enriched.loc[b44_enriched["intervene"]].copy()
    all_promote_higher_membership = bool(
        (
            b44_interventions["proposal_independent_member_count"]
            > b44_interventions["baseline_independent_member_count"]
        ).all()
    )
    corrected_probability = b44_interventions.loc[
        b44_interventions["corrected"], "proposal_probability"
    ]
    introduced_probability = b44_interventions.loc[
        b44_interventions["introduced"], "proposal_probability"
    ]

    result = {
        "status": "bioaware_b44_external_failure_root_cause_complete",
        "formal": False,
        "audit_type": "post-hoc read-only mechanism audit of a consumed external panel",
        "sealed_b44_decision_unchanged": True,
        "engineering_reconciliation": {
            "queries": int(len(b44_queries)),
            "candidate_rows": int(len(b44_candidates)),
            "one_truth_per_query": True,
            "candidate_count_reconciled": True,
            "corrected": b44_corrected,
            "introduced": b44_introduced,
            "delta_top1_count": int(b44_queries["delta_top1"].sum()),
        },
        "catalogue_mapping_shift": {
            "b42_consensus_oof": mapped_rates(b42_candidates),
            "b44_massbank": mapped_rates(b44_candidates),
        },
        "same_frozen_gate_transport": {
            "b42": {
                "queries": int(len(b42_queries)),
                "baseline_errors": b42_errors,
                "baseline_correct": b42_right,
                "interventions": int(b42_queries["fixed_intervene"].sum()),
                "corrected": b42_corrected,
                "introduced": b42_introduced,
                "delta_recall1": float((b42_corrected - b42_introduced) / len(b42_queries)),
                "error_recovery_fraction": float(b42_corrected / b42_errors),
                "harm_fraction_among_baseline_correct": float(b42_introduced / b42_right),
                "nonneutral_precision": float(b42_corrected / (b42_corrected + b42_introduced)),
            },
            "b44": {
                "queries": int(len(b44_queries)),
                "baseline_errors": b44_errors,
                "baseline_correct": b44_right,
                "interventions": int(b44_queries["intervene"].sum()),
                "corrected": b44_corrected,
                "introduced": b44_introduced,
                "delta_recall1": float(b44_queries["delta_top1"].mean()),
                "error_recovery_fraction": float(b44_corrected / b44_errors),
                "harm_fraction_among_baseline_correct": float(b44_introduced / b44_right),
                "nonneutral_precision": float(b44_corrected / (b44_corrected + b44_introduced)),
            },
        },
        "b44_coverage_strata": strata,
        "b44_intervention_mechanisms": mechanism_rows,
        "b44_gate_diagnostic": {
            "all_interventions_promote_higher_member_count": all_promote_higher_membership,
            "corrected_probability_median": float(corrected_probability.median()),
            "introduced_probability_median": float(introduced_probability.median()),
            "existing_confidence_is_anti_calibrated_for_external_harm": bool(
                introduced_probability.median() > corrected_probability.median()
            ),
        },
        "root_cause": (
            "B42 truth candidates were strongly enriched for KEGG/Rhea catalogue "
            "membership. B44 reverses that relation, especially in negative mode. "
            "The frozen expert therefore transports a catalogue-ascertainment prior, "
            "not query-specific reaction evidence."
        ),
        "decision": {
            "catalogue_v1_general_deployment": "stop",
            "retune_on_consumed_b44": False,
            "static_membership_as_positive_score": False,
            "catalogue_membership_allowed_use": "coverage/missingness mask only",
            "next_decisive_test": (
                "coverage-neutral, source-held-out ranking in which truth and wrong "
                "candidates share catalogue membership status; require real topology "
                "to beat formula-preserving topology permutations"
            ),
        },
        "provenance": {
            "b44_report": sha256(b44_report_path),
            "b44_per_query": sha256(b44_query_path),
            "b44_candidate_scores": sha256(b44_candidate_path),
            "b42_transitions": sha256(b42_transition_path),
            "b42_candidate_features": sha256(b42_candidate_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This post-hoc audit explains the consumed B44 failure. It does not "
            "create a new external performance estimate or validate sample-context BioAware."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    interventions_path = args.output_dir / "interventions.csv.gz"
    strata_path = args.output_dir / "coverage_strata.csv"
    report_path = args.output_dir / "report.json"
    atomic_csv(interventions_path, b44_interventions)
    atomic_csv(strata_path, pd.DataFrame(strata))
    result["provenance"]["interventions"] = sha256(interventions_path)
    result["provenance"]["coverage_strata"] = sha256(strata_path)
    atomic_json(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
