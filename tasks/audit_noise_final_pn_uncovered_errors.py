"""Materialise and stratify official errors outside the frozen P/N supervision.

The output is a fail-closed design ledger for expanding the noise action space.
It does not train a model and does not use P2b to define recoverability.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold, strict_rank  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--error-analysis", type=Path, default=ROOT / "data/validation/g8r_real_error_analysis")
    parser.add_argument("--r1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--fivepoint", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_fivepoint_headroom.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_uncovered_errors")
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    return parser.parse_args()


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    output = {}
    for column in columns:
        if column in frame:
            output[column] = int(strict_bool(frame[column].fillna(False), column).sum())
    return output


def main() -> None:
    args = arguments()
    signature_path = args.error_analysis / "query_error_signatures.csv.gz"
    analysis_report = args.error_analysis / "report.json"
    n_path = args.r1_dir / "corrective_teacher_actions.csv.gz"
    n_report = args.r1_dir / "report.json"
    p_path = args.c1_dir / "crossfit_examples.csv.gz"
    p_report = args.c1_dir / "decision.json"
    required = [
        args.graph, signature_path, analysis_report, n_path, n_report,
        p_path, p_report, args.fivepoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite uncovered-error ledger: {args.output_dir}")

    fivepoint = json.loads(args.fivepoint.read_text(encoding="utf-8"))
    if fivepoint.get("status") != "noise_final_pn_fivepoint_headroom_complete":
        raise RuntimeError("five-point audit is missing or malformed")
    graph = CandidateGraph(args.graph)
    official_rank = np.asarray([
        strict_rank(graph.official_molecule_scores(query)) for query in range(graph.n_queries)
    ], dtype=np.int16)
    official_errors = set(map(int, np.flatnonzero(official_rank != 1)))
    signature = pd.read_csv(signature_path)
    if signature["query_index"].duplicated().any() or len(signature) != graph.n_queries:
        raise RuntimeError("error signature ledger does not cover the graph one-to-one")
    signature = signature.sort_values("query_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(signature["query_index"].to_numpy(np.int64), np.arange(graph.n_queries)):
        raise RuntimeError("error signature query order/index mismatch")
    if set(map(int, signature.loc[~strict_bool(signature["dreams_correct"], "dreams_correct"), "query_index"])) != official_errors:
        raise RuntimeError("error signature official-error set does not reproduce graph ranking")
    for column in (
        "dreams_correct", "has_near_candidate", "positive_deficit", "negative_excess",
        "both_score_arms", "comparative_boundary_error", "shared_major_peak_screen",
        "neutral_loss_convergence_screen", "cross_condition_positive_screen",
        "raw_evidence_can_rescue", "rules_favor_positive", "rules_favor_wrong",
    ):
        if column in signature:
            signature[column] = strict_bool(signature[column].fillna(False), column)

    n_frame = pd.read_csv(n_path)
    n_queries = set(map(int, n_frame["query_index"]))
    if not n_queries <= official_errors:
        raise RuntimeError("N supervision contains a graph-baseline-correct query")
    p_frame = pd.read_csv(p_path)
    for column in ("query_index", "corrected", "baseline_rank", "teacher_rank"):
        if column not in p_frame:
            raise RuntimeError(f"C1 ledger missing {column}")
    p_corrected = strict_bool(p_frame["corrected"], "C1 corrected")
    p_instance_queries = set(map(int, p_frame.loc[p_corrected, "query_index"]))
    p_queries = p_instance_queries & official_errors
    union = n_queries | p_queries
    uncovered_set = official_errors - union
    if len(union) != int(fivepoint["p_n_union_recoverable_queries"]):
        raise RuntimeError("P/N union does not reproduce five-point audit")
    if len(uncovered_set) != int(fivepoint["remaining_official_errors_outside_supervision"]):
        raise RuntimeError("uncovered set does not reproduce five-point audit")

    p_summary = p_frame.groupby("query_index", sort=False).agg(
        c1_instances=("query_index", "size"),
        c1_baseline_wrong_instances=("baseline_rank", lambda values: int(np.sum(np.asarray(values) != 1))),
        c1_teacher_correct_instances=("teacher_rank", lambda values: int(np.sum(np.asarray(values) == 1))),
        c1_corrected_instances=("corrected", lambda values: int(np.sum(strict_bool(pd.Series(values), "C1 corrected group")))),
    ).reset_index()
    signature = signature.merge(p_summary, on="query_index", how="left", validate="one_to_one")
    for column in (
        "c1_instances", "c1_baseline_wrong_instances", "c1_teacher_correct_instances",
        "c1_corrected_instances",
    ):
        signature[column] = signature[column].fillna(0).astype(np.int32)
    signature["n_arm_recoverable"] = signature["query_index"].isin(n_queries)
    signature["p_arm_recoverable"] = signature["query_index"].isin(p_queries)
    signature["pn_recoverable"] = signature["query_index"].isin(union)
    signature["pn_uncovered"] = signature["query_index"].isin(uncovered_set)
    signature["formula_fold"] = [
        stable_fold(str(value), 5, args.formula_fold_seed) for value in signature["query_formula"]
    ]
    uncovered = signature.loc[signature["pn_uncovered"]].copy()
    expected_uncovered = int(fivepoint["remaining_official_errors_outside_supervision"])
    if len(uncovered) != expected_uncovered:
        raise RuntimeError(f"expected {expected_uncovered} uncovered official errors, got {len(uncovered)}")

    # Mutually exclusive design strata.  These are routing labels for the next
    # experiment, not causal error ground truth.
    strata = np.full(len(uncovered), "other_boundary", dtype=object)
    positive = uncovered["score_error_family"].astype(str).isin({
        "positive_deficit_only", "positive_deficit_and_negative_excess",
    }).to_numpy()
    c1_available = uncovered["c1_instances"].to_numpy(int) > 0
    negative = uncovered["score_error_family"].astype(str).isin({
        "negative_excess_only", "positive_deficit_and_negative_excess",
    }).to_numpy()
    strata[positive & ~c1_available] = "positive_deficit_no_support_pair"
    strata[positive & c1_available] = "positive_deficit_support_not_rescued"
    strata[negative & ~positive] = "negative_excess_action_not_rescued"
    uncovered["next_design_stratum"] = strata

    boolean_columns = [
        "positive_deficit", "negative_excess", "both_score_arms",
        "comparative_boundary_error", "shared_major_peak_screen",
        "neutral_loss_convergence_screen", "cross_condition_positive_screen",
        "raw_evidence_can_rescue", "rules_favor_positive", "rules_favor_wrong",
    ]
    family_table = (
        uncovered.groupby(["formula_fold", "next_design_stratum"], dropna=False)
        .agg(
            queries=("query_index", "size"),
            identities=("query_ik14", "nunique"),
            formulas=("query_formula", "nunique"),
            median_margin=("dreams_margin", "median"),
            near_fraction=("has_near_candidate", "mean"),
            raw_rescue_fraction=("raw_evidence_can_rescue", "mean"),
            rules_positive_fraction=("rules_favor_positive", "mean"),
            cross_condition_fraction=("cross_condition_positive_screen", "mean"),
        ).reset_index()
    )
    overall_strata = (
        uncovered.groupby("next_design_stratum", dropna=False)
        .agg(
            queries=("query_index", "size"), identities=("query_ik14", "nunique"),
            formulas=("query_formula", "nunique"), median_margin=("dreams_margin", "median"),
        ).reset_index().sort_values("queries", ascending=False)
    )
    report = {
        "status": "noise_final_pn_uncovered_error_audit_complete",
        "formal": True,
        "official_errors": len(official_errors),
        "n_arm_recoverable": len(n_queries),
        "p_arm_official_error_recoverable": len(p_queries),
        "p_n_union": len(union),
        "uncovered_errors": len(uncovered),
        "additional_unique_errors_required_for_five_points": int(
            fivepoint["required_net_corrections"] - len(union)
        ),
        "uncovered_score_error_families": uncovered["score_error_family"].value_counts().astype(int).to_dict(),
        "uncovered_next_design_strata": uncovered["next_design_stratum"].value_counts().astype(int).to_dict(),
        "uncovered_screens": counts(uncovered, boolean_columns),
        "uncovered_formula_folds": uncovered["formula_fold"].value_counts().sort_index().astype(int).to_dict(),
        "c1_support_among_uncovered": {
            "queries_with_any_c1_instance": int((uncovered["c1_instances"] > 0).sum()),
            "queries_without_c1_instance": int((uncovered["c1_instances"] == 0).sum()),
            "queries_with_teacher_correct_instance_but_not_strict_correction": int(
                ((uncovered["c1_teacher_correct_instances"] > 0) & (uncovered["c1_corrected_instances"] == 0)).sum()
            ),
        },
        "design_requirement": {
            "minimum_new_unique_recoverable_errors": int(fivepoint["required_net_corrections"] - len(union)),
            "recommended_headroom_buffer": 350,
            "rule": "new action families must report unique union gain over the frozen 922-query P/N set",
        },
        "contracts": {
            "P2b_used_to_define_recoverability": False,
            "outcome_labels_used_only_for_headroom_membership": True,
            "all_uncovered_errors_materialized": True,
            "formula_folds_frozen": True,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "signature_sha256": sha256_file(signature_path),
            "n_actions_sha256": sha256_file(n_path),
            "p_examples_sha256": sha256_file(p_path),
            "fivepoint_sha256": sha256_file(args.fivepoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Descriptive routing of uncovered official errors; no new action or model gain yet.",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pn_uncovered_", dir=args.output_dir.parent))
    try:
        uncovered.to_csv(staging / "uncovered_errors.csv.gz", index=False, compression="gzip")
        signature.loc[signature["pn_recoverable"]].to_csv(
            staging / "covered_errors.csv.gz", index=False, compression="gzip"
        )
        family_table.to_csv(staging / "formula_fold_strata.csv", index=False)
        overall_strata.to_csv(staging / "overall_strata.csv", index=False)
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
