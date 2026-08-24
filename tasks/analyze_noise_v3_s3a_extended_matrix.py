"""Decision analysis for the preregistered Noise-v3 S3A action matrix.

S3A is an intervention headroom experiment, not a trained model.  This
analyser therefore reports both sides of every action: official errors that
are corrected and official correct queries that are newly broken.  It also
materialises query-level transitions and observed rule evidence so that a
large average gain cannot hide a chemically concentrated failure mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_candidate_gradient import (  # noqa: E402
    cluster_ci, query_candidate_block, strict_metrics,
)
from build_g8r_real_error_atlas import (  # noqa: E402
    Cache, candidate_grade_name, rule_comparison, sha256_file,
)


DEFAULT_S3A = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix"
DEFAULT_CACHE = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_RULES = ROOT / "data/validation/g8r_chemaware_g0_rule_cache.npz"
DEFAULT_S1C = ROOT / "data/validation/g8r_noise_v3_s1c_topk_matrix"
DEFAULT_S2 = ROOT / "data/validation/g8r_noise_v3_s2_sequential"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3a-dir", type=Path, default=DEFAULT_S3A)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rule-cache", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--s1c-dir", type=Path, default=DEFAULT_S1C)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def baseline_table(cache: Cache) -> pd.DataFrame:
    score_column = cache.feature_names.index("dreams_similarity")
    rows: list[dict] = []
    for query in range(cache.n_queries):
        pair_scores, candidate_rows, ptr = query_candidate_block(
            cache, query, score_column,
        )
        rank, _, margin = strict_metrics(pair_scores, ptr)
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        winner_local = int(np.argmax(molecule_scores))
        winner_pair_local = int(
            ptr[winner_local]
            + np.argmax(pair_scores[ptr[winner_local]:ptr[winner_local + 1]])
        )
        molecule_global = int(cache.query_ptr[query]) + winner_local
        rows.append({
            "query_index": query,
            "baseline_rank_rebuilt": int(rank),
            "baseline_margin_rebuilt": float(margin),
            "baseline_winner_pair_row": int(candidate_rows[winner_pair_local]),
            "baseline_winner_ik14": str(cache.molecule_ik14[molecule_global]),
            "baseline_winner_formula": str(cache.molecule_formula[molecule_global]),
            "baseline_winner_mces_grade": candidate_grade_name(
                int(cache.molecule_mces_grade[molecule_global])
            ),
        })
    return pd.DataFrame(rows)


def corrected_queries(frame: pd.DataFrame) -> set[int]:
    return set(map(int, frame.loc[
        frame["baseline_rank"].gt(1) & frame["target_rank"].eq(1),
        "query_index",
    ]))


def introduced_queries(frame: pd.DataFrame) -> set[int]:
    return set(map(int, frame.loc[
        frame["baseline_rank"].eq(1) & frame["target_rank"].gt(1),
        "query_index",
    ]))


def action_name(selector: str, attenuation: float, step: int) -> str:
    return f"{selector}|a={float(attenuation):.2f}|step={int(step)}"


def load_rule_lookup(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as body:
        rows = np.asarray(body["hdf5_row"], dtype=np.int64)
        packed = np.asarray(body["packed_rule_hits"], dtype=np.uint8)
        n_rules = int(np.asarray(body["n_rules"]).reshape(-1)[0])
        categories = np.asarray(body["rule_category"], dtype=object).astype(str)
        libraries = np.asarray(body["rule_library"], dtype=object).astype(str)
    if len(rows) != len(packed) or len(categories) != n_rules:
        raise RuntimeError("rule cache arrays are not aligned")
    vectors = np.unpackbits(packed, axis=1, bitorder="little")[:, :n_rules].astype(bool)
    return {int(row): vectors[index] for index, row in enumerate(rows)}, categories, libraries


def add_rule_evidence(
    transitions: pd.DataFrame,
    lookup: dict[int, np.ndarray],
    categories: np.ndarray,
    libraries: np.ndarray,
) -> pd.DataFrame:
    if transitions.empty:
        return transitions
    output: list[dict] = []
    for row in transitions.itertuples(index=False):
        query_vector = lookup.get(int(row.query_row))
        baseline_vector = lookup.get(int(row.baseline_winner_pair_row))
        target_vector = lookup.get(int(row.winner_pair_row))
        evidence: dict[str, object] = {}
        for prefix, candidate in (("baseline", baseline_vector), ("target", target_vector)):
            if query_vector is None or candidate is None:
                evidence[f"{prefix}_rule_available"] = False
                continue
            evidence[f"{prefix}_rule_available"] = True
            for key, value in rule_comparison(query_vector, candidate, categories).items():
                evidence[f"{prefix}_{key}"] = value
            for library in ("core", "massbank"):
                mask = libraries == library
                left = query_vector[mask]
                right = candidate[mask]
                union = int(np.logical_or(left, right).sum())
                evidence[f"{prefix}_rule_jaccard_{library}"] = (
                    float(np.logical_and(left, right).sum() / union) if union else np.nan
                )
        output.append(evidence)
    evidence_frame = pd.DataFrame(output, index=transitions.index)
    return pd.concat([transitions, evidence_frame], axis=1)


def top_counts(frame: pd.DataFrame, columns: list[str], n: int = 30) -> list[dict]:
    if frame.empty:
        return []
    counts = frame.groupby(columns, dropna=False).size().sort_values(ascending=False).head(n)
    records: list[dict] = []
    for key, value in counts.items():
        values = key if isinstance(key, tuple) else (key,)
        records.append({
            **{column: str(item) for column, item in zip(columns, values)},
            "queries": int(value),
        })
    return records


def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or not frame[column].notna().any():
        return None
    return float(frame[column].mean())


def main() -> None:
    args = parse_args()
    paired_path = args.s3a_dir / "paired_interventions.csv.gz"
    report_path = args.s3a_dir / "report.json"
    validation_path = args.s3a_dir / "matrix_validation.json"
    for path in (
        paired_path, report_path, validation_path, args.cache, args.rule_cache,
        args.s1c_dir / "paired_interventions.csv.gz",
        args.s2_dir / "paired_interventions.csv.gz",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "noise_v3_s3a_matrix_validation_passed":
        raise RuntimeError("S3A matrix validation has not passed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path)
    cache = Cache(args.cache)
    baseline = baseline_table(cache)
    if len(baseline) != int(report["queries"]):
        raise RuntimeError("S3A query count and locked baseline differ")
    paired = paired.merge(baseline, on="query_index", validate="many_to_one")
    disagreements = int((
        paired["baseline_rank"].astype(int)
        != paired["baseline_rank_rebuilt"].astype(int)
    ).sum())
    if disagreements:
        raise RuntimeError(f"stored/rebuilt baseline rank disagreements: {disagreements}")
    paired["winner_mces_grade_name"] = paired["winner_mces_grade"].map(
        lambda value: candidate_grade_name(int(value))
    )
    paired["winner_same_formula"] = (
        paired["query_formula"].astype(str) == paired["winner_formula"].astype(str)
    )
    paired["transition"] = np.select(
        [paired["corrected"], paired["introduced"], paired["baseline_rank"].gt(1)],
        ["corrected", "introduced", "persistent_wrong"],
        default="protected_correct",
    )

    s1c = pd.read_csv(args.s1c_dir / "paired_interventions.csv.gz")
    s2 = pd.read_csv(args.s2_dir / "paired_interventions.csv.gz")
    prior = corrected_queries(s1c) | corrected_queries(s2)
    base_wrong = set(map(int, baseline.loc[
        baseline["baseline_rank_rebuilt"].gt(1), "query_index",
    ]))
    near_wrong = set(map(int, cache.query_has_near.nonzero()[0])) & base_wrong

    action_results: dict[str, dict] = {}
    action_corrected: dict[str, set[int]] = {}
    action_introduced: dict[str, set[int]] = {}
    for position, (key, group) in enumerate(paired.groupby(
        ["selector", "attenuation", "step"], sort=True,
    )):
        selector, attenuation, step = key
        name = action_name(str(selector), float(attenuation), int(step))
        corrected = corrected_queries(group)
        introduced = introduced_queries(group)
        wrong = group.loc[group["baseline_rank"].gt(1)].copy()
        action_corrected[name] = corrected
        action_introduced[name] = introduced
        identity_ci = cluster_ci(
            wrong, "target_minus_random_top1", "query_ik14", args.bootstrap,
            args.seed + position,
        )
        formula_ci = cluster_ci(
            wrong, "target_minus_random_top1", "query_formula", args.bootstrap,
            args.seed + 10_000 + position,
        )
        action_results[name] = {
            "queries": int(len(group)),
            "identities": int(group["query_ik14"].nunique()),
            "formulas": int(group["query_formula"].nunique()),
            "corrected": len(corrected),
            "introduced": len(introduced),
            "net": len(corrected) - len(introduced),
            "correction_precision": (
                float(len(corrected) / (len(corrected) + len(introduced)))
                if corrected or introduced else None
            ),
            "unique_corrections_beyond_s1c_s2": len(corrected - prior),
            "near_corrected": len(corrected & near_wrong),
            "identity_target_minus_random_top1_95ci": identity_ci,
            "formula_target_minus_random_top1_95ci": formula_ci,
            "specificity_gate": bool(
                identity_ci is not None and formula_ci is not None
                and identity_ci[0] > 0 and formula_ci[0] > 0
            ),
        }

    transitions = paired.loc[paired["transition"].isin(["corrected", "introduced"])].copy()
    rule_lookup, categories, libraries = load_rule_lookup(args.rule_cache)
    transitions = add_rule_evidence(transitions, rule_lookup, categories, libraries)
    corrected_frame = transitions.loc[transitions["transition"] == "corrected"]
    introduced_frame = transitions.loc[transitions["transition"] == "introduced"]

    all_corrected = set().union(*action_corrected.values()) if action_corrected else set()
    all_introduced = set().union(*action_introduced.values()) if action_introduced else set()
    output = {
        "status": "noise_v3_s3a_extended_matrix_decision_complete",
        "queries": int(len(baseline)),
        "official_errors": int(len(base_wrong)),
        "official_near_errors": int(len(near_wrong)),
        "matrix_cells": int(len(action_results)),
        "action_results": action_results,
        "no_op_aware_headroom": {
            "s3a_recoverable_errors": len(all_corrected),
            "s3a_unique_beyond_s1c_s2": len(all_corrected - prior),
            "combined_s1c_s2_s3a_recoverable": len(prior | all_corrected),
            "combined_delta_recall1_upper_bound": float(len(prior | all_corrected) / len(baseline)),
            "combined_near_recoverable": len((prior | all_corrected) & near_wrong),
            "any_action_can_introduce": len(all_introduced),
            "claim_limit": "Outcome-selected no-op oracle; this is headroom, not a deployable policy.",
        },
        "introduced_error_audit": {
            "transition_rows": int(len(introduced_frame)),
            "unique_queries": int(introduced_frame["query_index"].nunique()),
            "top_action_step_role": top_counts(
                introduced_frame, ["selector", "attenuation", "step", "target_role"],
            ),
            "top_query_to_wrong_formula": top_counts(
                introduced_frame, ["query_formula", "winner_formula"],
            ),
            "wrong_candidate_mces_grade": top_counts(
                introduced_frame, ["winner_mces_grade_name"],
            ),
            "same_formula_fraction": (
                float(introduced_frame["winner_same_formula"].mean())
                if len(introduced_frame) else None
            ),
            "mean_target_rule_jaccard": safe_mean(introduced_frame, "target_rule_jaccard"),
            "mean_baseline_rule_jaccard": safe_mean(introduced_frame, "baseline_rule_jaccard"),
            "interpretation": (
                "Every introduced-error row is retained. Rule overlap is observed evidence "
                "for stratification and explanation, never the correctness label."
            ),
        },
        "corrected_error_audit": {
            "transition_rows": int(len(corrected_frame)),
            "unique_queries": int(corrected_frame["query_index"].nunique()),
            "top_action_step_role": top_counts(
                corrected_frame, ["selector", "attenuation", "step", "target_role"],
            ),
            "mean_target_rule_jaccard": safe_mean(corrected_frame, "target_rule_jaccard"),
            "mean_baseline_rule_jaccard": safe_mean(corrected_frame, "baseline_rule_jaccard"),
        },
        "policy_entry_gates": {
            "combined_headroom_ge_1000_errors": bool(len(prior | all_corrected) >= 1000),
            "at_least_one_cell_net_positive": bool(
                any(item["net"] > 0 for item in action_results.values())
            ),
            "at_least_one_cell_specific": bool(
                any(item["specificity_gate"] for item in action_results.values())
            ),
            "introduced_errors_fully_materialized": True,
        },
        "claim_limit": (
            "S3A establishes action-space coverage and action-specific risks. It does not "
            "establish an OOF policy or a fine-tuned DreaMS improvement."
        ),
        "provenance": {
            "paired_sha256": sha256_file(paired_path),
            "matrix_validation_sha256": sha256_file(validation_path),
            "candidate_graph_sha256": sha256_file(args.cache),
            "rule_cache_sha256": sha256_file(args.rule_cache),
            "s1c_paired_sha256": sha256_file(args.s1c_dir / "paired_interventions.csv.gz"),
            "s2_paired_sha256": sha256_file(args.s2_dir / "paired_interventions.csv.gz"),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    output["policy_entry_gates"]["pass"] = all(output["policy_entry_gates"].values())
    decision_path = args.s3a_dir / "decision.json"
    transition_path = args.s3a_dir / "transition_audit.csv.gz"
    cell_path = args.s3a_dir / "cell_summary.csv"
    decision_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    transitions.to_csv(transition_path, index=False)
    pd.DataFrame([
        {"cell": name, **values} for name, values in action_results.items()
    ]).to_csv(cell_path, index=False)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
