#!/usr/bin/env python
"""Develop a candidate-protocol-matched BioAware v3 network expert.

The development universe is restricted to chemically approved exact [M-H]-
candidates whose calculated formula equals the query truth formula.  The
sample-MS1 mass-recovery feature is excluded because it is structurally
constant in this deployment protocol.  Model capacity and all gates remain at
the previously frozen v2 values; no ST001154 outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    PRIMARY_BASELINE_MARGIN_MAX,
    PRIMARY_C,
    PRIMARY_PROPOSAL_PROBABILITY_MIN,
    RISK_WEIGHT_BASELINE_CORRECT,
    evaluate_fold,
    formula_bootstrap,
    summarize,
)


FEATURES = [
    "spectral_score",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_same_formula_candidates(
    candidates: pd.DataFrame, integrity: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "best_library_row", "unit_id", *FEATURES,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"candidate table missing columns: {missing}")
    integrity_required = {
        "library_row", "calculated_formula", "approved_m_h_reference",
        "structure_identity_consistent",
    }
    missing_integrity = sorted(integrity_required - set(integrity.columns))
    if missing_integrity:
        raise RuntimeError(f"integrity table missing columns: {missing_integrity}")
    mapped = candidates.merge(
        integrity[list(integrity_required)],
        left_on="best_library_row", right_on="library_row",
        how="left", validate="many_to_one",
    )
    if mapped["calculated_formula"].isna().any():
        raise RuntimeError("candidate row lacks calculated formula")
    approved = (
        mapped["approved_m_h_reference"].astype(bool)
        & mapped["structure_identity_consistent"].astype(bool)
    )
    same_formula = mapped["calculated_formula"].astype(str).eq(
        mapped["truth_formula"].astype(str)
    )
    kept = mapped.loc[approved & same_formula].copy()
    sizes = kept.groupby("query_id", sort=False).size()
    valid_queries = set(sizes[sizes >= 2].index)
    kept = kept.loc[kept["query_id"].isin(valid_queries)].copy()
    if kept.empty:
        raise RuntimeError("same-formula development protocol is empty")
    positive_counts = kept.groupby("query_id", sort=False).apply(
        lambda group: int(
            group["candidate_id"].astype(str).eq(
                str(group["truth_candidate_id"].iloc[0])
            ).sum()
        ),
        include_groups=False,
    )
    if positive_counts.ne(1).any():
        raise RuntimeError("same-formula query does not contain exactly one truth candidate")
    if not kept["calculated_formula"].astype(str).eq(kept["truth_formula"].astype(str)).all():
        raise RuntimeError("formula mismatch survived same-formula filtering")
    return kept, {
        "source_candidate_rows": int(len(candidates)),
        "same_formula_candidate_rows": int(len(kept)),
        "same_formula_queries": int(kept["query_id"].nunique()),
        "same_formula_truth_identities": int(kept["truth_candidate_id"].nunique()),
        "same_formula_truth_formulas": int(kept["truth_formula"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/"
            "candidate_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--integrity-table", type=Path,
        default=Path(
            "data/validation/mona_negative_library_chemical_integrity_v1/"
            "library_row_integrity.csv.gz"
        ),
    )
    parser.add_argument(
        "--integrity-report", type=Path,
        default=Path("data/validation/mona_negative_library_chemical_integrity_v1/report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_development"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    for path in (args.candidate_features, args.integrity_table, args.integrity_report):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    integrity_report = json.loads(args.integrity_report.read_text(encoding="utf-8"))
    if integrity_report.get("status") != "mona_negative_library_chemical_integrity_complete":
        raise RuntimeError("MONA chemical-integrity report is invalid")
    candidates, filtering = filter_same_formula_candidates(
        pd.read_csv(args.candidate_features), pd.read_csv(args.integrity_table)
    )
    results: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for unit in sorted(candidates["unit_id"].astype(str).unique()):
        test = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        test_identities = set(test["truth_candidate_id"].astype(str))
        test_formulas = set(test["truth_formula"].astype(str))
        train = candidates.loc[
            ~candidates["unit_id"].astype(str).eq(unit)
            & ~candidates["truth_candidate_id"].astype(str).isin(test_identities)
            & ~candidates["truth_formula"].astype(str).isin(test_formulas)
        ].copy()
        if train["query_id"].nunique() < 100:
            raise RuntimeError(f"insufficient formula-purged training queries for {unit}")
        if set(train["truth_candidate_id"].astype(str)) & test_identities:
            raise RuntimeError(f"identity purge failed for {unit}")
        if set(train["truth_formula"].astype(str)) & test_formulas:
            raise RuntimeError(f"formula purge failed for {unit}")
        transition, fold = evaluate_fold(
            train, test, unit, features=FEATURES, require_raw_step0_edge=True
        )
        fold.update({
            "test_truth_identities": int(len(test_identities)),
            "test_truth_formulas": int(len(test_formulas)),
            "training_truth_identity_overlap": 0,
            "training_truth_formula_overlap": 0,
            "result": summarize(transition),
        })
        results.append(transition)
        fold_reports.append(fold)
        print(f"[same-formula LOSO] {unit}: {fold['result']}", flush=True)
    transitions = pd.concat(results, ignore_index=True)
    pooled = summarize(transitions)
    bootstrap = formula_bootstrap(transitions, args.bootstrap_resamples, args.seed)
    gates = {
        "queries_ge_400": candidates["query_id"].nunique() >= 400,
        "truth_identities_ge_100": candidates["truth_candidate_id"].nunique() >= 100,
        "truth_formulas_ge_100": candidates["truth_formula"].nunique() >= 100,
        "corrected_gt_introduced": pooled["corrected"] > pooled["introduced"],
        "risk_weighted_net_positive": pooled["risk_weighted_net_lambda2"] > 0,
        "formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
        "all_units_recall_nonnegative": all(
            fold["result"]["delta_recall1"] >= 0 for fold in fold_reports
        ),
    }
    report = {
        "status": "bioaware_same_formula_network_expert_v3_development_complete",
        "formal": True,
        "protocol": (
            "chemically approved exact [M-H]- same-formula candidate groups; eight-source LOSO; "
            "test truth identity and formula purged from training; fixed v2 capacity and gates"
        ),
        "filtering": filtering,
        "features": FEATURES,
        "excluded_protocol_mismatched_feature": "known_mass_candidate_fraction",
        "configuration": {
            "C": PRIMARY_C,
            "baseline_correct_training_weight": RISK_WEIGHT_BASELINE_CORRECT,
            "maximum_baseline_margin": PRIMARY_BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": PRIMARY_PROPOSAL_PROBABILITY_MIN,
            "requires_raw_step0_edge_validation": True,
        },
        "pooled": pooled,
        "formula_cluster_bootstrap": bootstrap,
        "folds": fold_reports,
        "gates": gates,
        "pass_to_falsification": all(gates.values()),
        "contracts": {
            "ST001154_outcomes_read": False,
            "candidate_formula_used_as_model_feature": False,
            "truth_identity_used_as_model_feature": False,
            "test_unit_fit_or_threshold_tuning": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
        },
        "provenance": {
            "source_candidate_features_sha256": sha256(args.candidate_features),
            "integrity_table_sha256": sha256(args.integrity_table),
            "integrity_report_sha256": sha256(args.integrity_report),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened-cohort protocol-matched development result. It is neither independent external "
            "confirmation nor a SOTA claim; candidate-label falsification and a new holdout are required."
        ),
    }
    args.output_dir.mkdir(parents=True)
    candidate_path = args.output_dir / "candidate_features.csv.gz"
    transition_path = args.output_dir / "query_transitions.csv.gz"
    candidates.to_csv(candidate_path, index=False, compression="gzip")
    transitions.to_csv(transition_path, index=False, compression="gzip")
    report["provenance"]["candidate_features_sha256"] = sha256(candidate_path)
    report["provenance"]["query_transitions_sha256"] = sha256(transition_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
