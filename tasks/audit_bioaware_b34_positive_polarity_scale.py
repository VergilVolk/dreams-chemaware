#!/usr/bin/env python
"""Scale-test the frozen B17+B30 BioAware action on held positive-ion queries.

B17 and B30 were selected on 860 negative-ion development rows.  This audit
reconstructs every B17 outer-domain model exactly, keeps component/gate/policy
selection on the original negative-ion inner folds, and applies the resulting
model once to the positive-ion rows of the held Full16 biological source.

The B30 candidate-sink set is also reconstructed only from the original
negative-ion actions in other sources.  Positive-ion outcomes never select a
model, threshold, policy, or sink.  The 860 original rows are replayed exactly
before the 878-row polarity-transfer result is reported.

This is a larger, source-held-out polarity-transfer audit.  It is not a new
external cohort and therefore cannot by itself establish blind SOTA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES,
    apply_gate as apply_linear_gate,
    atomic_json,
    cluster_bootstrap,
    score_queries as score_linear,
    sha256,
    summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    INTERNAL_DOMAINS,
    build_universe,
    split_domain,
)
from audit_bioaware_b16_pairwise_nonlinear_action import (  # noqa: E402
    FEATURE_FAMILIES,
    apply_gate as apply_nonlinear_gate,
    prepare,
    score_queries as score_nonlinear,
)
from audit_bioaware_b17_nested_union_action import (  # noqa: E402
    combine,
)
from audit_bioaware_b27_candidate_spectral_veto import load_b20  # noqa: E402
from audit_bioaware_b30_cross_source_sink_veto import (  # noqa: E402
    apply_veto,
    candidate_history,
    physical,
)


EXPECTED_NEGATIVE_ROWS = 860
EXPECTED_POSITIVE_ROWS = 878
EXPECTED_COMBINED_ROWS = EXPECTED_NEGATIVE_ROWS + EXPECTED_POSITIVE_ROWS
RISK_PENALTY = 2


def arguments() -> argparse.Namespace:
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
    parser.add_argument(
        "--b17-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/nested_domain_loso_transitions.csv.gz",
    )
    parser.add_argument(
        "--b17-report", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/report.json",
    )
    parser.add_argument(
        "--b20-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b20_direct_action_manifest_localcheck_20260907_v1",
    )
    parser.add_argument(
        "--b30-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b30_cross_source_sink_veto_localcheck_20260907_v1/nested_sink_veto_transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--model-seed", type=int, default=20260907)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def compare_replay(
    replay: pd.DataFrame, frozen_path: Path, label: str, *, require_exact: bool,
) -> dict[str, int]:
    frozen = pd.read_csv(frozen_path)
    columns = (
        "baseline_candidate_id", "baseline_correct", "final_candidate_id",
        "intervene", "final_correct", "corrected", "introduced", "delta",
    )
    joined = replay[["query_id", *columns]].merge(
        frozen[["query_id", *columns]], on="query_id",
        suffixes=("_replay", "_frozen"), validate="one_to_one",
    )
    if len(joined) != len(replay) or len(joined) != len(frozen):
        raise RuntimeError(
            f"{label} replay coverage mismatch: replay={len(replay)} "
            f"frozen={len(frozen)} joined={len(joined)}"
        )
    mismatches: dict[str, int] = {}
    for column in columns:
        left = joined[f"{column}_replay"]
        right = joined[f"{column}_frozen"]
        if column in {
            "baseline_correct", "intervene", "final_correct", "corrected",
            "introduced",
        }:
            count = int(left.astype(bool).ne(right.astype(bool)).sum())
        else:
            count = int(left.astype(str).ne(right.astype(str)).sum())
        mismatches[column] = count
    if mismatches["baseline_candidate_id"] or mismatches["baseline_correct"]:
        raise RuntimeError(f"{label} baseline replay failed: {mismatches}")
    if require_exact and any(mismatches.values()):
        raise RuntimeError(f"{label} querywise replay failed: {mismatches}")
    return mismatches


def score_b17(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Refit only the exact frozen B17 fold configurations.

    B17 did not save fitted sklearn estimators.  Refitting under a different
    sklearn build can move boundary decisions, so negative rows are retained
    only as a version-drift audit.  The original negative result is always
    loaded from the frozen transition artifact in ``main``.
    """
    candidates, _ = build_universe(args)
    candidates = prepare(candidates)
    frozen_report = json.loads(args.b17_report.read_text(encoding="utf-8"))
    if frozen_report.get("status") != "bioaware_b17_nested_union_action_complete":
        raise RuntimeError("B34 requires a passing frozen B17 report")
    frozen_folds = {
        str(item["outer_domain"]): item for item in frozen_report["folds"]
    }
    if set(frozen_folds) != set(EXPECTED_DOMAINS):
        raise RuntimeError("B34 frozen B17 fold coverage changed")
    negative_results: list[pd.DataFrame] = []
    positive_results: list[pd.DataFrame] = []
    folds: list[dict] = []

    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_negative = split_domain(candidates, outer_domain)
        held = candidates.loc[candidates["source"].eq(outer_domain)].copy()
        outer_positive = held.loc[held["polarity"].eq("positive")].copy()
        frozen = frozen_folds[outer_domain]
        linear_selected = frozen["linear_selected"]
        nonlinear_selected = frozen["nonlinear_selected"]
        policy = str(frozen["policy_selected"])
        linear_recipe = str(linear_selected["recipe"])
        nonlinear_family = str(nonlinear_selected["feature_family"])

        linear_negative, _ = score_linear(
            outer_train, outer_negative, FEATURE_RECIPES[linear_recipe], outer_domain
        )
        linear_negative_result = apply_linear_gate(
            linear_negative,
            float(linear_selected["margin"]),
            float(linear_selected["probability"]),
        )
        nonlinear_negative, _ = score_nonlinear(
            outer_train, outer_negative, FEATURE_FAMILIES[nonlinear_family],
            outer_domain, args.model_seed + 100 * outer_index,
        )
        nonlinear_negative_result = apply_nonlinear_gate(
            nonlinear_negative,
            float(nonlinear_selected["margin"]),
            float(nonlinear_selected["probability"]),
        )
        negative = combine(linear_negative_result, nonlinear_negative_result, policy)
        negative_results.append(negative)

        if not outer_positive.empty:
            linear_positive, _ = score_linear(
                outer_train, outer_positive, FEATURE_RECIPES[linear_recipe],
                f"{outer_domain}|positive-transfer",
            )
            linear_positive_result = apply_linear_gate(
                linear_positive,
                float(linear_selected["margin"]),
                float(linear_selected["probability"]),
            )
            nonlinear_positive, _ = score_nonlinear(
                outer_train, outer_positive, FEATURE_FAMILIES[nonlinear_family],
                f"{outer_domain}|positive-transfer",
                args.model_seed + 100 * outer_index,
            )
            nonlinear_positive_result = apply_nonlinear_gate(
                nonlinear_positive,
                float(nonlinear_selected["margin"]),
                float(nonlinear_selected["probability"]),
            )
            positive = combine(
                linear_positive_result, nonlinear_positive_result, policy
            )
            positive_results.append(positive)
        folds.append({
            "outer_domain": outer_domain,
            "positive_transfer_queries": int(outer_positive["query_id"].nunique()),
            "linear_recipe": linear_recipe,
            "linear_margin": float(linear_selected["margin"]),
            "linear_probability": float(linear_selected["probability"]),
            "nonlinear_family": nonlinear_family,
            "nonlinear_margin": float(nonlinear_selected["margin"]),
            "nonlinear_probability": float(nonlinear_selected["probability"]),
            "combination_policy": policy,
        })
        print(
            f"[B34 {outer_domain}] positive_queries={outer_positive['query_id'].nunique()} "
            f"linear={linear_recipe} nonlinear={nonlinear_family} policy={policy}",
            flush=True,
        )

    negative = pd.concat(negative_results, ignore_index=True)
    positive = pd.concat(positive_results, ignore_index=True)
    if len(negative) != EXPECTED_NEGATIVE_ROWS or negative["query_id"].nunique() != len(negative):
        raise RuntimeError(f"B34 negative coverage changed: {len(negative)}")
    if len(positive) != EXPECTED_POSITIVE_ROWS or positive["query_id"].nunique() != len(positive):
        raise RuntimeError(f"B34 positive coverage changed: {len(positive)}")
    if set(positive["source"].astype(str)) != set(INTERNAL_DOMAINS):
        raise RuntimeError("B34 positive sources changed")
    return negative, positive, folds


def sink_sets(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, int]]:
    frame, _, _ = load_b20(args.b20_dir)
    physical_frame = physical(frame)
    sinks: dict[str, set[str]] = {}
    action_counts: dict[str, int] = {}
    for outer_source in EXPECTED_DOMAINS:
        development = physical_frame.loc[
            ~physical_frame["source"].eq(outer_source)
            & physical_frame["intervene"].astype(bool)
        ].copy()
        history = candidate_history(development)
        sinks[outer_source] = set(
            history.loc[history["candidate_sink"], "final_candidate_id"].astype(str)
        )
        action_counts[outer_source] = int(len(development))
    return sinks, action_counts


def describe(
    frame: pd.DataFrame, args: argparse.Namespace, seed_offset: int,
) -> dict:
    summary = summarize(frame)
    corrected = frame.loc[frame["corrected"].astype(bool)]
    introduced = frame.loc[frame["introduced"].astype(bool)]
    by_source = {
        source: summarize(frame.loc[frame["source"].eq(source)])
        for source in sorted(frame["source"].astype(str).unique())
    }
    return {
        **summary,
        "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
        "corrected_formulas": int(corrected["truth_formula"].nunique()),
        "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
        "introduced_formulas": int(introduced["truth_formula"].nunique()),
        "identity_cluster_bootstrap": cluster_bootstrap(
            frame, "truth_candidate_id", args.bootstrap_resamples,
            args.seed + seed_offset,
        ),
        "formula_cluster_bootstrap": cluster_bootstrap(
            frame, "truth_formula", args.bootstrap_resamples,
            args.seed + seed_offset + 1,
        ),
        "by_source": by_source,
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    dependencies = (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b17_transitions,
        args.b17_report, args.b30_transitions, args.b20_dir / "report.json",
        args.b20_dir / "direct_actions.csv.gz",
        args.b20_dir / "direct_action_manifest.npz",
    )
    for path in dependencies:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    b17_negative, b17_positive, folds = score_b17(args)
    b17_replay = compare_replay(
        b17_negative, args.b17_transitions, "B17 refit", require_exact=False
    )
    sinks, history_action_counts = sink_sets(args)
    frozen_b17 = pd.read_csv(args.b17_transitions)
    frozen_b30_reconstruction: list[pd.DataFrame] = []
    b30_positive_parts: list[pd.DataFrame] = []
    for source in EXPECTED_DOMAINS:
        frozen_b30_reconstruction.append(apply_veto(
            frozen_b17.loc[frozen_b17["source"].eq(source)].copy(),
            sinks[source],
        ))
        positive = b17_positive.loc[b17_positive["source"].eq(source)].copy()
        if not positive.empty:
            b30_positive_parts.append(apply_veto(positive, sinks[source]))
    reconstructed_b30 = pd.concat(frozen_b30_reconstruction, ignore_index=True)
    b30_positive = pd.concat(b30_positive_parts, ignore_index=True)
    b30_replay = compare_replay(
        reconstructed_b30, args.b30_transitions, "B30", require_exact=True
    )
    b30_negative = pd.read_csv(args.b30_transitions)

    common_columns = [
        "query_id", "held_label", "source", "truth_candidate_id",
        "truth_formula", "baseline_candidate_id", "baseline_correct",
        "B17_final_candidate_id", "b30_candidate_sink", "b30_veto",
        "final_candidate_id", "intervene", "final_correct", "corrected",
        "introduced", "delta", "combination_policy",
    ]
    b30_negative = b30_negative[common_columns].copy()
    b30_negative["evaluation_stratum"] = "original_negative_replay"
    b30_positive = b30_positive[common_columns].copy()
    b30_positive["evaluation_stratum"] = "held_positive_transfer"
    combined = pd.concat([b30_negative, b30_positive], ignore_index=True)
    if len(combined) != EXPECTED_COMBINED_ROWS:
        raise RuntimeError(f"B34 combined coverage changed: {len(combined)}")
    if combined["query_id"].nunique() != len(combined):
        raise RuntimeError("B34 combined query IDs are not unique")

    positive_report = describe(b30_positive, args, 10)
    combined_report = describe(combined, args, 20)
    negative_report = describe(b30_negative, args, 30)
    positive_formula_ci = positive_report["formula_cluster_bootstrap"]
    gates = {
        "B17_original_baseline_replay_exact": (
            b17_replay["baseline_candidate_id"] == 0
            and b17_replay["baseline_correct"] == 0
        ),
        "B30_original_860_querywise_replay_exact": not any(b30_replay.values()),
        "positive_transfer_rows_eq_878": len(b30_positive) == EXPECTED_POSITIVE_ROWS,
        "positive_transfer_gain_ge_3pp": positive_report["delta_recall1"] >= 0.03,
        "positive_transfer_formula_ci_low_positive": positive_formula_ci["ci_low"] > 0,
        "positive_transfer_corrected_gt_2x_introduced": (
            positive_report["corrected"] > RISK_PENALTY * positive_report["introduced"]
        ),
        "positive_transfer_every_source_nonnegative": all(
            item["delta_recall1"] >= 0
            for item in positive_report["by_source"].values()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transitions_path = args.output_dir / "b34_large_scale_transitions.csv.gz"
    combined.to_csv(transitions_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b34_positive_polarity_scale_complete",
        "formal": True,
        "protocol": (
            "B17 component/gate/policy selection remains negative-ion inner-domain OOF; "
            "held Full16 positive-ion outcomes are evaluated once; B30 sinks use only "
            "original negative-ion actions from other sources"
        ),
        "original_negative_replay": negative_report,
        "held_positive_transfer_primary": positive_report,
        "mixed_polarity_descriptive_only": {
            **combined_report,
            "valid_as_larger_same_task_confirmation": False,
            "reason": (
                "the negative stratum loads archived B30 decisions while the positive "
                "stratum refits the frozen B17 recipe in the current sklearn environment; "
                "the strata also represent different ionisation tasks"
            ),
        },
        "development_effect_reference": {
            "B30_delta_recall1": 54 / 860 - 3 / 860,
            "B30_corrected": 54,
            "B30_introduced": 3,
            "interpretation": "opened 860-row comparator, not a guaranteed target",
        },
        "folds": folds,
        "model_seed": args.model_seed,
        "B17_recipe_refit_version_drift": {
            "mismatches": b17_replay,
            "final_candidate_mismatch_fraction": (
                b17_replay["final_candidate_id"] / EXPECTED_NEGATIVE_ROWS
            ),
            "interpretation": (
                "B17 saved decisions but not fitted sklearn estimators; boundary "
                "drift under the current sklearn build is reported, never substituted "
                "for the frozen 860-row result"
            ),
        },
        "B30_sink_history": {
            source: {
                "development_physical_actions": history_action_counts[source],
                "sink_candidates": sorted(sinks[source]),
            }
            for source in EXPECTED_DOMAINS
        },
        "gates": gates,
        "pass_large_scale_polarity_transfer": bool(all(gates.values())),
        "contracts": {
            "held_source_positive_outcomes_used_for_fitting_or_selection": False,
            "other_source_positive_outcomes_used_for_model_fitting": True,
            "positive_outcomes_used_for_component_gate_or_policy_selection": False,
            "positive_outcomes_used_for_sink_definition": False,
            "held_source_removed_from_training": True,
            "held_truth_identity_and_formula_purged": True,
            "B30_original_querywise_replay_required": True,
            "B17_frozen_fold_configuration_loaded": True,
            "B17_fitted_estimators_were_not_archived": True,
            "B17_positive_model_refitted_in_current_environment": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "internal_candidates_sha256": sha256(args.internal_candidates),
            "st_candidates_sha256": sha256(args.st_candidates),
            "st_queries_sha256": sha256(args.st_queries),
            "kgmn_candidates_sha256": sha256(args.kgmn_candidates),
            "kgmn_seeds_sha256": sha256(args.kgmn_seeds),
            "B17_transitions_sha256": sha256(args.b17_transitions),
            "B17_report_sha256": sha256(args.b17_report),
            "B30_transitions_sha256": sha256(args.b30_transitions),
            "transitions_sha256": sha256(transitions_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A source-held-out positive-polarity transfer audit in opened Full16 data. "
            "It is not a larger same-task replication of the 5.93 pp negative-ion "
            "result. The mixed-polarity aggregate is descriptive only and cannot "
            "establish independent validation, reaction causality, SOTA, or shared-"
            "embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
