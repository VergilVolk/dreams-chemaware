#!/usr/bin/env python
"""Test explicit seed--reaction paths beyond the B37 catalogue-topology prior.

Primary estimand
-----------------
On the frozen six-domain B37 development graph, compare the exact B37
catalogue-topology action with the same model plus one truth-blind scalar:
the fraction of pre-existing seed rotations in which a candidate has an
explicit direct reaction neighbour.  Twenty degree-preserving rewired graphs
provide capacity-matched one-feature null arms.

An enriched event-summary arm is reported as exploratory headroom only.  This
stage changes neither DreaMS embeddings nor the candidate graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

EXPECTED_QUERIES = 860
EXPECTED_CANDIDATES = 6695
EXPECTED_CONTEXT_ROWS = 38999
TOPOLOGY_FEATURES = ["spectral_score", "network_member", "known_log_degree"]
PRIMARY_PATH_FEATURE = "explicit_direct_path_context_fraction"
ENRICHED_PATH_FEATURES = [
    PRIMARY_PATH_FEATURE,
    "explicit_direct_seed_count_mean",
    "explicit_direct_event_count_mean",
    "explicit_rhea_context_fraction",
    "explicit_kegg_context_fraction",
    "explicit_direction_supported_context_fraction",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def aggregate_path_contexts(contexts: pd.DataFrame, rewires: int) -> pd.DataFrame:
    required = {
        "query_id", "candidate_id", "seed_stratum", "has_direct_path",
        "direct_seed_count", "direct_event_count", "direct_rhea_event_count",
        "direct_kegg_event_count", "direction_supported_event_count",
        *(f"rewire_support_{repeat:02d}" for repeat in range(rewires)),
    }
    if missing := required - set(contexts.columns):
        raise RuntimeError(f"B38-M0 candidate contexts miss: {sorted(missing)}")
    keys = ["query_id", "candidate_id"]
    if contexts.duplicated([*keys, "seed_stratum"]).any():
        raise RuntimeError("duplicate query/candidate/seed-context rows")
    local = contexts.copy()
    for column in required - {"query_id", "candidate_id", "seed_stratum"}:
        local[column] = pd.to_numeric(local[column], errors="raise")
    local["rhea_present"] = local["direct_rhea_event_count"].gt(0).astype(float)
    local["kegg_present"] = local["direct_kegg_event_count"].gt(0).astype(float)
    local["direction_supported_present"] = (
        local["direction_supported_event_count"].gt(0).astype(float)
    )
    named: dict[str, tuple[str, str]] = {
        "seed_contexts": ("seed_stratum", "count"),
        PRIMARY_PATH_FEATURE: ("has_direct_path", "mean"),
        "explicit_direct_seed_count_mean": ("direct_seed_count", "mean"),
        "explicit_direct_event_count_mean": ("direct_event_count", "mean"),
        "explicit_rhea_context_fraction": ("rhea_present", "mean"),
        "explicit_kegg_context_fraction": ("kegg_present", "mean"),
        "explicit_direction_supported_context_fraction": (
            "direction_supported_present", "mean"
        ),
    }
    for repeat in range(rewires):
        named[f"rewire_context_fraction_{repeat:02d}"] = (
            f"rewire_support_{repeat:02d}", "mean"
        )
    aggregated = local.groupby(keys, sort=False).agg(**named).reset_index()
    numeric = [column for column in aggregated.columns if column not in keys]
    if not np.isfinite(aggregated[numeric].to_numpy(float)).all():
        raise RuntimeError("non-finite B38 path aggregate")
    return aggregated


def compare_topology_replay(observed: pd.DataFrame, archived_path: Path) -> dict[str, Any]:
    archived = pd.read_csv(archived_path)
    archived = archived.loc[archived["arm"].eq("catalog_topology")].copy()
    current = observed.loc[observed["arm"].eq("catalog_topology")].copy()
    columns = [
        "query_id", "source", "truth_candidate_id", "truth_formula",
        "baseline_candidate_id", "proposed_candidate_id", "final_candidate_id",
        "baseline_correct", "proposal_unique", "intervene", "final_correct",
        "corrected", "introduced", "delta", "gate_name",
        "proposal_probability", "gate_margin", "gate_probability",
    ]
    for label, frame in (("archived", archived), ("current", current)):
        if len(frame) != EXPECTED_QUERIES or frame["query_id"].nunique() != EXPECTED_QUERIES:
            raise RuntimeError(f"{label} topology arm is not the frozen 860 queries")
        if missing := set(columns) - set(frame.columns):
            raise RuntimeError(f"{label} topology misses: {sorted(missing)}")
    joined = archived[columns].merge(
        current[columns], on="query_id", how="outer", validate="one_to_one",
        suffixes=("_archived", "_current"), indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("B37/B38 topology query sets differ")
    categorical = [column for column in columns if column not in {
        "query_id", "proposal_probability", "gate_margin", "gate_probability"
    }]
    mismatches = {
        column: int((
            joined[f"{column}_archived"].astype(str)
            != joined[f"{column}_current"].astype(str)
        ).sum())
        for column in categorical
    }
    numeric_errors = {
        column: float(np.max(np.abs(
            pd.to_numeric(joined[f"{column}_archived"], errors="raise").to_numpy(float)
            - pd.to_numeric(joined[f"{column}_current"], errors="raise").to_numpy(float)
        ), initial=0.0))
        for column in ("proposal_probability", "gate_margin", "gate_probability")
    }
    passed = not any(mismatches.values()) and all(value <= 1e-12 for value in numeric_errors.values())
    report = {
        "queries": int(len(joined)),
        "categorical_mismatches": mismatches,
        "maximum_numeric_errors": numeric_errors,
        "pass": bool(passed),
    }
    if not passed:
        raise RuntimeError(f"B38 failed exact B37 topology replay: {report}")
    return report


def empirical_upper_p(real: float, null: list[float]) -> float:
    return float((1 + sum(value >= real - 1e-15 for value in null)) / (1 + len(null)))


def main() -> None:
    import sklearn
    from audit_bioaware_b12_multicohort_catalog_action import (
        EXPECTED_DOMAINS, build_universe, split_domain,
    )
    from audit_bioaware_b36_reaction_specificity_ablation import (
        apply_selected_gate, choose_gate, paired_contrast, prepare_candidates,
        score_arm, summarize_arm,
    )
    from audit_bioaware_b11_catalog_interaction_action import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m0-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b38_m0_localcheck_20260913_v4",
    )
    parser.add_argument(
        "--b37-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b37_local_fullcheck_v2_20260913",
    )
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
    parser.add_argument("--rewire-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    # Fixed to the archived B37 common-random-number schedule so the topology
    # arm is an exact querywise replay, not merely a numerically similar refit.
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.rewire_repeats != 20:
        raise ValueError("B38-M1 freezes exactly 20 degree-preserving rewires")
    if args.bootstrap_resamples < 500:
        raise ValueError("bootstrap-resamples must be at least 500")
    required = [
        args.m0_dir / "report.json",
        args.m0_dir / "candidate_path_coverage.csv.gz",
        args.b37_dir / "catalog_ablation_transitions.csv.gz",
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    m0_report = json.loads((args.m0_dir / "report.json").read_text(encoding="utf-8"))
    if not m0_report.get("pass_to_b38_m1_primary", False):
        raise RuntimeError("B38-M0 primary explicit-path gate did not pass")

    candidates, universe = build_universe(args)
    candidates = prepare_candidates(candidates)
    contexts = pd.read_csv(args.m0_dir / "candidate_path_coverage.csv.gz")
    if len(contexts) != EXPECTED_CONTEXT_ROWS:
        raise RuntimeError(f"B38-M0 context rows changed: {len(contexts)}")
    aggregates = aggregate_path_contexts(contexts, args.rewire_repeats)
    if len(aggregates) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"B38 aggregate candidate rows changed: {len(aggregates)}")
    before = len(candidates)
    candidates = candidates.merge(
        aggregates, on=["query_id", "candidate_id"], how="left",
        validate="one_to_one", sort=False,
    )
    if len(candidates) != before or candidates[PRIMARY_PATH_FEATURE].isna().any():
        raise RuntimeError("B38 explicit path merge did not exactly cover B37 graph")

    arms: dict[str, list[str]] = {
        "catalog_topology": TOPOLOGY_FEATURES,
        "catalog_topology_plus_explicit_direct_path": [
            *TOPOLOGY_FEATURES, PRIMARY_PATH_FEATURE,
        ],
        "catalog_topology_plus_enriched_explicit_event_summary": [
            *TOPOLOGY_FEATURES, *ENRICHED_PATH_FEATURES,
        ],
    }
    for repeat in range(args.rewire_repeats):
        arms[f"catalog_topology_plus_degree_rewire_{repeat:02d}"] = [
            *TOPOLOGY_FEATURES, f"rewire_context_fraction_{repeat:02d}"
        ]

    outer_tables: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        for arm, features in arms.items():
            specification = {
                "arm": arm, "features": features,
                "null_method": None, "null_seed": None,
            }
            # Match B37 common-random-number schedule exactly.
            model_seed = args.seed + 100 * outer_index
            outer_scored, outer_fit, _ = score_arm(
                outer_train, outer_test, specification, outer_domain, model_seed
            )
            inner_parts: list[pd.DataFrame] = []
            inner_fits: list[dict[str, Any]] = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                scored, fit_report, _ = score_arm(
                    inner_train, inner_test, specification,
                    f"outer={outer_domain}|inner={inner_domain}",
                    model_seed + inner_index + 1,
                )
                inner_parts.append(scored)
                inner_fits.append(fit_report)
            selected, ledger = choose_gate(pd.concat(inner_parts, ignore_index=True))
            for item in ledger:
                selection_rows.append({
                    "outer_domain": outer_domain, "arm": arm, **item,
                })
            result = apply_selected_gate(outer_scored, selected)
            result["arm"] = arm
            result["outer_domain"] = outer_domain
            outer_tables.append(result)
            summary = summarize(result)
            fold_reports.append({
                "outer_domain": outer_domain,
                "arm": arm,
                "features": features,
                "selected_gate": selected,
                "outer_fit": outer_fit,
                "inner_fits": inner_fits,
                "outer_result": summary,
            })
            print(
                f"[B38-M1 {outer_domain} {arm}] "
                f"dR1={summary['delta_recall1']:+.4f} "
                f"C/I={summary['corrected']}/{summary['introduced']}",
                flush=True,
            )

    results = pd.concat(outer_tables, ignore_index=True)
    for arm in arms:
        local = results.loc[results["arm"].eq(arm)]
        if len(local) != EXPECTED_QUERIES or local["query_id"].nunique() != EXPECTED_QUERIES:
            raise RuntimeError(f"{arm}: incomplete 860-query outer OOF coverage")
    replay = compare_topology_replay(
        results, args.b37_dir / "catalog_ablation_transitions.csv.gz"
    )
    arm_reports = {
        arm: {
            **summarize_arm(
                results.loc[results["arm"].eq(arm)].copy(),
                args.bootstrap_resamples,
                args.seed + 100000 + index * 10,
            ),
            "features": features,
            "by_domain": {
                domain: summarize(results.loc[
                    results["arm"].eq(arm) & results["source"].eq(domain)
                ])
                for domain in EXPECTED_DOMAINS
            },
        }
        for index, (arm, features) in enumerate(arms.items())
    }
    real_arm = "catalog_topology_plus_explicit_direct_path"
    enriched_arm = "catalog_topology_plus_enriched_explicit_event_summary"
    real_vs_topology = paired_contrast(
        results, real_arm, "catalog_topology",
        args.bootstrap_resamples, args.seed + 200000,
    )
    enriched_vs_topology = paired_contrast(
        results, enriched_arm, "catalog_topology",
        args.bootstrap_resamples, args.seed + 200010,
    )
    null_contrasts: list[dict[str, Any]] = []
    for repeat in range(args.rewire_repeats):
        arm = f"catalog_topology_plus_degree_rewire_{repeat:02d}"
        null_contrasts.append(paired_contrast(
            results, arm, "catalog_topology",
            args.bootstrap_resamples, args.seed + 201000 + repeat * 10,
        ))
    null_effects = [item["mean_top1_difference"] for item in null_contrasts]
    real_effect = float(real_vs_topology["mean_top1_difference"])
    by_domain_increment = {
        domain: float(
            results.loc[
                results["arm"].eq(real_arm) & results["source"].eq(domain),
                "final_correct",
            ].astype(int).mean()
            - results.loc[
                results["arm"].eq("catalog_topology") & results["source"].eq(domain),
                "final_correct",
            ].astype(int).mean()
        )
        for domain in EXPECTED_DOMAINS
    }
    real_report = arm_reports[real_arm]
    empirical_p = empirical_upper_p(real_effect, null_effects)
    gates = {
        "exact_b37_topology_querywise_replay": bool(replay["pass"]),
        "incremental_recall1_ge_3pp": bool(real_effect >= 0.03),
        "incremental_identity_cluster_ci_positive": bool(
            real_vs_topology["identity_cluster_bootstrap"]["ci_low"] > 0
        ),
        "incremental_formula_cluster_ci_positive": bool(
            real_vs_topology["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "corrected_gt_2x_introduced_vs_dreams": bool(
            real_report["corrected"] > 2 * real_report["introduced"]
        ),
        "increment_nonnegative_in_every_domain": bool(
            all(value >= -1e-15 for value in by_domain_increment.values())
        ),
        "beats_all_degree_preserving_rewires": bool(
            real_effect > max(null_effects, default=float("-inf")) + 1e-15
        ),
        "degree_rewire_empirical_p_le_0_05": bool(empirical_p <= 0.05),
    }
    pass_to_path_set_model = bool(all(gates.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "explicit_path_arm_transitions.csv.gz"
    selection_path = args.output_dir / "inner_gate_selection_ledger.csv.gz"
    aggregate_path = args.output_dir / "candidate_explicit_path_features.csv.gz"
    results.to_csv(transition_path, index=False, compression="gzip")
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False, compression="gzip")
    aggregates.to_csv(aggregate_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b38_m1_explicit_path_incremental_complete",
        "formal": True,
        "pass_to_b38_m2_path_set_model": pass_to_path_set_model,
        "protocol": (
            "six-domain nested LOSO with held identity/formula purge; exact B37 "
            "topology replay; actual direct-path support versus 20 degree-preserving "
            "one-feature rewire controls"
        ),
        "b37_topology_replay": replay,
        "primary_real_vs_topology": real_vs_topology,
        "primary_by_domain_increment": by_domain_increment,
        "primary_degree_rewire_null": {
            "repeats": args.rewire_repeats,
            "effects": null_effects,
            "mean": float(np.mean(null_effects)),
            "maximum": float(max(null_effects)),
            "empirical_upper_p": empirical_p,
            "contrasts": null_contrasts,
        },
        "exploratory_enriched_event_summary_vs_topology": enriched_vs_topology,
        "arms": arm_reports,
        "gates": gates,
        "folds": fold_reports,
        "contracts": {
            "same_candidate_graph_across_arms": True,
            "same_model_capacity_primary_vs_each_rewire": True,
            "held_truth_identity_and_formula_purged": True,
            "outer_outcomes_used_for_selection": False,
            "degree_rewire_nulls_selected_without_outcomes": True,
            "joint_spectral_coabundance_claim_made": False,
            "shared_embedding_changed": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            **universe["provenance"],
            "m0_report": sha256(args.m0_dir / "report.json"),
            "m0_candidate_contexts": sha256(args.m0_dir / "candidate_path_coverage.csv.gz"),
            "b37_report": sha256(args.b37_dir / "report.json"),
            "b37_transitions": sha256(args.b37_dir / "catalog_ablation_transitions.csv.gz"),
            "candidate_explicit_path_features": sha256(aggregate_path),
            "explicit_path_arm_transitions": sha256(transition_path),
            "inner_gate_selection_ledger": sha256(selection_path),
            "script": sha256(Path(__file__)),
            "scikit_learn_version": sklearn.__version__,
        },
        "claim_limit": (
            "B38-M1 is an opened-development incremental ranking test. Passing would "
            "justify a path-event model, not establish biological causality, clean "
            "shared-embedding improvement, external confirmation, or SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "topology": arm_reports["catalog_topology"],
        "explicit_direct_path": real_report,
        "real_vs_topology": real_vs_topology,
        "rewire_null": report["primary_degree_rewire_null"],
        "enriched_vs_topology": enriched_vs_topology,
        "by_domain_increment": by_domain_increment,
        "gates": gates,
        "pass_to_b38_m2_path_set_model": pass_to_path_set_model,
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
