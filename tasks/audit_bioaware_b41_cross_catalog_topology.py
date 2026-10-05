#!/usr/bin/env python
"""DEPRECATED historical B41 catalogue audit.

The arm historically named ``rhea_topology_replay`` actually replays B37's
strict-KEGG-derived fields.  It is not an independent Rhea arm.  The frozen
historical output remains useful for provenance, but new scientific use must
run B42 (``audit_bioaware_b42_independent_catalog_topology.py``), which builds
Rhea independently and audits resource dependence explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    split_domain,
)
from audit_bioaware_b36_reaction_specificity_ablation import (  # noqa: E402
    SPECTRAL_FEATURES,
    apply_selected_gate,
    choose_gate,
    paired_contrast,
    prepare_candidates,
    score_arm,
    summarize_arm,
)
from evaluate_bioaware_b39_m2_fixed_action import atomic_csv_gzip  # noqa: E402


EXPECTED_QUERIES = 860
CROSS_FEATURES = [
    "independent_member_count",
    "independent_member_intersection",
    "independent_log_degree_mean",
    "independent_log_degree_min",
]


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def topology_signature(edges_path: Path) -> tuple[set[str], dict[str, float], dict[str, int]]:
    edges = pd.read_csv(edges_path, usecols=["ik14_a", "ik14_b"])
    left = edges["ik14_a"].dropna().astype(str).str.strip()
    right = edges["ik14_b"].dropna().astype(str).str.strip()
    pairs = pd.DataFrame({"a": left.to_numpy(), "b": right.to_numpy()})
    pairs = pairs.loc[pairs["a"].ne("") & pairs["b"].ne("") & pairs["a"].ne(pairs["b"])].copy()
    ordered = np.sort(pairs[["a", "b"]].to_numpy(str), axis=1)
    unique = pd.DataFrame(ordered, columns=["a", "b"]).drop_duplicates()
    endpoints = pd.concat([unique["a"], unique["b"]], ignore_index=True)
    degree = endpoints.value_counts().astype(int)
    members = set(degree.index.astype(str))
    log_degree = {str(key): float(np.log1p(value)) for key, value in degree.items()}
    report = {
        "raw_edges": int(len(edges)),
        "unique_undirected_nonself_edges": int(len(unique)),
        "members": int(len(members)),
        "maximum_degree": int(degree.max()) if len(degree) else 0,
    }
    return members, log_degree, report


def add_catalog_features(
    frame: pd.DataFrame,
    kegg_edges: Path,
    emrn_edges: Path,
    null_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    candidate = output["candidate_id"].astype(str)
    kegg_members, kegg_degree, kegg_report = topology_signature(kegg_edges)
    emrn_members, emrn_degree, emrn_report = topology_signature(emrn_edges)
    output["rhea_member"] = (output["network_member"].astype(float) > 0.5).astype(float)
    output["rhea_log_degree"] = output["known_log_degree"].astype(float)
    output["kegg_member"] = candidate.isin(kegg_members).astype(float)
    output["kegg_log_degree"] = candidate.map(kegg_degree).fillna(0.0).astype(float)
    output["emrn_member"] = candidate.isin(emrn_members).astype(float)
    output["emrn_log_degree"] = candidate.map(emrn_degree).fillna(0.0).astype(float)
    output["independent_member_count"] = output["rhea_member"] + output["kegg_member"]
    output["independent_member_intersection"] = output["rhea_member"] * output["kegg_member"]
    output["independent_log_degree_mean"] = 0.5 * (
        output["rhea_log_degree"] + output["kegg_log_degree"]
    )
    output["independent_log_degree_min"] = np.minimum(
        output["rhea_log_degree"], output["kegg_log_degree"]
    )

    identity_formula = output[["candidate_id", "truth_formula"]].drop_duplicates()
    conflicts = identity_formula.groupby("candidate_id")["truth_formula"].nunique()
    # A small number of catalogue identities are presented under more than one
    # query-formula label across the six source-specific candidate builders.
    # The null must therefore use the explicit (candidate, query formula) key;
    # silently selecting one formula would move attributes across mass/formula
    # candidate sets and invalidate the control.
    signature = output[
        ["candidate_id", "truth_formula", *CROSS_FEATURES]
    ].drop_duplicates(["candidate_id", "truth_formula"])
    if signature.duplicated(["candidate_id", "truth_formula"]).any():
        raise RuntimeError("candidate/formula signature is not unique")
    null_reports: list[dict[str, Any]] = []
    for repeat in range(null_repeats):
        assignment: dict[tuple[str, str], str] = {}
        movable_identities = 0
        for formula, local in signature.groupby("truth_formula", sort=True):
            identities = np.sort(local["candidate_id"].astype(str).to_numpy())
            if len(identities) < 2:
                assignment[(str(formula), str(identities[0]))] = str(identities[0])
                continue
            rng = np.random.default_rng(stable_seed(seed, "formula-null", repeat, formula))
            order = identities[rng.permutation(len(identities))]
            shift = 1 + stable_seed(seed, "shift", repeat, formula) % (len(order) - 1)
            sources = np.roll(order, int(shift))
            assignment.update({
                (str(formula), str(dst)): str(src)
                for dst, src in zip(order, sources)
            })
            movable_identities += len(identities)
        source_lookup = signature.set_index(["truth_formula", "candidate_id"])[CROSS_FEATURES]
        assigned = pd.Series(
            [
                assignment[(str(formula), str(candidate_id))]
                for formula, candidate_id in zip(
                    signature["truth_formula"], signature["candidate_id"]
                )
            ],
            index=signature.index,
            dtype=str,
        )
        source_keys = pd.MultiIndex.from_arrays(
            [signature["truth_formula"].astype(str), assigned.astype(str)],
            names=["truth_formula", "candidate_id"],
        )
        permuted = source_lookup.loc[source_keys].reset_index(drop=True)
        renamed = {column: f"null{repeat}_{column}" for column in CROSS_FEATURES}
        permuted = permuted.rename(columns=renamed)
        permuted["candidate_id"] = signature["candidate_id"].astype(str).to_numpy()
        permuted["truth_formula"] = signature["truth_formula"].astype(str).to_numpy()
        output = output.merge(
            permuted,
            on=["candidate_id", "truth_formula"],
            validate="many_to_one",
        )
        before = signature.groupby("truth_formula")[CROSS_FEATURES].sum().sort_index()
        after_identity = signature[["candidate_id", "truth_formula"]].merge(
            permuted,
            on=["candidate_id", "truth_formula"],
            validate="one_to_one",
        )
        after = after_identity.groupby("truth_formula")[[*renamed.values()]].sum().sort_index()
        maximum_sum_error = float(np.max(np.abs(before.to_numpy(float) - after.to_numpy(float)), initial=0.0))
        if maximum_sum_error > 1e-9:
            raise RuntimeError(f"formula-preserving null {repeat} changed feature multisets")
        changed = assigned.to_numpy(str) != signature["candidate_id"].astype(str).to_numpy()
        null_reports.append({
            "repeat": repeat,
            "identities": int(len(signature)),
            "movable_identities": int(movable_identities),
            "assigned_to_other_identity": int(changed.sum()),
            "maximum_formula_feature_sum_error": maximum_sum_error,
        })

    feature_columns = [
        "rhea_member", "rhea_log_degree", "kegg_member", "kegg_log_degree",
        "emrn_member", "emrn_log_degree", *CROSS_FEATURES,
        *(f"null{repeat}_{column}" for repeat in range(null_repeats) for column in CROSS_FEATURES),
    ]
    if not np.isfinite(output[feature_columns].to_numpy(float)).all():
        raise RuntimeError("cross-catalog features contain non-finite values")

    mapping: dict[str, Any] = {}
    for source, local in output.groupby("source", sort=True):
        truth = local["is_positive"].astype(bool)
        mapping[str(source)] = {
            "candidate_rows": int(len(local)),
            "queries": int(local["query_id"].nunique()),
            "rhea_candidate_fraction": float(local["rhea_member"].mean()),
            "kegg_candidate_fraction": float(local["kegg_member"].mean()),
            "emrn_candidate_fraction": float(local["emrn_member"].mean()),
            "rhea_truth_fraction": float(local.loc[truth, "rhea_member"].mean()),
            "kegg_truth_fraction": float(local.loc[truth, "kegg_member"].mean()),
            "emrn_truth_fraction": float(local.loc[truth, "emrn_member"].mean()),
            "rhea_kegg_membership_disagreement_fraction": float(
                local["rhea_member"].ne(local["kegg_member"]).mean()
            ),
        }
    audit = {
        "kegg": kegg_report,
        "emrn": emrn_report,
        "candidate_identities_with_multiple_query_formulas": int(conflicts.gt(1).sum()),
        "formula_preserving_nulls": null_reports,
        "source_mapping": mapping,
    }
    return output, audit


def arm_registry(null_repeats: int) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {
        "spectral_only": list(SPECTRAL_FEATURES),
        "rhea_topology_replay": list(SPECTRAL_FEATURES) + ["rhea_member", "rhea_log_degree"],
        "kegg_topology": list(SPECTRAL_FEATURES) + ["kegg_member", "kegg_log_degree"],
        "emrn_topology": list(SPECTRAL_FEATURES) + ["emrn_member", "emrn_log_degree"],
        "rhea_kegg_consensus": list(SPECTRAL_FEATURES) + CROSS_FEATURES,
        "all_catalog_topologies": list(SPECTRAL_FEATURES) + [
            "rhea_member", "rhea_log_degree", "kegg_member", "kegg_log_degree",
            "emrn_member", "emrn_log_degree",
        ],
    }
    for repeat in range(null_repeats):
        arms[f"formula_permuted_consensus_r{repeat:02d}"] = list(SPECTRAL_FEATURES) + [
            f"null{repeat}_{column}" for column in CROSS_FEATURES
        ]
    return arms


def run_nested_loso(
    candidates: pd.DataFrame,
    arms: dict[str, list[str]],
    seed: int,
    bootstrap_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    outer_tables: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        for arm, features in arms.items():
            specification = {"arm": arm, "features": features, "null_method": None, "null_seed": None}
            model_seed = seed + 100 * outer_index
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
                    inner_train,
                    inner_test,
                    specification,
                    f"outer={outer_domain}|inner={inner_domain}",
                    model_seed + inner_index + 1,
                )
                inner_parts.append(scored)
                inner_fits.append(fit_report)
            selected, ledger = choose_gate(pd.concat(inner_parts, ignore_index=True))
            for row in ledger:
                selection_rows.append({"outer_domain": outer_domain, "arm": arm, **row})
            result = apply_selected_gate(outer_scored, selected)
            result["arm"] = arm
            result["outer_domain"] = outer_domain
            outer_tables.append(result)
            fold_reports.append({
                "outer_domain": outer_domain,
                "arm": arm,
                "features": features,
                "selected_gate": selected,
                "outer_fit": outer_fit,
                "inner_fits": inner_fits,
            })
            print(
                f"[B41 {outer_domain} {arm}] gate={selected['gate_name']} "
                f"C/I={int(result['corrected'].sum())}/{int(result['introduced'].sum())}",
                flush=True,
            )
    transitions = pd.concat(outer_tables, ignore_index=True)
    for arm in arms:
        local = transitions.loc[transitions["arm"].eq(arm)]
        if len(local) != EXPECTED_QUERIES or local["query_id"].nunique() != EXPECTED_QUERIES:
            raise RuntimeError(f"{arm}: incomplete B41 OOF coverage")
    reports = {
        arm: {
            **summarize_arm(
                transitions.loc[transitions["arm"].eq(arm)].copy(),
                bootstrap_resamples,
                seed + 100000 + index * 10,
            ),
            "features": features,
            "by_domain": {
                domain: {
                    "queries": int(len(local)),
                    "delta_recall1": float(local["delta"].mean()),
                    "corrected": int(local["corrected"].sum()),
                    "introduced": int(local["introduced"].sum()),
                }
                for domain in EXPECTED_DOMAINS
                for local in [transitions.loc[
                    transitions["arm"].eq(arm) & transitions["source"].eq(domain)
                ]]
            },
        }
        for index, (arm, features) in enumerate(arms.items())
    }
    return transitions, pd.DataFrame(selection_rows), reports, fold_reports


def compare_rhea_replay(observed: pd.DataFrame, archived: pd.DataFrame) -> dict[str, Any]:
    left = archived.loc[archived["arm"].eq("catalog_topology")].copy()
    right = observed.loc[observed["arm"].eq("rhea_topology_replay")].copy()
    columns = [
        "query_id", "source", "baseline_candidate_id", "proposed_candidate_id",
        "final_candidate_id", "baseline_correct", "proposal_unique", "intervene",
        "final_correct", "corrected", "introduced", "delta", "gate_name",
        "proposal_probability", "gate_margin", "gate_probability",
    ]
    joined = left[columns].merge(
        right[columns], on="query_id", how="outer", validate="one_to_one",
        suffixes=("_b37", "_b41"), indicator=True,
    )
    if len(joined) != EXPECTED_QUERIES or not joined["_merge"].eq("both").all():
        raise RuntimeError("B37/B41 Rhea replay query set mismatch")
    categorical = [column for column in columns if column not in {
        "query_id", "proposal_probability", "gate_margin", "gate_probability"
    }]
    mismatches = {
        column: int((joined[f"{column}_b37"].astype(str) != joined[f"{column}_b41"].astype(str)).sum())
        for column in categorical
    }
    numeric = {
        column: float(np.max(np.abs(
            joined[f"{column}_b37"].to_numpy(float)
            - joined[f"{column}_b41"].to_numpy(float)
        ), initial=0.0))
        for column in ("proposal_probability", "gate_margin", "gate_probability")
    }
    passed = not any(mismatches.values()) and all(value <= 1e-12 for value in numeric.values())
    report = {"queries": len(joined), "categorical_mismatches": mismatches, "maximum_numeric_error": numeric, "pass": passed}
    if not passed:
        raise RuntimeError(f"B41 failed exact B37 Rhea replay: {report}")
    return report


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
    parser.add_argument(
        "--kegg-edges", type=Path,
        default=ROOT / "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz",
    )
    parser.add_argument(
        "--emrn-edges", type=Path,
        default=ROOT / "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz",
    )
    parser.add_argument(
        "--b37-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b37_local_fullcheck_v2_20260913",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    raise RuntimeError(
        "B41 is scientifically superseded: its historical 'rhea' arm is "
        "strict-KEGG-derived. Run audit_bioaware_b42_independent_catalog_topology.py."
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.null_repeats != 3:
        raise ValueError("formal B41 requires exactly three formula-preserving nulls")
    if args.bootstrap_resamples < 10000:
        raise ValueError("formal B41 requires at least 10,000 bootstrap resamples")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.kegg_edges, args.emrn_edges,
        args.b37_dir / "report.json",
        args.b37_dir / "catalog_ablation_transitions.csv.gz",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    universe, universe_provenance = build_universe(args)
    candidates = prepare_candidates(universe)
    candidates, mapping_audit = add_catalog_features(
        candidates, args.kegg_edges, args.emrn_edges,
        args.null_repeats, args.seed,
    )
    arms = arm_registry(args.null_repeats)
    transitions, selection, arm_reports, folds = run_nested_loso(
        candidates, arms, args.seed, args.bootstrap_resamples
    )
    archived_path = args.b37_dir / "catalog_ablation_transitions.csv.gz"
    archived = pd.read_csv(archived_path, low_memory=False)
    replay = compare_rhea_replay(transitions, archived)

    null_arms = [f"formula_permuted_consensus_r{repeat:02d}" for repeat in range(args.null_repeats)]
    null_deltas = [arm_reports[arm]["delta_recall1"] for arm in null_arms]
    consensus = arm_reports["rhea_kegg_consensus"]
    all_catalogues = arm_reports["all_catalog_topologies"]
    rhea = arm_reports["rhea_topology_replay"]
    kegg = arm_reports["kegg_topology"]
    emrn = arm_reports["emrn_topology"]

    contrast_specs = {
        "all_catalogues_vs_rhea": ("all_catalog_topologies", "rhea_topology_replay"),
        "rhea_kegg_consensus_vs_rhea": ("rhea_kegg_consensus", "rhea_topology_replay"),
        "kegg_vs_rhea": ("kegg_topology", "rhea_topology_replay"),
        "emrn_vs_rhea": ("emrn_topology", "rhea_topology_replay"),
        **{
            f"consensus_vs_null_r{repeat:02d}": (
                "rhea_kegg_consensus", f"formula_permuted_consensus_r{repeat:02d}"
            )
            for repeat in range(args.null_repeats)
        },
    }
    paired_contrasts = {
        name: paired_contrast(
            transitions, left, right, args.bootstrap_resamples,
            args.seed + 300000 + index * 10,
        )
        for index, (name, (left, right)) in enumerate(contrast_specs.items())
    }

    def every_domain_nonnegative(report: dict[str, Any]) -> bool:
        return all(item["delta_recall1"] >= -1e-15 for item in report["by_domain"].values())

    consensus_gates = {
        "delta_ge_0_03": bool(consensus["delta_recall1"] >= 0.03),
        "corrected_gt_2x_introduced": bool(consensus["corrected"] > 2 * consensus["introduced"]),
        "formula_ci_low_positive": bool(consensus["formula_cluster_bootstrap"]["ci_low"] > 0),
        "identity_ci_low_positive": bool(consensus["identity_cluster_bootstrap"]["ci_low"] > 0),
        "every_domain_nonnegative": every_domain_nonnegative(consensus),
        "beats_each_formula_permuted_null": bool(consensus["delta_recall1"] > max(null_deltas) + 1e-15),
    }
    portability_gates = {
        "exact_rhea_replay": bool(replay["pass"]),
        "kegg_delta_positive": bool(kegg["delta_recall1"] > 0),
        "kegg_formula_ci_low_positive": bool(kegg["formula_cluster_bootstrap"]["ci_low"] > 0),
        "emrn_delta_positive": bool(emrn["delta_recall1"] > 0),
        "emrn_formula_ci_low_positive": bool(emrn["formula_cluster_bootstrap"]["ci_low"] > 0),
        "consensus_pass": bool(all(consensus_gates.values())),
    }
    incremental = paired_contrasts["all_catalogues_vs_rhea"]
    incremental_by_domain = {
        domain: float(
            all_catalogues["by_domain"][domain]["delta_recall1"]
            - rhea["by_domain"][domain]["delta_recall1"]
        )
        for domain in EXPECTED_DOMAINS
    }
    multi_catalog_incremental_gates = {
        "mean_top1_difference_positive": bool(incremental["mean_top1_difference"] > 0),
        "identity_ci_low_positive": bool(
            incremental["identity_cluster_bootstrap"]["ci_low"] > 0
        ),
        "formula_ci_low_positive": bool(
            incremental["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "left_better_gt_2x_right_better": bool(
            incremental["left_better"] > 2 * incremental["right_better"]
        ),
        "every_domain_nonnegative": bool(
            all(value >= -1e-15 for value in incremental_by_domain.values())
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "cross_catalog_transitions.csv.gz"
    selection_path = args.output_dir / "inner_selection_ledger.csv.gz"
    features_path = args.output_dir / "candidate_catalog_features.csv.gz"
    atomic_csv_gzip(transitions_path, transitions)
    atomic_csv_gzip(selection_path, selection)
    atomic_csv_gzip(features_path, candidates[[
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula", "source",
        "is_positive", "rhea_member", "rhea_log_degree", "kegg_member",
        "kegg_log_degree", "emrn_member", "emrn_log_degree", *CROSS_FEATURES,
        *(f"null{repeat}_{column}" for repeat in range(args.null_repeats) for column in CROSS_FEATURES),
    ]])
    report = {
        "status": "bioaware_b41_cross_catalog_topology_complete",
        "formal": True,
        "protocol": "exact B37 six-domain nested LOSO with Rhea, strict KEGG, MetDNA2-EMRN and formula-preserving catalogue nulls",
        "queries": EXPECTED_QUERIES,
        "rhea_replay": replay,
        "mapping_audit": mapping_audit,
        "arm_reports": arm_reports,
        "paired_contrasts": paired_contrasts,
        "consensus_gates": consensus_gates,
        "portability_gates": portability_gates,
        "portable_catalog_prior": bool(all(portability_gates.values())),
        "multi_catalog_incremental": {
            "all_catalogues_vs_rhea_by_domain": incremental_by_domain,
            "gates": multi_catalog_incremental_gates,
            "pass": bool(all(multi_catalog_incremental_gates.values())),
        },
        "pass_to_shared_embedding": False,
        "folds": folds,
        "contracts": {
            "held_truth_identity_and_formula_purged": True,
            "outer_outcomes_used_for_selection": False,
            "same_capacity_and_rng_schedule_across_arms": True,
            "catalog_features_candidate_static": True,
            "sample_context_used": False,
            "P2b_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_provenance["provenance"],
            "kegg_edges": sha256(args.kegg_edges),
            "emrn_edges": sha256(args.emrn_edges),
            "b37_report": sha256(args.b37_dir / "report.json"),
            "b37_transitions": sha256(archived_path),
            "cross_catalog_transitions": sha256(transitions_path),
            "inner_selection_ledger": sha256(selection_path),
            "candidate_catalog_features": sha256(features_path),
            "script": sha256(Path(__file__)),
            "scikit_learn_version": sklearn.__version__,
        },
        "claim_limit": "Opened cross-resource catalogue-prior audit. A pass would establish candidate-prior portability across these resources and domains, not reaction propagation, prospective annotation, biological mechanism, shared-embedding gain or SOTA.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "rhea_replay": replay,
        "arms": {
            arm: {key: values[key] for key in (
                "delta_recall1", "corrected", "introduced", "risk_net_lambda2"
            )}
            for arm, values in arm_reports.items()
        },
        "consensus_gates": consensus_gates,
        "portability_gates": portability_gates,
        "portable_catalog_prior": report["portable_catalog_prior"],
        "multi_catalog_incremental": report["multi_catalog_incremental"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
