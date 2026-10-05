#!/usr/bin/env python
"""Audit the B37 catalogue prior against a genuinely separate Rhea graph.

B37's archived ``network_member`` and ``known_log_degree`` columns come from
the MetDNA2 EMRN ``minimum_step == 0`` graph (strict KEGG), not from Rhea.
Some historical builders restricted degree construction to the observed MS1
subgraph, so the archived fields are replayed exactly and a static rebuild is
reported as a separate arm rather than asserted elementwise identical.
This audit therefore treats strict KEGG as the replay baseline, rebuilds it
from the frozen edge file, and separately constructs a currency-filtered Rhea
compound graph.  It distinguishes three questions:

1. does strict KEGG replay exactly;
2. does Rhea provide a portable catalogue prior on its own;
3. does combining Rhea with strict KEGG add paired query-level value.

No sample context, phenotype, P2b score, or shared-embedding update is used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
from audit_bioaware_b12_multicohort_catalog_action import build_universe  # noqa: E402
from audit_bioaware_b36_reaction_specificity_ablation import (  # noqa: E402
    paired_contrast,
    prepare_candidates,
)
from audit_bioaware_b41_cross_catalog_topology import (  # noqa: E402
    EXPECTED_DOMAINS,
    EXPECTED_QUERIES,
    run_nested_loso,
    stable_seed,
    topology_signature,
)
from evaluate_bioaware_b39_m2_fixed_action import atomic_csv_gzip  # noqa: E402


CONSENSUS_FEATURES = [
    "independent_member_count",
    "independent_member_intersection",
    "independent_log_degree_mean",
    "independent_log_degree_min",
]


def bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    values = series.astype(str).str.strip().str.lower()
    bad = ~values.isin({"true", "false", "1", "0"})
    if bad.any():
        raise RuntimeError(f"unrecognized boolean values: {sorted(values[bad].unique())[:5]}")
    return values.isin({"true", "1"})


def rhea_signature(
    participants_path: Path,
    maximum_participants: int = 8,
) -> tuple[set[str], dict[str, float], set[tuple[str, str]], dict[str, Any]]:
    participants = pd.read_csv(participants_path, low_memory=False)
    required = {"compound_id", "reaction_id", "is_currency"}
    if not required.issubset(participants.columns):
        raise RuntimeError(f"Rhea participants missing columns: {sorted(required - set(participants.columns))}")
    local = participants.loc[~bool_values(participants["is_currency"])].copy()
    edges: set[tuple[str, str]] = set()
    reactions_used = 0
    reactions_too_large = 0
    for _, group in local.groupby("reaction_id", sort=False):
        identities = sorted({value.strip() for value in group["compound_id"].astype(str) if value.strip()})
        if len(identities) <= 1:
            continue
        if len(identities) > maximum_participants:
            reactions_too_large += 1
            continue
        reactions_used += 1
        edges.update(itertools.combinations(identities, 2))
    endpoints = pd.Series([node for edge in edges for node in edge], dtype=str)
    degree = endpoints.value_counts().astype(int)
    members = set(degree.index.astype(str))
    log_degree = {str(key): float(np.log1p(value)) for key, value in degree.items()}
    report = {
        "participant_rows": int(len(participants)),
        "noncurrency_rows": int(len(local)),
        "reactions_used": int(reactions_used),
        "reactions_excluded_above_maximum_participants": int(reactions_too_large),
        "maximum_participants": int(maximum_participants),
        "unique_undirected_nonself_edges": int(len(edges)),
        "members": int(len(members)),
        "maximum_degree": int(degree.max()) if len(degree) else 0,
    }
    return members, log_degree, edges, report


def edge_signature(edges_path: Path) -> set[tuple[str, str]]:
    edges = pd.read_csv(edges_path, usecols=["ik14_a", "ik14_b"])
    result: set[tuple[str, str]] = set()
    for left, right in edges[["ik14_a", "ik14_b"]].itertuples(index=False):
        left, right = str(left).strip(), str(right).strip()
        if left and right and left != right:
            result.add(tuple(sorted((left, right))))
    return result


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def add_independent_features(
    frame: pd.DataFrame,
    kegg_edges: Path,
    emrn_edges: Path,
    rhea_participants: Path,
    null_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    candidate = output["candidate_id"].astype(str)
    kegg_members, kegg_degree, kegg_report = topology_signature(kegg_edges)
    emrn_members, emrn_degree, emrn_report = topology_signature(emrn_edges)
    rhea_members, rhea_degree, rhea_edges, rhea_report = rhea_signature(rhea_participants)
    kegg_edge_set = edge_signature(kegg_edges)

    output["archived_strict_kegg_member"] = (
        output["network_member"].astype(float) > 0.5
    ).astype(float)
    output["archived_strict_kegg_log_degree"] = output["known_log_degree"].astype(float)
    output["strict_kegg_member"] = candidate.isin(kegg_members).astype(float)
    output["strict_kegg_log_degree"] = candidate.map(kegg_degree).fillna(0.0).astype(float)
    output["rhea_member"] = candidate.isin(rhea_members).astype(float)
    output["rhea_log_degree"] = candidate.map(rhea_degree).fillna(0.0).astype(float)
    output["emrn_member"] = candidate.isin(emrn_members).astype(float)
    output["emrn_log_degree"] = candidate.map(emrn_degree).fillna(0.0).astype(float)
    output["independent_member_count"] = output["strict_kegg_member"] + output["rhea_member"]
    output["independent_member_intersection"] = output["strict_kegg_member"] * output["rhea_member"]
    output["independent_log_degree_mean"] = 0.5 * (
        output["strict_kegg_log_degree"] + output["rhea_log_degree"]
    )
    output["independent_log_degree_min"] = np.minimum(
        output["strict_kegg_log_degree"], output["rhea_log_degree"]
    )

    archived_rebuild_errors = {
        "member_mismatches": int(
            output["archived_strict_kegg_member"].ne(output["strict_kegg_member"]).sum()
        ),
        "maximum_log_degree_error": float(np.max(np.abs(
            output["archived_strict_kegg_log_degree"].to_numpy(float)
            - output["strict_kegg_log_degree"].to_numpy(float)
        ), initial=0.0)),
    }
    archived_rebuild_errors["exact"] = bool(
        archived_rebuild_errors["member_mismatches"] == 0
        and archived_rebuild_errors["maximum_log_degree_error"] <= 1e-12
    )
    archived_rebuild_errors["candidate_row_mismatch_fraction"] = float(
        (
            output["archived_strict_kegg_member"].ne(output["strict_kegg_member"])
            | ~np.isclose(
                output["archived_strict_kegg_log_degree"],
                output["strict_kegg_log_degree"], atol=1e-12,
            )
        ).mean()
    )

    signature = output[["candidate_id", "truth_formula", *CONSENSUS_FEATURES]].drop_duplicates(
        ["candidate_id", "truth_formula"]
    )
    if signature.duplicated(["candidate_id", "truth_formula"]).any():
        raise RuntimeError("candidate/formula consensus signature is not unique")
    source_lookup = signature.set_index(["truth_formula", "candidate_id"])[CONSENSUS_FEATURES]
    null_reports: list[dict[str, Any]] = []
    for repeat in range(null_repeats):
        assignment: dict[tuple[str, str], str] = {}
        movable = 0
        for formula, local in signature.groupby("truth_formula", sort=True):
            identities = np.sort(local["candidate_id"].astype(str).to_numpy())
            if len(identities) == 1:
                assignment[(str(formula), str(identities[0]))] = str(identities[0])
                continue
            rng = np.random.default_rng(stable_seed(seed, "b42-null", repeat, formula))
            order = identities[rng.permutation(len(identities))]
            shift = 1 + stable_seed(seed, "b42-shift", repeat, formula) % (len(order) - 1)
            sources = np.roll(order, int(shift))
            assignment.update({
                (str(formula), str(destination)): str(source)
                for destination, source in zip(order, sources)
            })
            movable += len(identities)
        assigned = pd.Series([
            assignment[(str(formula), str(candidate_id))]
            for formula, candidate_id in zip(signature["truth_formula"], signature["candidate_id"])
        ], index=signature.index, dtype=str)
        source_keys = pd.MultiIndex.from_arrays(
            [signature["truth_formula"].astype(str), assigned],
            names=["truth_formula", "candidate_id"],
        )
        permuted = source_lookup.loc[source_keys].reset_index(drop=True)
        renamed = {column: f"null{repeat}_{column}" for column in CONSENSUS_FEATURES}
        permuted = permuted.rename(columns=renamed)
        permuted["candidate_id"] = signature["candidate_id"].astype(str).to_numpy()
        permuted["truth_formula"] = signature["truth_formula"].astype(str).to_numpy()
        output = output.merge(permuted, on=["candidate_id", "truth_formula"], validate="many_to_one")
        before = signature.groupby("truth_formula")[CONSENSUS_FEATURES].sum().sort_index().to_numpy(float)
        after = permuted.groupby("truth_formula")[list(renamed.values())].sum().sort_index().to_numpy(float)
        maximum_error = float(np.max(np.abs(before - after), initial=0.0))
        if maximum_error > 1e-9:
            raise RuntimeError(f"formula-preserving B42 null {repeat} changed feature multisets")
        null_reports.append({
            "repeat": repeat,
            "movable_candidate_formula_keys": int(movable),
            "assigned_to_other_identity": int(
                (assigned.to_numpy(str) != signature["candidate_id"].astype(str).to_numpy()).sum()
            ),
            "maximum_formula_feature_sum_error": maximum_error,
        })

    feature_columns = [
        "archived_strict_kegg_member", "archived_strict_kegg_log_degree",
        "strict_kegg_member", "strict_kegg_log_degree", "rhea_member", "rhea_log_degree",
        "emrn_member", "emrn_log_degree", *CONSENSUS_FEATURES,
        *(f"null{repeat}_{column}" for repeat in range(null_repeats) for column in CONSENSUS_FEATURES),
    ]
    if not np.isfinite(output[feature_columns].to_numpy(float)).all():
        raise RuntimeError("B42 catalogue features contain non-finite values")

    source_mapping: dict[str, Any] = {}
    for source, local in output.groupby("source", sort=True):
        truth = local["is_positive"].astype(bool)
        source_mapping[str(source)] = {
            "candidate_rows": int(len(local)),
            "queries": int(local["query_id"].nunique()),
            "strict_kegg_candidate_fraction": float(local["strict_kegg_member"].mean()),
            "rhea_candidate_fraction": float(local["rhea_member"].mean()),
            "emrn_candidate_fraction": float(local["emrn_member"].mean()),
            "strict_kegg_truth_fraction": float(local.loc[truth, "strict_kegg_member"].mean()),
            "rhea_truth_fraction": float(local.loc[truth, "rhea_member"].mean()),
            "emrn_truth_fraction": float(local.loc[truth, "emrn_member"].mean()),
            "kegg_rhea_membership_disagreement_fraction": float(
                local["strict_kegg_member"].ne(local["rhea_member"]).mean()
            ),
        }
    resource_dependence = {
        "strict_kegg": kegg_report,
        "rhea": rhea_report,
        "emrn_extended": emrn_report,
        "strict_kegg_rhea_member_jaccard": jaccard(kegg_members, rhea_members),
        "strict_kegg_rhea_edge_jaccard": jaccard(kegg_edge_set, rhea_edges),
        "strict_kegg_emrn_member_jaccard": jaccard(kegg_members, emrn_members),
        "strict_kegg_emrn_edges_are_subset": bool(kegg_edge_set.issubset(edge_signature(emrn_edges))),
        "scientifically_independent_comparison": "strict KEGG versus currency-filtered Rhea",
        "emrn_dependency_note": "EMRN contains the strict KEGG step-0 graph and is not an independent catalogue replication.",
    }
    return output, {
        "archived_strict_kegg_rebuild": archived_rebuild_errors,
        "resource_dependence": resource_dependence,
        "formula_preserving_nulls": null_reports,
        "source_mapping": source_mapping,
    }


def arm_registry(null_repeats: int) -> dict[str, list[str]]:
    spectral = ["spectral_score"]
    arms = {
        "spectral_only": spectral,
        "strict_kegg_archived_replay": spectral + [
            "archived_strict_kegg_member", "archived_strict_kegg_log_degree"
        ],
        "strict_kegg_rebuilt": spectral + ["strict_kegg_member", "strict_kegg_log_degree"],
        "rhea_independent_topology": spectral + ["rhea_member", "rhea_log_degree"],
        "emrn_extended_topology": spectral + ["emrn_member", "emrn_log_degree"],
        "kegg_rhea_consensus": spectral + CONSENSUS_FEATURES,
        "all_catalog_topologies": spectral + [
            "strict_kegg_member", "strict_kegg_log_degree",
            "rhea_member", "rhea_log_degree", "emrn_member", "emrn_log_degree",
        ],
    }
    for repeat in range(null_repeats):
        arms[f"formula_permuted_consensus_r{repeat:02d}"] = spectral + [
            f"null{repeat}_{column}" for column in CONSENSUS_FEATURES
        ]
    return arms


def exact_archived_transition_replay(
    observed: pd.DataFrame,
    archived_path: Path,
) -> dict[str, Any]:
    archived = pd.read_csv(archived_path, low_memory=False)
    left = archived.loc[archived["arm"].eq("catalog_topology")].copy()
    right = observed.loc[observed["arm"].eq("strict_kegg_archived_replay")].copy()
    columns = [
        "query_id", "source", "baseline_candidate_id", "proposed_candidate_id",
        "final_candidate_id", "baseline_correct", "proposal_unique", "intervene",
        "final_correct", "corrected", "introduced", "delta", "gate_name",
        "proposal_probability", "gate_margin", "gate_probability",
    ]
    joined = left[columns].merge(
        right[columns], on="query_id", how="outer", validate="one_to_one",
        suffixes=("_archived", "_b42"), indicator=True,
    )
    if len(joined) != EXPECTED_QUERIES or not joined["_merge"].eq("both").all():
        raise RuntimeError("B42 archived strict-KEGG transition query set mismatch")
    categorical = [column for column in columns if column not in {
        "query_id", "proposal_probability", "gate_margin", "gate_probability"
    }]
    mismatches = {
        column: int((joined[f"{column}_archived"].astype(str) != joined[f"{column}_b42"].astype(str)).sum())
        for column in categorical
    }
    numeric = {
        column: float(np.max(np.abs(
            joined[f"{column}_archived"].to_numpy(float)
            - joined[f"{column}_b42"].to_numpy(float)
        ), initial=0.0))
        for column in ("proposal_probability", "gate_margin", "gate_probability")
    }
    passed = not any(mismatches.values()) and all(value <= 1e-12 for value in numeric.values())
    report = {
        "queries": int(len(joined)), "categorical_mismatches": mismatches,
        "maximum_numeric_error": numeric, "pass": bool(passed),
    }
    if not passed:
        raise RuntimeError(f"B42 archived strict-KEGG transition replay failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-candidates", type=Path, default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz")
    parser.add_argument("--st-candidates", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz")
    parser.add_argument("--st-queries", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz")
    parser.add_argument("--kgmn-candidates", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz")
    parser.add_argument("--kgmn-seeds", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz")
    parser.add_argument("--kegg-edges", type=Path, default=ROOT / "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz")
    parser.add_argument("--emrn-edges", type=Path, default=ROOT / "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz")
    parser.add_argument("--rhea-participants", type=Path, default=ROOT / "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz")
    parser.add_argument("--b37-dir", type=Path, default=ROOT / "data/validation/bioaware_b37_local_fullcheck_v2_20260913")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.null_repeats != 3 or args.bootstrap_resamples < 10000:
        raise ValueError("formal B42 requires three nulls and at least 10,000 bootstrap resamples")
    archived_path = args.b37_dir / "catalog_ablation_transitions.csv.gz"
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.kegg_edges, args.emrn_edges,
        args.rhea_participants, args.b37_dir / "report.json", archived_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    universe, universe_provenance = build_universe(args)
    candidates = prepare_candidates(universe)
    candidates, mapping_audit = add_independent_features(
        candidates, args.kegg_edges, args.emrn_edges, args.rhea_participants,
        args.null_repeats, args.seed,
    )
    arms = arm_registry(args.null_repeats)
    transitions, selection, arm_reports, folds = run_nested_loso(
        candidates, arms, args.seed, args.bootstrap_resamples
    )
    archived_replay = exact_archived_transition_replay(transitions, archived_path)

    contrast_specs = {
        "strict_kegg_rebuilt_vs_archived": ("strict_kegg_rebuilt", "strict_kegg_archived_replay"),
        "rhea_vs_strict_kegg": ("rhea_independent_topology", "strict_kegg_archived_replay"),
        "emrn_extended_vs_strict_kegg": ("emrn_extended_topology", "strict_kegg_archived_replay"),
        "kegg_rhea_consensus_vs_strict_kegg": ("kegg_rhea_consensus", "strict_kegg_archived_replay"),
        "all_catalogues_vs_strict_kegg": ("all_catalog_topologies", "strict_kegg_archived_replay"),
        **{
            f"consensus_vs_null_r{repeat:02d}": (
                "kegg_rhea_consensus", f"formula_permuted_consensus_r{repeat:02d}"
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

    rhea = arm_reports["rhea_independent_topology"]
    consensus = arm_reports["kegg_rhea_consensus"]
    null_arms = [f"formula_permuted_consensus_r{repeat:02d}" for repeat in range(args.null_repeats)]
    portability_gates = {
        "archived_strict_kegg_replay": bool(archived_replay["pass"]),
        "static_strict_kegg_rebuild_near_exact": bool(
            mapping_audit["archived_strict_kegg_rebuild"]["candidate_row_mismatch_fraction"] < 0.001
        ),
        "rhea_delta_positive": bool(rhea["delta_recall1"] > 0),
        "rhea_formula_ci_low_positive": bool(rhea["formula_cluster_bootstrap"]["ci_low"] > 0),
        "rhea_identity_ci_low_positive": bool(rhea["identity_cluster_bootstrap"]["ci_low"] > 0),
        "rhea_corrected_gt_2x_introduced": bool(rhea["corrected"] > 2 * rhea["introduced"]),
        "rhea_every_domain_nonnegative": every_domain_nonnegative(rhea),
        "consensus_formula_ci_low_positive": bool(consensus["formula_cluster_bootstrap"]["ci_low"] > 0),
        "consensus_beats_each_formula_permuted_null": bool(
            consensus["delta_recall1"] > max(arm_reports[arm]["delta_recall1"] for arm in null_arms)
        ),
    }
    incremental = paired_contrasts["kegg_rhea_consensus_vs_strict_kegg"]
    incremental_by_domain = {
        domain: float(
            arm_reports["kegg_rhea_consensus"]["by_domain"][domain]["delta_recall1"]
            - arm_reports["strict_kegg_archived_replay"]["by_domain"][domain]["delta_recall1"]
        )
        for domain in EXPECTED_DOMAINS
    }
    incremental_gates = {
        "mean_top1_difference_positive": bool(incremental["mean_top1_difference"] > 0),
        "identity_ci_low_positive": bool(incremental["identity_cluster_bootstrap"]["ci_low"] > 0),
        "formula_ci_low_positive": bool(incremental["formula_cluster_bootstrap"]["ci_low"] > 0),
        "left_better_gt_2x_right_better": bool(incremental["left_better"] > 2 * incremental["right_better"]),
        "every_domain_nonnegative": bool(all(value >= -1e-15 for value in incremental_by_domain.values())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "cross_catalog_transitions.csv.gz"
    selection_path = args.output_dir / "inner_selection_ledger.csv.gz"
    features_path = args.output_dir / "candidate_catalog_features.csv.gz"
    atomic_csv_gzip(transitions_path, transitions)
    atomic_csv_gzip(selection_path, selection)
    kept = [
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula", "source",
        "is_positive", "archived_strict_kegg_member", "archived_strict_kegg_log_degree",
        "strict_kegg_member", "strict_kegg_log_degree", "rhea_member", "rhea_log_degree",
        "emrn_member", "emrn_log_degree", *CONSENSUS_FEATURES,
        *(f"null{repeat}_{column}" for repeat in range(args.null_repeats) for column in CONSENSUS_FEATURES),
    ]
    atomic_csv_gzip(features_path, candidates[kept])
    report = {
        "status": "bioaware_b42_independent_catalog_topology_complete",
        "formal": True,
        "protocol": "exact B37 strict-KEGG replay plus independently constructed currency-filtered Rhea topology; six-domain nested LOSO",
        "queries": EXPECTED_QUERIES,
        "semantic_correction": "B37 archived network_member/known_log_degree are strict-KEGG-derived (EMRN minimum_step=0; some builders condition degree on the observed MS1 subgraph), not Rhea.",
        "archived_replay": archived_replay,
        "mapping_audit": mapping_audit,
        "arm_reports": arm_reports,
        "paired_contrasts": paired_contrasts,
        "portability_gates": portability_gates,
        "independent_rhea_portability": bool(all(portability_gates.values())),
        "multi_catalog_incremental": {
            "consensus_vs_strict_kegg_by_domain": incremental_by_domain,
            "gates": incremental_gates,
            "pass": bool(all(incremental_gates.values())),
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
            "strict_kegg_edges": sha256(args.kegg_edges),
            "emrn_edges": sha256(args.emrn_edges),
            "rhea_participants": sha256(args.rhea_participants),
            "b37_report": sha256(args.b37_dir / "report.json"),
            "b37_transitions": sha256(archived_path),
            "cross_catalog_transitions": sha256(transitions_path),
            "inner_selection_ledger": sha256(selection_path),
            "candidate_catalog_features": sha256(features_path),
            "script": sha256(Path(__file__)),
            "scikit_learn_version": sklearn.__version__,
        },
        "claim_limit": "Opened cross-resource candidate-prior audit. A pass can establish portability across strict KEGG and this Rhea construction, not reaction-context propagation, prospective annotation, biology, shared-embedding gain or SOTA.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "semantic_correction": report["semantic_correction"],
        "resource_dependence": mapping_audit["resource_dependence"],
        "arms": {
            arm: {key: values[key] for key in ("delta_recall1", "corrected", "introduced", "risk_net_lambda2")}
            for arm, values in arm_reports.items()
        },
        "portability_gates": portability_gates,
        "independent_rhea_portability": report["independent_rhea_portability"],
        "multi_catalog_incremental": report["multi_catalog_incremental"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
