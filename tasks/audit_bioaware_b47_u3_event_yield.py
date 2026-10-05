#!/usr/bin/env python
"""Build the B47 truth-blind atomic event table and opportunity/null audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation.bioaware import compound_reaction_degree, degree_preserving_reaction_decoy  # noqa: E402
from bioaware_b47_u3_core import (  # noqa: E402
    aggregate_candidate_features, materialize_sample_events,
    permute_candidate_identities_within_query_adduct,
    permute_seed_events_across_samples, query_opportunities,
    reaction_pair_table, safe_participants, summarize_opportunity,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_input_chain(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    seed_report_path = args.seed_dir / "report.json"
    u1c_report_path = args.u1c_dir / "report.json"
    u2_report_path = args.u2_dir / "report.json"
    for path in (
        seed_report_path, u1c_report_path, u2_report_path,
        args.seed_dir / "candidate_scores.csv.gz",
        args.seed_dir / "seeds_primary.csv.gz",
        args.graph_dir / "queries.csv.gz",
        args.graph_dir / "candidate_references.csv.gz",
        args.participants,
        args.participants.parent / "report.json",
        args.protocol_amendment,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_report = json.loads(seed_report_path.read_text(encoding="utf-8"))
    u1c_report = json.loads(u1c_report_path.read_text(encoding="utf-8"))
    u2_report = json.loads(u2_report_path.read_text(encoding="utf-8"))
    if seed_report.get("contracts", {}).get("truth_opened") is not False:
        raise RuntimeError("U3 requires truth-blind seeds")
    if seed_report.get("provenance", {}).get("graph_report_sha256") != sha256_file(
        args.graph_dir / "report.json"
    ):
        raise RuntimeError("U3 graph does not match the graph used to freeze seeds")
    if seed_report.get("provenance", {}).get("candidate_scores_sha256") != sha256_file(
        args.seed_dir / "candidate_scores.csv.gz"
    ):
        raise RuntimeError("candidate-score provenance mismatch")
    if seed_report.get("provenance", {}).get("seeds_primary_sha256") != sha256_file(
        args.seed_dir / "seeds_primary.csv.gz"
    ):
        raise RuntimeError("primary-seed provenance mismatch")
    if u1c_report.get("status") != "bioaware_b47_u1c_seed_denominator_audit_complete":
        raise RuntimeError("U3 requires canonical U1c")
    if u2_report.get("status") != "bioaware_b47_u2_catalog_coverage_complete":
        raise RuntimeError("U3 requires canonical U2")
    if u2_report.get("contracts", {}).get("truth_opened") is not False:
        raise RuntimeError("U2 truth-blind contract changed")
    direction_report = json.loads(
        (args.participants.parent / "report.json").read_text(encoding="utf-8")
    )
    if direction_report.get("status") != "bioaware_rhea_reactome_direction_cache_complete":
        raise RuntimeError("U3 v2 requires the canonical Reactome-direction cache")
    if direction_report.get("provenance", {}).get("participants_sha256") != sha256_file(
        args.participants
    ):
        raise RuntimeError("Reactome-direction participant provenance mismatch")
    # U2 audited the original Rhea connectivity table. U3 may enrich only the
    # direction annotation; it must not silently change the reaction graph.
    u2_rhea_hash = u2_report.get("provenance", {}).get("rhea_participants_sha256")
    direction_source_hash = direction_report.get("provenance", {}).get(
        "source_participants_sha256"
    )
    if not u2_rhea_hash or u2_rhea_hash != direction_source_hash:
        raise RuntimeError(
            "U3 direction cache is not an annotation-only enrichment of the "
            "Rhea graph audited by U2"
        )
    return seed_report, u1c_report, u2_report


def attach_reference_mz(
    scores: pd.DataFrame, candidate_references_path: Path,
) -> pd.DataFrame:
    references = pd.read_csv(
        candidate_references_path,
        usecols=["reference_row", "reference_precursor_mz"],
        dtype={"reference_row": np.int64, "reference_precursor_mz": float},
    ).drop_duplicates()
    if references.groupby("reference_row")["reference_precursor_mz"].nunique().gt(1).any():
        raise RuntimeError("reference row maps to multiple precursor masses")
    reference_mz = references.drop_duplicates("reference_row").set_index("reference_row")[
        "reference_precursor_mz"
    ]
    output = scores.copy()
    output["best_reference_row"] = pd.to_numeric(
        output["best_reference_row"], errors="raise"
    ).astype(np.int64)
    output["candidate_reference_mz"] = output["best_reference_row"].map(reference_mz)
    if output["candidate_reference_mz"].isna().any():
        raise RuntimeError("best reference row missing precursor mass")
    return output


def run_view(
    name: str,
    participants: pd.DataFrame,
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    seeds: pd.DataFrame,
    maximum_degree: int,
    *,
    pairs: pd.DataFrame | None = None,
    pair_audit: dict[str, int] | None = None,
    seed_reference_candidates: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pairs is None:
        pairs, pair_audit = reaction_pair_table(
            participants,
            set(seeds["seed_compound_id"].astype(str)),
            set(candidates["candidate_id"].astype(str)),
            maximum_degree,
        )
    if pair_audit is None:
        pair_audit = {"reused_pair_table": 1, "typed_pairs": int(len(pairs))}
    events = materialize_sample_events(
        candidates, queries, seeds, pairs,
        seed_reference_candidates=seed_reference_candidates,
    )
    features = aggregate_candidate_features(candidates, events)
    opportunities = query_opportunities(features, queries)
    summary = summarize_opportunity(events, opportunities)
    summary["view"] = name
    summary["pair_audit"] = pair_audit
    return summary, events, features, opportunities


def _scalar_null_record(family: str, repeat: int, summary: dict) -> dict:
    """Keep the frozen null table machine-readable and serialization-stable."""

    output = {"null_family": family, "repeat": int(repeat)}
    for key, value in summary.items():
        if key == "by_study" and isinstance(value, dict):
            for study, study_summary in value.items():
                for metric, metric_value in study_summary.items():
                    if isinstance(metric_value, np.generic):
                        metric_value = metric_value.item()
                    if not isinstance(metric_value, (dict, list, tuple, set)):
                        output[f"study__{study}__{metric}"] = metric_value
            continue
        if isinstance(value, (dict, list, tuple, set)):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--u1c-dir", type=Path, required=True)
    parser.add_argument("--u2-dir", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-degree", type=int, default=250)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.maximum_degree <= 0 or args.null_repeats < 20:
        raise ValueError("maximum degree must be positive and at least 20 null repeats are required")
    for name in ("graph_dir", "seed_dir", "u1c_dir", "u2_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.participants = args.participants.resolve()
    args.protocol_amendment = args.protocol_amendment.resolve()
    output = args.output.resolve()
    if output.exists() and not args.preflight_only:
        raise RuntimeError(f"refusing to overwrite U3 result: {output}")
    seed_report, u1c_report, u2_report = verify_input_chain(args)

    queries = pd.read_csv(
        args.graph_dir / "queries.csv.gz",
        usecols=["query_id", "study", "sample", "feature_id", "feature_mz"],
        dtype={"query_id": str, "study": str, "sample": str, "feature_id": str},
    )
    scores = pd.read_csv(args.seed_dir / "candidate_scores.csv.gz", dtype={"query_id": str})
    required_scores = {
        "query_id", "candidate_id", "candidate_formula", "spectral_score",
        "best_reference_row", "best_reference_adduct",
    }
    if not required_scores.issubset(scores.columns):
        raise RuntimeError(f"candidate scores lack {sorted(required_scores-set(scores.columns))}")
    scores = attach_reference_mz(scores, args.graph_dir / "candidate_references.csv.gz")
    seeds = pd.read_csv(args.seed_dir / "seeds_primary.csv.gz", dtype=str)
    for column in ("seed_score", "seed_margin"):
        seeds[column] = pd.to_numeric(seeds[column], errors="raise")
    participants = pd.read_csv(args.participants, low_memory=False)
    semantics = set(participants.get("direction_semantics", pd.Series(dtype=str)).astype(str))
    if not any(value.startswith("reactome_consensus_") for value in semantics):
        raise RuntimeError(
            "U3 v2 requires the Reactome-direction-enriched Rhea cache; "
            "canonical Rhea left/right serialization is not biological direction"
        )
    degree = compound_reaction_degree(safe_participants(participants))
    scores["candidate_catalogue_degree"] = (
        scores["candidate_id"].astype(str).str[:14].str.upper().map(degree).fillna(0).astype(int)
    )

    expected_stage = u1c_report["stages"]["after_sample_candidate_collapse"]
    if len(seeds) != int(expected_stage["query_events"]):
        raise RuntimeError("U3 seed-event denominator does not reproduce U1c")
    if seeds["seed_compound_id"].nunique() != int(expected_stage["candidate_identities"]):
        raise RuntimeError("U3 seed identities do not reproduce U1c")
    if int(u2_report["coverage"]["rhea_safe"]) != seeds["seed_compound_id"].nunique():
        raise RuntimeError("U3 seed identity denominator does not reproduce U2")
    forbidden = ("truth", "phenotype", "disease", "case_control")
    for name, frame in (("queries", queries), ("scores", scores), ("seeds", seeds)):
        bad = [column for column in frame if any(token in column.casefold() for token in forbidden)]
        if bad:
            raise RuntimeError(f"forbidden columns in {name}: {bad}")

    if args.preflight_only:
        print(json.dumps({
            "status": "bioaware_b47_u3_v3_preflight_passed",
            "queries": int(queries["query_id"].nunique()),
            "candidate_rows": int(len(scores)),
            "seed_events": int(len(seeds)),
            "seed_identities": int(seeds["seed_compound_id"].nunique()),
            "participant_rows": int(len(participants)),
            "protocol_version": "U3-v3-reaction-signature-20260922",
        }, indent=2, sort_keys=True), flush=True)
        return

    real_pairs, real_pair_audit = reaction_pair_table(
        participants,
        set(seeds["seed_compound_id"].astype(str)),
        set(scores["candidate_id"].astype(str)),
        args.maximum_degree,
    )
    real, events, features, opportunities = run_view(
        "real_rhea", participants, scores, queries, seeds, args.maximum_degree,
        pairs=real_pairs, pair_audit=real_pair_audit,
    )
    print(
        f"[U3 real] events={real['atomic_events']:,} "
        f"supported_queries={real['event_supported_queries']:,} "
        f"candidate_specific={real['candidate_specific_queries']:,} "
        f"intervention={real['candidate_specific_intervention_queries']:,}",
        flush=True,
    )

    safe = safe_participants(participants)
    noncurrency = safe[~safe["is_currency"].astype(bool)].copy()
    valid_reactions = noncurrency.groupby("reaction_id")["side"].nunique()
    noncurrency = noncurrency[
        noncurrency["reaction_id"].isin(valid_reactions[valid_reactions.eq(2)].index)
    ].copy()
    noncurrency["is_currency"] = False
    null_rows: list[dict] = []
    seed_null_audits: list[dict] = []
    candidate_null_audits: list[dict] = []
    real_structural_opportunity = int(real["candidate_specific_queries"])
    early_stop = None
    for repeat in range(args.null_repeats):
        repeat_started = time.perf_counter()
        print(
            f"[U3 null {repeat + 1}/{args.null_repeats}] degree rewiring started",
            flush=True,
        )
        rewired = degree_preserving_reaction_decoy(
            noncurrency, seed=args.seed + repeat, swaps_per_edge=2,
        )
        rewired_summary, _, _, _ = run_view(
            f"degree_rewired_{repeat}", rewired, scores, queries, seeds,
            args.maximum_degree,
        )
        null_rows.append(_scalar_null_record(
            "degree_rewired", repeat, rewired_summary,
        ))

        print(
            f"[U3 null {repeat + 1}/{args.null_repeats}] seed-context permutation started",
            flush=True,
        )
        permuted_seeds, seed_audit = permute_seed_events_across_samples(
            seeds, degree, args.seed + 1000 + repeat,
        )
        permuted_summary, _, _, _ = run_view(
            f"seed_context_permuted_{repeat}", participants, scores, queries,
            permuted_seeds, args.maximum_degree, pairs=real_pairs,
            pair_audit=real_pair_audit, seed_reference_candidates=scores,
        )
        null_rows.append(_scalar_null_record(
            "seed_context_permuted", repeat, permuted_summary,
        ))
        seed_null_audits.append({"repeat": repeat, **seed_audit})

        print(
            f"[U3 null {repeat + 1}/{args.null_repeats}] candidate-identity diagnostic started",
            flush=True,
        )
        permuted_candidates, candidate_audit = permute_candidate_identities_within_query_adduct(
            scores, args.seed + 2000 + repeat,
        )
        candidate_summary, _, _, _ = run_view(
            f"candidate_identity_permuted_{repeat}", participants,
            permuted_candidates, queries, seeds, args.maximum_degree,
            pairs=real_pairs, pair_audit=real_pair_audit,
            seed_reference_candidates=scores,
        )
        null_rows.append(_scalar_null_record(
            "candidate_identity_permuted", repeat, candidate_summary,
        ))
        candidate_null_audits.append({"repeat": repeat, **candidate_audit})
        print(
            f"[U3 null {repeat + 1}/{args.null_repeats}] "
            f"rewired={rewired_summary['candidate_specific_intervention_queries']:,} "
            f"seed_context={permuted_summary['candidate_specific_intervention_queries']:,} "
            f"wrong_identity={candidate_summary['candidate_specific_intervention_queries']:,} "
            f"seconds={time.perf_counter() - repeat_started:.1f}",
            flush=True,
        )
        current_structural_nulls = {
            "degree_rewired": int(
                rewired_summary["candidate_specific_queries"]
            ),
            "seed_context_permuted": int(
                permuted_summary["candidate_specific_queries"]
            ),
        }
        dominating = {
            family: value for family, value in current_structural_nulls.items()
            if value >= real_structural_opportunity
        }
        if dominating:
            early_stop = {
                "reason": "STRUCTURAL_NULL_DOMINATES_REAL_CANDIDATE_SPECIFIC_OPPORTUNITY",
                "repeat": int(repeat),
                "comparison_metric": "candidate_specific_queries",
                "real_candidate_specific_queries": real_structural_opportunity,
                "dominating_nulls": dominating,
                "scientific_consequence": (
                    "The preregistered real-greater-than-every-null gate can no "
                    "longer pass; remaining repeats cannot reverse this failure."
                ),
            }
            print(f"[U3 fail-fast] {json.dumps(early_stop, sort_keys=True)}", flush=True)
            break
    null_table = pd.DataFrame(null_rows)
    completed_repeats = int(len(seed_null_audits))

    minimum_source_specific = min(
        int(row["candidate_specific_queries"])
        for row in real["by_study"].values()
    ) if real["by_study"] else 0
    minimum_source_intervention = min(
        int(row["candidate_specific_intervention_queries"])
        for row in real["by_study"].values()
    ) if real["by_study"] else 0
    null_families = ("degree_rewired", "seed_context_permuted")
    diagnostic_null_family = "candidate_identity_permuted"
    null_metric = "candidate_specific_queries"
    null_maxima = {
        family: int(null_table.loc[
            null_table["null_family"].eq(family), null_metric
        ].max())
        for family in null_families
    }
    real_specific = int(real[null_metric])
    empirical_p = {
        family: float((1 + int((null_table.loc[
            null_table["null_family"].eq(family), null_metric
        ] >= real_specific).sum())) / (completed_repeats + 1))
        for family in null_families
    }
    worst_null = max(null_maxima.values())
    relative_lift = float((real_specific - worst_null) / max(1, worst_null))
    source_nulls = {}
    for source, source_summary in real["by_study"].items():
        source_column = f"study__{source}__{null_metric}"
        if source_column not in null_table:
            raise RuntimeError(f"structured null table lacks source metric: {source_column}")
        source_real = int(source_summary[null_metric])
        source_maxima = {
            family: int(null_table.loc[
                null_table["null_family"].eq(family), source_column
            ].max())
            for family in null_families
        }
        source_p = {
            family: float((1 + int((null_table.loc[
                null_table["null_family"].eq(family), source_column
            ] >= source_real).sum())) / (completed_repeats + 1))
            for family in null_families
        }
        source_worst = max(source_maxima.values())
        source_nulls[source] = {
            "real": source_real,
            "maximum": source_maxima,
            "empirical_upper_tail_p": source_p,
            "relative_lift_over_worst_null": float(
                (source_real - source_worst) / max(1, source_worst)
            ),
        }
    minimum_seed_move = min(float(row["moved_fraction"]) for row in seed_null_audits)
    minimum_candidate_move = min(float(row["moved_fraction"]) for row in candidate_null_audits)
    diagnostic_identity_null = null_table[
        null_table["null_family"].eq(diagnostic_null_family)
    ]
    minimum_actions_for_three_points = int(math.ceil(0.03 * int(real["queries"])))
    gates = {
        "all_null_families_have_20_repeats": completed_repeats >= 20,
        "event_supported_queries_ge_1000": int(real["event_supported_queries"]) >= 1000,
        "candidate_specific_queries_ge_500": int(real["candidate_specific_queries"]) >= 500,
        "both_sources_ge_200_candidate_specific_queries": minimum_source_specific >= 200,
        "intervention_opportunities_cover_three_point_target": int(
            real["candidate_specific_intervention_queries"]
        ) >= minimum_actions_for_three_points,
        "both_sources_ge_100_intervention_opportunities": minimum_source_intervention >= 100,
        "intervention_candidate_identities_ge_100": int(
            real["intervention_candidate_identities"]
        ) >= 100,
        "intervention_candidate_formulas_ge_100": int(
            real["intervention_candidate_formulas"]
        ) >= 100,
        "effective_intervention_candidate_identities_ge_50": float(
            real["effective_intervention_candidate_identities"]
        ) >= 50,
        "largest_intervention_candidate_fraction_le_0_05": float(
            real["largest_intervention_candidate_fraction"]
        ) <= 0.05,
        "largest_intervention_formula_fraction_le_0_10": float(
            real["largest_intervention_formula_fraction"]
        ) <= 0.10,
        "used_seed_identities_ge_100": int(real["seed_identities_used"]) >= 100,
        "used_reactions_ge_100": int(real["reactions_used"]) >= 100,
        "effective_seed_identities_ge_50": float(real["effective_seed_identities"]) >= 50,
        "largest_seed_event_fraction_le_0_10": float(real["largest_seed_event_fraction"]) <= 0.10,
        "supported_adduct_pair_fraction_ge_0_99": float(
            real["supported_adduct_pair_fraction"]
        ) >= 0.99,
        "formula_mass_pair_fraction_ge_0_99": float(
            real["formula_mass_pair_fraction"]
        ) >= 0.99,
        "identity_mass_contract_fraction_ge_0_99": float(
            real["identity_mass_contract_fraction"]
        ) >= 0.99,
        "reaction_transform_signature_match_fraction_ge_0_90": float(
            real["reaction_transform_signature_match_fraction"]
        ) >= 0.90,
        "eligible_transform_fraction_ge_0_90": float(real["transform_consistent_fraction"]) >= 0.90,
        "real_specific_opportunity_gt_every_structural_null": all(
            real_specific > value for value in null_maxima.values()
        ),
        "real_specific_opportunity_lift_over_worst_null_ge_0_10": relative_lift >= 0.10,
        "all_structural_null_empirical_p_le_0_05": all(
            value <= 0.05 for value in empirical_p.values()
        ),
        "each_source_beats_every_structural_null": all(
            item["real"] > max(item["maximum"].values())
            for item in source_nulls.values()
        ),
        "each_source_null_lift_ge_0_10": all(
            item["relative_lift_over_worst_null"] >= 0.10
            for item in source_nulls.values()
        ),
        "each_source_structural_null_empirical_p_le_0_05": all(
            value <= 0.05
            for item in source_nulls.values()
            for value in item["empirical_upper_tail_p"].values()
        ),
        "seed_context_null_moves_ge_0_50": minimum_seed_move >= 0.50,
        "candidate_identity_null_moves_ge_0_50": minimum_candidate_move >= 0.50,
        "both_sources_present": set(real["by_study"]) == {"ST001122", "ST003356"},
        "truth_and_phenotype_absent": True,
    }

    staging = Path(tempfile.mkdtemp(prefix=".bioaware_b47_u3_", dir=output.parent))
    try:
        events.to_csv(staging / "atomic_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        features.to_csv(staging / "candidate_event_features.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        opportunities.to_csv(staging / "query_event_opportunities.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        null_table.to_csv(staging / "structured_null_summary.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        permutation_audit = pd.concat([
            pd.DataFrame(seed_null_audits).assign(null_family="seed_context_permuted"),
            pd.DataFrame(candidate_null_audits).assign(null_family="candidate_identity_permuted"),
        ], ignore_index=True)
        permutation_audit.to_csv(
            staging / "null_permutation_audit.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        report = {
            "status": "bioaware_b47_u3_truthblind_event_yield_complete",
            "protocol_version": "U3-v3-reaction-signature-20260922",
            "formal": True,
            "truth_blind": True,
            "real_event_yield": real,
            "structured_nulls": {
                "requested_repeats_per_family": int(args.null_repeats),
                "completed_repeats_per_family": completed_repeats,
                "early_stop": early_stop,
                "comparison_metric": null_metric,
                "minimum_interventions_for_three_point_target": minimum_actions_for_three_points,
                "maximum_candidate_specific_queries": null_maxima,
                "empirical_upper_tail_p": empirical_p,
                "real_relative_lift_over_worst_null": relative_lift,
                "by_source": source_nulls,
                "seed_context_permutation_minimum_moved_fraction": minimum_seed_move,
                "candidate_identity_permutation_minimum_moved_fraction": minimum_candidate_move,
                "candidate_identity_control": (
                    "within-query, same-adduct identity/formula/degree permutation; "
                    "spectral score and reference multiplicity remain in place; "
                    "reported as a non-directional identity-spectrum mismatch "
                    "diagnostic, not an event-count null"
                ),
                "candidate_identity_diagnostic": {
                    "candidate_specific_queries_min": int(
                        diagnostic_identity_null["candidate_specific_queries"].min()
                    ),
                    "candidate_specific_queries_median": float(
                        diagnostic_identity_null["candidate_specific_queries"].median()
                    ),
                    "candidate_specific_queries_max": int(
                        diagnostic_identity_null["candidate_specific_queries"].max()
                    ),
                    "intervention_queries_min": int(
                        diagnostic_identity_null[
                            "candidate_specific_intervention_queries"
                        ].min()
                    ),
                    "intervention_queries_median": float(
                        diagnostic_identity_null[
                            "candidate_specific_intervention_queries"
                        ].median()
                    ),
                    "intervention_queries_max": int(
                        diagnostic_identity_null[
                            "candidate_specific_intervention_queries"
                        ].max()
                    ),
                },
                "direction_diagnostic": {
                    "counts": real.get("direction_labels", {}),
                    "note": (
                        "Reactome-consensus direction is reported only as a diagnostic. "
                        "At steady state, observing a product seed can still support a "
                        "substrate candidate, so reverse orientation is not treated as a null."
                    ),
                },
            },
            "gates": {key: bool(value) for key, value in gates.items()},
            "pass_to_frozen_event_ranking_evaluation": all(bool(value) for value in gates.values()),
            "contracts": {
                "truth_opened": False,
                "phenotype_used": False,
                "performance_computed": False,
                "model_fitted": False,
                "P2b_used": False,
                "sample_local_events_only": True,
                "same_query_seed_excluded": True,
                "unchanged_stoichiometric_participants_excluded": True,
                "independent_seed_before_noisy_or": True,
                "baseline_ties_excluded_from_disagreement": True,
                "reference_multiplicity_preserved_in_candidate_null": True,
                "EMRN_used_as_exact_event": False,
                "catalogue_membership_used_as_outcome": False,
            },
            "provenance": {
                "graph_report_sha256": sha256_file(args.graph_dir / "report.json"),
                "seed_report_sha256": sha256_file(args.seed_dir / "report.json"),
                "u1c_report_sha256": sha256_file(args.u1c_dir / "report.json"),
                "u2_report_sha256": sha256_file(args.u2_dir / "report.json"),
                "participants_sha256": sha256_file(args.participants),
                "direction_report_sha256": sha256_file(
                    args.participants.parent / "report.json"
                ),
                "protocol_amendment_sha256": sha256_file(args.protocol_amendment),
                "atomic_events_sha256": sha256_file(staging / "atomic_events.csv.gz"),
                "candidate_features_sha256": sha256_file(staging / "candidate_event_features.csv.gz"),
                "query_opportunities_sha256": sha256_file(staging / "query_event_opportunities.csv.gz"),
                "null_summary_sha256": sha256_file(staging / "structured_null_summary.csv.gz"),
                "null_permutation_audit_sha256": sha256_file(
                    staging / "null_permutation_audit.csv.gz"
                ),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "parameters": {
                "maximum_degree": int(args.maximum_degree),
                "null_repeats": int(args.null_repeats),
                "seed": int(args.seed),
                "minimum_relative_lift_over_worst_null": 0.10,
            },
            "claim_limit": (
                "Truth-blind event opportunity and structured-null audit only. "
                "It does not measure annotation accuracy, BioAware gain, shared-embedding gain, or SOTA."
            ),
        }
        atomic_json(staging / "report.json", report)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
